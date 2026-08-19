import sys
import os
import re
import csv
import base64
import difflib
import cv2
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from anthropic import Anthropic

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import pytesseract
    _ruta_tesseract = os.environ.get("TESSERACT_CMD")
    if _ruta_tesseract:
        pytesseract.pytesseract.tesseract_cmd = _ruta_tesseract
except ImportError:
    pytesseract = None

# =========================================================
# 1. PREPROCESAMIENTO Y DETECCIÓN DE CAJAS "CLÁSICA" (OpenCV)
#    (se conserva como referencia / tercera pestaña)
# =========================================================
def preprocesar_imagen(img):
    """Limpia la imagen unificando la iluminación antes de binarizar."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fondo_sombras = cv2.GaussianBlur(gray, (101, 101), 0)
    sin_sombras = cv2.addWeighted(gray, 1, fondo_sombras, -1, 255)

    blur = cv2.GaussianBlur(sin_sombras, (5, 5), 0)
    bin_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 15)

    kernel_lineas = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    lineas = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_lineas, iterations=2)
    sin_lineas = cv2.subtract(bin_img, lineas)

    kernel_ruido = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    limpia = cv2.morphologyEx(sin_lineas, cv2.MORPH_OPEN, kernel_ruido, iterations=1)

    h, w = limpia.shape
    margen_x = int(w * 0.08)
    margen_y = int(h * 0.05)
    limpia[:, :margen_x] = 0
    limpia[:, w - margen_x:] = 0
    limpia[:margen_y, :] = 0
    limpia[h - margen_y:, :] = 0

    return limpia


def detectar_cajas_opencv(img_bgr, bin_img):
    """Detección "cruda" por morfología, útil como referencia/backup."""
    img_cajas = img_bgr.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(bin_img, kernel, iterations=2)

    contornos, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cajas = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if 10 < h < 80 and w > 20:
            cajas.append((x, y, w, h))
            cv2.rectangle(img_cajas, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cajas = sorted(cajas, key=lambda b: (b[1] // 30, b[0]))
    return img_cajas, cajas


def conv_img_base64(img_bgr):
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buffer).decode('utf-8')


# =========================================================
# 2. TRANSCRIPCIÓN CON CLAUDE
# =========================================================
PROMPT_TRANSCRIPCION = (
    "Transcribe literalmente el texto manuscrito de la imagen, letra por letra. "
    "Mantén cualquier error ortográfico, letra omitida o palabra mal formada "
    "exactamente como la percibas. Prohibido corregir la gramática o la ortografía. "
    "Devuelve ÚNICAMENTE el texto transcrito, sin comentarios adicionales, "
    "sin markdown, sin comillas."
)


def transcribir_con_claude(imagen_bgr, api_key, modelo="claude-sonnet-5"):
    client = Anthropic(api_key=api_key)
    b64_img = conv_img_base64(imagen_bgr)

    respuesta = client.messages.create(
        model=modelo,
        max_tokens=800,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_img}},
                {"type": "text", "text": PROMPT_TRANSCRIPCION},
            ],
        }],
    )

    partes_texto = [b.text for b in respuesta.content if getattr(b, "type", None) == "text"]
    return "".join(partes_texto).strip()


# =========================================================
# 3. SCORE DE FIDELIDAD (referencia vs transcripción de Claude)
# =========================================================
def limpiar_palabras(texto):
    texto = texto.replace("\n", " ")
    texto = re.sub(r'[^\w\sáéíóúÁÉÍÓÚñÑüÜ]', '', texto)
    return texto.split()


def calcular_fidelidad(texto_ref, texto_trans):
    """
    Devuelve:
      score (0-1): similitud global tipo difflib.ratio()
      opcodes: lista de dicts con tipo ('equal'/'replace'/'delete'/'insert'),
               palabras de referencia y palabras escritas involucradas.
      palabras_ref, palabras_trans: listas tokenizadas.

    Lectura:
      - 'delete'  -> palabra que estaba en la referencia y Claude NO detectó
                     (posible zona no legible / mal trazada).
      - 'insert'  -> palabra que Claude "vio" pero no existe en la referencia
                     (posible alucinación o texto extra).
      - 'replace' -> palabra distinta a la esperada (posible error de trazo
                     o error ortográfico real).
    """
    palabras_ref = limpiar_palabras(texto_ref)
    palabras_trans = limpiar_palabras(texto_trans)

    sm = difflib.SequenceMatcher(None, palabras_ref, palabras_trans, autojunk=False)
    score = sm.ratio()

    opcodes = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        opcodes.append({
            "tipo": tag,
            "ref": palabras_ref[i1:i2],
            "escrito": palabras_trans[j1:j2],
            "pos_ref": (i1, i2),
            "pos_trans": (j1, j2),
        })

    return score, opcodes, palabras_ref, palabras_trans


def generar_reporte_html(score, opcodes, len_ref, len_trans):
    color = {"equal": "#222222", "replace": "#B8860B", "delete": "#C0392B", "insert": "#2471A3"}
    etiqueta = {"equal": "OK", "replace": "SUSTITUCIÓN", "delete": "NO DETECTADO", "insert": "EXTRA / ALUCINADO"}

    html = "<div style='font-family:Consolas,monospace; font-size:13px;'>"
    html += f"<h3>Score de fidelidad: {score * 100:.1f}%</h3>"
    html += f"<p>Palabras esperadas (referencia): {len_ref} &nbsp;|&nbsp; Palabras transcritas (Claude): {len_trans}</p>"
    html += "<hr>"

    for op in opcodes:
        tag = op["tipo"]
        c = color[tag]
        etq = etiqueta[tag]
        ref_txt = " ".join(op["ref"]) if op["ref"] else "—"
        esc_txt = " ".join(op["escrito"]) if op["escrito"] else "—"
        if tag == "equal":
            html += f"<span style='color:{c};'>{ref_txt} </span>"
        else:
            html += (f"<br><span style='background:#FDEDEC; color:{c}; font-weight:bold;'>"
                      f"[{etq}] ref: '{ref_txt}'  ->  escrito: '{esc_txt}'</span><br>")

    html += "</div>"
    return html


# =========================================================
# 4. DETECCIÓN DE CAJAS: EasyOCR y Tesseract + comparación IoU
# =========================================================
_LECTOR_EASYOCR = None


def _obtener_lector_easyocr():
    global _LECTOR_EASYOCR
    if _LECTOR_EASYOCR is None:
        if easyocr is None:
            raise RuntimeError("easyocr no está instalado (pip install easyocr)")
        _LECTOR_EASYOCR = easyocr.Reader(['es'], gpu=False)
    return _LECTOR_EASYOCR


def dividir_cajas_multipalabra(cajas):
    """
    EasyOCR frecuentemente detecta VARIAS palabras dentro de una sola caja
    cuando el espaciado entre ellas es chico (ej. una caja con texto
    'de un niño huérfano' en vez de 4 cajas separadas). Esto divide esa
    caja en una caja por palabra, repartiendo el ancho proporcionalmente
    a la cantidad de letras de cada palabra (una palabra de 8 letras se
    lleva más ancho que una de 2), para que cada palabra quede en su
    propia caja en vez de compartir una caja ancha con otras.
    """
    nuevas = []
    for c in cajas:
        texto = (c.get("texto") or "").strip()
        tokens = texto.split()

        if len(tokens) <= 1:
            nuevas.append(c)
            continue

        longitudes = [max(len(t), 1) for t in tokens]
        total_letras = sum(longitudes)
        x_actual = c["x"]
        ancho_restante = c["w"]

        for i, (tok, n_letras) in enumerate(zip(tokens, longitudes)):
            es_ultimo = (i == len(tokens) - 1)
            if es_ultimo:
                ancho_tok = ancho_restante  # el último se queda con lo que sobre (evita perder px por redondeo)
            else:
                ancho_tok = max(int(round(c["w"] * (n_letras / total_letras))), 1)

            nueva = dict(c)
            nueva["x"] = x_actual
            nueva["w"] = max(ancho_tok, 1)
            nueva["texto"] = tok
            nueva["dividida"] = True
            nuevas.append(nueva)

            x_actual += ancho_tok
            ancho_restante -= ancho_tok

    return nuevas


def _agrupar_en_lineas(cajas):
    """
    Ordena las cajas en orden de lectura (renglón por renglón, izquierda
    a derecha) SIN asumir un tamaño de renglón fijo en píxeles. En vez de
    eso, agrupa cajas consecutivas (ordenadas por el centro vertical) que
    estén separadas por menos del ~65% del alto promedio de esas dos
    cajas. Esto se adapta solo a la resolución real de la imagen (una
    foto de celular puede tener renglones de 80px, un scan de 20px) y
    tolera la inclinación natural de la letra manuscrita.
    """
    if not cajas:
        return cajas

    ordenadas = sorted(cajas, key=lambda c: c["y"] + c["h"] / 2)
    lineas = [[ordenadas[0]]]

    for c in ordenadas[1:]:
        anterior = lineas[-1][-1]
        centro_c = c["y"] + c["h"] / 2
        centro_prev = anterior["y"] + anterior["h"] / 2
        umbral = 0.65 * ((c["h"] + anterior["h"]) / 2)

        if abs(centro_c - centro_prev) <= umbral:
            lineas[-1].append(c)
        else:
            lineas.append([c])

    resultado = []
    for linea in lineas:
        resultado.extend(sorted(linea, key=lambda c: c["x"]))
    return resultado


def detectar_bboxes_easyocr(img_bgr):
    reader = _obtener_lector_easyocr()
    resultados = reader.readtext(img_bgr)
    cajas = []
    for (bbox, texto, conf) in resultados:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        cajas.append({
            "x": x, "y": y, "w": w, "h": h,
            "texto": texto, "conf": float(conf),
            "fuente": "easyocr", "coincide": True, "seleccionada": False,
        })
    cajas = dividir_cajas_multipalabra(cajas)
    cajas = _agrupar_en_lineas(cajas)
    return cajas



def detectar_bboxes_tesseract(img_bgr, lang="spa"):
    if pytesseract is None:
        raise RuntimeError("pytesseract no está instalado (pip install pytesseract)")
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, lang=lang, output_type=pytesseract.Output.DICT)
    cajas = []
    n = len(data["text"])
    for i in range(n):
        texto = data["text"][i].strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if texto and conf > 0:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            cajas.append({"x": x, "y": y, "w": w, "h": h, "texto": texto, "conf": conf, "fuente": "tesseract"})
    return cajas


def _iou(b1, b2):
    xa = max(b1["x"], b2["x"])
    ya = max(b1["y"], b2["y"])
    xb = min(b1["x"] + b1["w"], b2["x"] + b2["w"])
    yb = min(b1["y"] + b1["h"], b2["y"] + b2["h"])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area1 = b1["w"] * b1["h"]
    area2 = b2["w"] * b2["h"]
    return inter / float(area1 + area2 - inter)


def emparejar_cajas(cajas_easyocr, cajas_tesseract, umbral_iou=0.3):
    """
    Marca cada caja de EasyOCR con si tiene o no una contraparte cercana
    en Tesseract (por IoU). Las que NO coinciden son candidatas a
    palabras mal trazadas / no reconocidas de forma consistente,
    y quedan pintadas en rojo en el editor para que el usuario las
    revise y, si aplica, las seleccione para evaluación posterior.
    """
    for ce in cajas_easyocr:
        mejor_iou, mejor_match = 0.0, None
        for ct in cajas_tesseract:
            val = _iou(ce, ct)
            if val > mejor_iou:
                mejor_iou, mejor_match = val, ct
        ce["iou_tesseract"] = mejor_iou
        ce["coincide"] = mejor_iou >= umbral_iou
        ce["texto_tesseract"] = mejor_match["texto"] if mejor_match else ""
    return cajas_easyocr


# =========================================================
# 4bis. EVALUACIÓN ORTOGRÁFICA AUTOMÁTICA
#       (alineamiento caja <-> palabra de referencia, tolera
#        cajas de más/de menos sin desalinear todo lo que sigue)
# =========================================================
UMBRAL_SIMILITUD_TEXTO = 0.85   # 0-1, qué tan parecida debe ser la palabra para darse por "correcta"
UMBRAL_ANOMALIA_ANCHO = 0.40    # % de desviación del ancho esperado para marcar alerta


def evaluar_ortografia_automatica(cajas, texto_referencia):
    """
    En vez de asumir 'caja #i = palabra #i' (que se rompe en cuanto falta
    o sobra UNA caja en cualquier punto), esto ALINEA las dos secuencias
    con difflib.SequenceMatcher, igual que ya se hace para el score de
    fidelidad. La ventaja: si en algún punto falta una caja, el resto
    del documento se puede volver a sincronizar solo, en vez de arrastrar
    el desfase hasta el final.

    Casos que puede reportar:
      - 'equal'/'replace' con caja y palabra emparejadas: se evalúan las
        3 señales (texto, longitud, ancho de caja) igual que antes.
      - Caja sin palabra de referencia cercana: probablemente ruido o
        una caja de más (falso positivo de detección).
      - Palabra de referencia sin ninguna caja: no se detectó nada ahí
        (posible zona no legible); no hay caja que marcar, pero se
        reporta para que sepas que falta agregar una caja manualmente.
    """
    palabras_ref = limpiar_palabras(texto_referencia)
    palabras_ref_low = [p.lower() for p in palabras_ref]

    textos_cajas = []
    for c in cajas:
        tokens = limpiar_palabras((c.get("texto") or "").strip())
        textos_cajas.append(tokens[0].lower() if tokens else "")

    # reset de campos por si se vuelve a correr la evaluación
    for c in cajas:
        c["palabra_esperada"] = None
        c["texto_detectado_norm"] = ""
        c["similitud_ortografia"] = None
        c["anomalia_ancho"] = False
        c["alerta_ortografia"] = False

    sm = difflib.SequenceMatcher(None, textos_cajas, palabras_ref_low, autojunk=False)
    opcodes = sm.get_opcodes()

    # ancho promedio por letra, estimado SOLO con los emparejamientos
    # 'equal' (los más confiables), para juzgar anomalías de ancho
    muestras_ancho = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for off in range(i2 - i1):
                n = len(palabras_ref[j1 + off])
                if n > 0:
                    muestras_ancho.append(cajas[i1 + off]["w"] / n)
    ancho_prom = (sum(muestras_ancho) / len(muestras_ancho)) if muestras_ancho else None

    alertas = []

    def evaluar_par(idx_caja, idx_ref, forzar_alerta=False):
        c = cajas[idx_caja]
        palabra_esperada = palabras_ref[idx_ref]
        texto_norm = textos_cajas[idx_caja]

        similitud = difflib.SequenceMatcher(None, texto_norm, palabra_esperada.lower()).ratio()
        diff_longitud = abs(len(texto_norm) - len(palabra_esperada))
        anomalia_ancho = False
        if ancho_prom:
            ancho_esperado = ancho_prom * len(palabra_esperada)
            if ancho_esperado > 0:
                anomalia_ancho = abs(c["w"] - ancho_esperado) / ancho_esperado > UMBRAL_ANOMALIA_ANCHO

        alerta = (
            forzar_alerta
            or similitud < UMBRAL_SIMILITUD_TEXTO
            or diff_longitud > 0
            or anomalia_ancho
            or (not c.get("coincide", True))
        )

        c["palabra_esperada"] = palabra_esperada
        c["texto_detectado_norm"] = texto_norm
        c["similitud_ortografia"] = round(similitud, 3)
        c["anomalia_ancho"] = anomalia_ancho
        c["alerta_ortografia"] = bool(alerta)
        if alerta:
            c["seleccionada"] = True

        alertas.append({
            "tipo": "par", "idx": idx_caja + 1, "esperada": palabra_esperada,
            "detectada": texto_norm, "similitud": similitud, "diff_longitud": diff_longitud,
            "anomalia_ancho": anomalia_ancho, "coincide_ocr": c.get("coincide", True), "alerta": alerta,
        })

    def caja_extra(idx_caja):
        c = cajas[idx_caja]
        c["alerta_ortografia"] = True
        c["seleccionada"] = True
        alertas.append({
            "tipo": "caja_extra", "idx": idx_caja + 1, "esperada": None,
            "detectada": textos_cajas[idx_caja], "alerta": True,
        })

    def palabra_sin_caja(idx_ref):
        alertas.append({
            "tipo": "palabra_faltante", "idx": None, "esperada": palabras_ref[idx_ref],
            "detectada": None, "alerta": True,
        })

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for off in range(i2 - i1):
                evaluar_par(i1 + off, j1 + off)
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            for off in range(n):
                evaluar_par(i1 + off, j1 + off, forzar_alerta=True)
            for off in range(n, i2 - i1):
                caja_extra(i1 + off)
            for off in range(n, j2 - j1):
                palabra_sin_caja(j1 + off)
        elif tag == "delete":
            for off in range(i2 - i1):
                caja_extra(i1 + off)
        elif tag == "insert":
            for off in range(j2 - j1):
                palabra_sin_caja(j1 + off)

    aviso_desajuste = None
    if len(cajas) != len(palabras_ref):
        aviso_desajuste = (
            f"Hay {len(cajas)} cajas y {len(palabras_ref)} palabras en la referencia "
            f"(diferencia de {abs(len(cajas) - len(palabras_ref))}). Gracias al alineamiento "
            f"esto ya no desincroniza todo lo que sigue, pero sí quedan sin evaluar las "
            f"palabras/cajas que no encontraron pareja (ve el detalle abajo)."
        )

    return cajas, alertas, aviso_desajuste


def generar_reporte_ortografia_html(alertas, aviso_desajuste):
    html = "<div style='font-family:Consolas,monospace; font-size:13px;'>"
    html += "<h3>Revisión ortográfica automática (por alineamiento)</h3>"

    if aviso_desajuste:
        html += f"<p style='color:#C0392B; font-weight:bold;'>{aviso_desajuste}</p><hr>"

    pares = [a for a in alertas if a["tipo"] == "par"]
    extra = [a for a in alertas if a["tipo"] == "caja_extra"]
    faltantes = [a for a in alertas if a["tipo"] == "palabra_faltante"]
    marcadas = [a for a in pares if a["alerta"]]

    html += (
        f"<p>{len(marcadas)} de {len(pares)} palabra(s) emparejadas quedaron marcadas para revisar "
        f"&nbsp;|&nbsp; {len(extra)} caja(s) sin palabra correspondiente "
        f"&nbsp;|&nbsp; {len(faltantes)} palabra(s) sin ninguna caja detectada.</p><hr>"
    )

    for a in marcadas:
        motivos = []
        if a["similitud"] < UMBRAL_SIMILITUD_TEXTO:
            motivos.append(f"texto no coincide (similitud {a['similitud']*100:.0f}%)")
        if a["diff_longitud"]:
            motivos.append(f"diferencia de {a['diff_longitud']} letra(s)")
        if a["anomalia_ancho"]:
            motivos.append("ancho de la caja fuera de lo esperado")
        if not a["coincide_ocr"]:
            motivos.append("EasyOCR y Tesseract no coinciden ahí")

        html += (
            f"<br><span style='background:#FDEDEC; color:#941E1E; font-weight:bold;'>"
            f"[caja #{a['idx']:03d}] esperaba: '{a['esperada']}'  ->  detectado: '{a['detectada']}' "
            f"&nbsp;({', '.join(motivos)})</span><br>"
        )

    if extra:
        html += "<hr><p><b>Cajas sin palabra correspondiente en la referencia</b> (revisa si son ruido o sobran):</p>"
        for a in extra:
            html += f"<span style='color:#6C3483;'>[caja #{a['idx']:03d}] detectado: '{a['detectada']}'</span><br>"

    if faltantes:
        html += "<hr><p><b>Palabras de la referencia sin ninguna caja detectada</b> (agrégalas a mano en el editor si quieres evaluarlas):</p>"
        for a in faltantes:
            html += f"<span style='color:#1A5276;'>esperaba: '{a['esperada']}'</span><br>"

    html += "</div>"
    return html


# =========================================================
# 5. HILOS DE TRABAJO (para no congelar la UI)
# =========================================================
class WorkerTranscripcion(QThread):
    progreso = pyqtSignal(str)
    terminado = pyqtSignal(str, float, list, int, int, str)  # texto, score, opcodes, len_ref, len_trans, error

    def __init__(self, imagen, texto_referencia, api_key, modelo="claude-sonnet-5"):
        super().__init__()
        self.imagen = imagen
        self.texto_referencia = texto_referencia
        self.api_key = api_key
        self.modelo = modelo

    def run(self):
        try:
            self.progreso.emit("Enviando imagen a Claude...")
            texto_trans = transcribir_con_claude(self.imagen, self.api_key, self.modelo)

            self.progreso.emit("Calculando score de fidelidad...")
            score, opcodes, palabras_ref, palabras_trans = calcular_fidelidad(self.texto_referencia, texto_trans)

            self.terminado.emit(texto_trans, score, opcodes, len(palabras_ref), len(palabras_trans), "")
        except Exception as e:
            self.terminado.emit("", 0.0, [], 0, 0, str(e))


class WorkerDeteccionCajas(QThread):
    progreso = pyqtSignal(str)
    terminado = pyqtSignal(list, list, str)  # cajas_easyocr, cajas_tesseract, error

    def __init__(self, imagen):
        super().__init__()
        self.imagen = imagen

    def run(self):
        try:
            self.progreso.emit("Detectando con EasyOCR y separando cajas multi-palabra...")
            cajas_easyocr = detectar_bboxes_easyocr(self.imagen)

            self.progreso.emit("Detectando con Tesseract...")
            cajas_tesseract = detectar_bboxes_tesseract(self.imagen)

            self.progreso.emit("Comparando coordenadas (IoU)...")
            cajas_easyocr = emparejar_cajas(cajas_easyocr, cajas_tesseract)

            self.terminado.emit(cajas_easyocr, cajas_tesseract, "")
        except Exception as e:
            self.terminado.emit([], [], str(e))


# =========================================================
# 6. WIDGETS DE VISUALIZACIÓN
# =========================================================
class VisorImagen(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.setLineWidth(2)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 400)


class VisorCajasEditable(QtWidgets.QLabel):
    """
    Muestra la imagen con las cajas de EasyOCR encima y permite:
      - Click derecho sobre una caja  -> eliminarla
      - Shift + click izquierdo       -> seleccionar/deseleccionar
                                          (para evaluación posterior)
      - Click y arrastrar (sin Shift) -> dibujar una caja nueva
    Verde = coincide con Tesseract | Rojo = no coincide | Naranja = seleccionada
    """
    cajasModificadas = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)

        self.image_bgr = None
        self.cajas = []
        self.escala = 1.0
        self.offset = (0, 0)
        self.punto_inicio = None

    def cargar(self, img_bgr, cajas):
        self.image_bgr = img_bgr
        self.cajas = cajas
        self.redibujar()

    def _img_a_widget(self, x, y):
        return x * self.escala + self.offset[0], y * self.escala + self.offset[1]

    def _widget_a_img(self, x, y):
        if self.escala == 0:
            return 0, 0
        return int((x - self.offset[0]) / self.escala), int((y - self.offset[1]) / self.escala)

    def _caja_bajo_cursor(self, x_img, y_img):
        # recorre de la última a la primera para priorizar cajas dibujadas encima
        for i in range(len(self.cajas) - 1, -1, -1):
            c = self.cajas[i]
            if c["x"] <= x_img <= c["x"] + c["w"] and c["y"] <= y_img <= c["y"] + c["h"]:
                return i
        return None

    def redibujar(self, rect_temporal=None):
        if self.image_bgr is None:
            return
        h, w = self.image_bgr.shape[:2]
        lbl_w, lbl_h = max(self.width(), 1), max(self.height(), 1)
        self.escala = min(lbl_w / w, lbl_h / h)
        nuevo_w, nuevo_h = max(int(w * self.escala), 1), max(int(h * self.escala), 1)
        self.offset = ((lbl_w - nuevo_w) // 2, (lbl_h - nuevo_h) // 2)

        img_dibujo = self.image_bgr.copy()
        for c in self.cajas:
            if c.get("seleccionada"):
                color = (0, 140, 255)   # naranja (BGR)
            elif c.get("coincide", True):
                color = (0, 200, 0)     # verde
            else:
                color = (0, 0, 255)     # rojo
            cv2.rectangle(img_dibujo, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), color, 2)

        if rect_temporal is not None:
            (x0, y0), (x1, y1) = rect_temporal
            cv2.rectangle(img_dibujo, (x0, y0), (x1, y1), (255, 0, 255), 1)

        rgb = cv2.cvtColor(img_dibujo, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(nuevo_w, nuevo_h, Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(pix)

    def mousePressEvent(self, event):
        if self.image_bgr is None:
            return
        x_img, y_img = self._widget_a_img(event.position().x(), event.position().y())

        if event.button() == Qt.MouseButton.RightButton:
            idx = self._caja_bajo_cursor(x_img, y_img)
            if idx is not None:
                del self.cajas[idx]
                self.redibujar()
                self.cajasModificadas.emit()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                idx = self._caja_bajo_cursor(x_img, y_img)
                if idx is not None:
                    self.cajas[idx]["seleccionada"] = not self.cajas[idx].get("seleccionada", False)
                    self.redibujar()
                    self.cajasModificadas.emit()
            else:
                self.punto_inicio = (x_img, y_img)

    def mouseMoveEvent(self, event):
        if self.punto_inicio is not None:
            x_img, y_img = self._widget_a_img(event.position().x(), event.position().y())
            self.redibujar(rect_temporal=(self.punto_inicio, (x_img, y_img)))

    def mouseReleaseEvent(self, event):
        if self.punto_inicio is not None and event.button() == Qt.MouseButton.LeftButton:
            x_img, y_img = self._widget_a_img(event.position().x(), event.position().y())
            x0, y0 = self.punto_inicio
            x, y = min(x0, x_img), min(y0, y_img)
            w, h = abs(x_img - x0), abs(y_img - y0)
            if w > 4 and h > 4:
                self.cajas.append({
                    "x": x, "y": y, "w": w, "h": h, "texto": "",
                    "fuente": "manual", "coincide": True, "seleccionada": False,
                })
                self.cajasModificadas.emit()
            self.punto_inicio = None
            self.redibujar()

    def resizeEvent(self, event):
        self.redibujar()
        super().resizeEvent(event)


# =========================================================
# 7. VENTANA PRINCIPAL
# =========================================================
class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Evaluador de Caligrafía y Ortografía - Base v2 (Claude + EasyOCR + Tesseract)")
        self.resize(1500, 900)

        self.OpenCV_image = None
        self.texto_referencia = ""
        self.cajas_opencv = []
        self.cajas_easyocr = []
        self.cajas_tesseract = []
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        self.crear_widgets()
        self.configurar_layout()
        self.conectar_senales()

    # ---------- UI ----------
    def crear_widgets(self):
        estilo_btn = "font-weight: bold; border-radius: 5px; font-size: 13px; min-height: 42px;"

        self.btnCargarTxt = QtWidgets.QPushButton("1. Cargar Referencia (.txt)")
        self.btnCargarTxt.setStyleSheet(f"background-color:#FFF2CC; color:#B38600; border:1px solid #B38600; {estilo_btn}")

        self.btnCargarImg = QtWidgets.QPushButton("2. Cargar Manuscrito (.jpg)")
        self.btnCargarImg.setStyleSheet(f"background-color:#D5FFCC; color:#4D941E; border:1px solid #4D941E; {estilo_btn}")

        self.btnTranscribir = QtWidgets.QPushButton("3. Transcribir con Claude + Score")
        self.btnTranscribir.setStyleSheet(f"background-color:#CCEDFF; color:#1E5C94; border:1px solid #1E5C94; {estilo_btn}")
        self.btnTranscribir.setEnabled(False)

        self.btnDetectarCajas = QtWidgets.QPushButton("4. Detectar cajas (EasyOCR + Tesseract)")
        self.btnDetectarCajas.setStyleSheet(f"background-color:#E6E6FA; color:#4B0082; border:1px solid #4B0082; {estilo_btn}")
        self.btnDetectarCajas.setEnabled(False)

        self.btnEvaluarOrtografia = QtWidgets.QPushButton("5. Evaluar Ortografía (auto)")
        self.btnEvaluarOrtografia.setStyleSheet(f"background-color:#FCE4EC; color:#AD1457; border:1px solid #AD1457; {estilo_btn}")
        self.btnEvaluarOrtografia.setEnabled(False)

        self.btnExportarSeleccion = QtWidgets.QPushButton("6. Exportar seleccionadas (CSV)")
        self.btnExportarSeleccion.setStyleSheet(f"background-color:#FFE0E0; color:#941E1E; border:1px solid #941E1E; {estilo_btn}")
        self.btnExportarSeleccion.setEnabled(False)

        self.tabs_imagenes = QtWidgets.QTabWidget()
        self.tabs_imagenes.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Weight.Bold))

        self.viewer_orig = VisorImagen()
        self.viewer_bin = VisorImagen()
        self.viewer_opencv = VisorImagen()
        self.viewer_cajas = VisorCajasEditable()

        self.tabs_imagenes.addTab(self.viewer_orig, "1. Original")
        self.tabs_imagenes.addTab(self.viewer_bin, "2. Binarización")
        self.tabs_imagenes.addTab(self.viewer_opencv, "3. Cajas OpenCV (ref.)")
        self.tabs_imagenes.addTab(self.viewer_cajas, "4. Editor EasyOCR vs Tesseract")

        self.visorReporte = QtWidgets.QTextEdit()
        self.visorReporte.setReadOnly(True)
        self.visorReporte.setFont(QtGui.QFont("Consolas", 11))
        self.visorReporte.setStyleSheet("background-color:#F8F9FA; border:1px solid #ccc;")

        self.lbl_leyenda = QtWidgets.QLabel(
            "Editor de cajas -> Click+arrastrar: agregar | Click derecho: eliminar | "
            "Shift+click: seleccionar manualmente.  Verde = coincide con Tesseract | "
            "Rojo = no coincide | Naranja = a revisar (manual o marcada por '5. Evaluar Ortografía'). "
            "Antes de evaluar ortografía, ajusta las cajas para que su cantidad coincida con las "
            "palabras de la referencia (el mapeo es por orden de lectura)."
        )
        self.lbl_leyenda.setWordWrap(True)
        self.lbl_leyenda.setStyleSheet("color:#555; font-size:11px;")

        self.lbl_estado = QtWidgets.QLabel("Esperando archivos...")
        self.lbl_estado.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Weight.Bold))

    def configurar_layout(self):
        layout_principal = QtWidgets.QVBoxLayout(self)

        layout_botones = QtWidgets.QHBoxLayout()
        for b in (self.btnCargarTxt, self.btnCargarImg, self.btnTranscribir,
                  self.btnDetectarCajas, self.btnEvaluarOrtografia, self.btnExportarSeleccion):
            layout_botones.addWidget(b)
        layout_principal.addLayout(layout_botones)

        layout_centro = QtWidgets.QHBoxLayout()
        col_izq = QtWidgets.QVBoxLayout()
        col_izq.addWidget(self.tabs_imagenes, 1)
        col_izq.addWidget(self.lbl_leyenda)
        layout_centro.addLayout(col_izq, 1)

        layout_texto = QtWidgets.QVBoxLayout()
        lbl_res = QtWidgets.QLabel("Reporte de fidelidad / detección:")
        lbl_res.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Weight.Bold))
        layout_texto.addWidget(lbl_res)
        layout_texto.addWidget(self.visorReporte)
        layout_centro.addLayout(layout_texto, 1)

        layout_principal.addLayout(layout_centro)
        layout_principal.addWidget(self.lbl_estado)

    def conectar_senales(self):
        self.btnCargarTxt.clicked.connect(self.cargar_txt)
        self.btnCargarImg.clicked.connect(self.cargar_img)
        self.btnTranscribir.clicked.connect(self.iniciar_transcripcion)
        self.btnDetectarCajas.clicked.connect(self.iniciar_deteccion_cajas)
        self.btnEvaluarOrtografia.clicked.connect(self.iniciar_evaluacion_ortografia)
        self.btnExportarSeleccion.clicked.connect(self.exportar_seleccion_csv)
        self.viewer_cajas.cajasModificadas.connect(self.actualizar_estado_seleccion)

    # ---------- Carga de archivos ----------
    def cargar_txt(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Seleccionar Texto de Referencia", ".", "Text Files (*.txt)")
        if path:
            with open(path, "r", encoding="utf-8") as f:
                self.texto_referencia = f.read().strip()
            self.visorReporte.setPlainText(f"[TEXTO DE REFERENCIA CARGADO]\n{self.texto_referencia}\n\nEsperando imagen...")
            self.validar_ejecucion()

    def cargar_img(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Seleccionar Imagen Manuscrita", ".", "Images (*.jpg *.png *.jpeg)")
        if path:
            self.OpenCV_image = cv2.imread(path)
            self.actualizar_pixmap(self.viewer_orig, self.OpenCV_image)

            bin_img = preprocesar_imagen(self.OpenCV_image)
            self.actualizar_pixmap(self.viewer_bin, cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR))

            img_cajas, self.cajas_opencv = detectar_cajas_opencv(self.OpenCV_image, bin_img)
            self.actualizar_pixmap(self.viewer_opencv, img_cajas)

            self.lbl_estado.setText(f"Imagen cargada. {len(self.cajas_opencv)} cajas detectadas por OpenCV (referencia).")
            self.btnDetectarCajas.setEnabled(True)
            self.validar_ejecucion()

    def validar_ejecucion(self):
        if self.texto_referencia and self.OpenCV_image is not None:
            self.btnTranscribir.setEnabled(True)

    # ---------- Transcripción + Score ----------
    def iniciar_transcripcion(self):
        if not self.api_key:
            QtWidgets.QMessageBox.warning(self, "Error", "Falta la variable de entorno ANTHROPIC_API_KEY.")
            return

        self.btnTranscribir.setEnabled(False)
        self.lbl_estado.setText("Iniciando transcripción...")

        self.worker_trans = WorkerTranscripcion(self.OpenCV_image, self.texto_referencia, self.api_key)
        self.worker_trans.progreso.connect(self.lbl_estado.setText)
        self.worker_trans.terminado.connect(self.finalizar_transcripcion)
        self.worker_trans.start()

    def finalizar_transcripcion(self, texto_trans, score, opcodes, len_ref, len_trans, error):
        self.btnTranscribir.setEnabled(True)
        if error:
            self.lbl_estado.setText("Error en la transcripción.")
            self.visorReporte.append(f"\n[ERROR]: {error}")
            return

        self.lbl_estado.setText(f"Transcripción completa. Score de fidelidad: {score * 100:.1f}%")
        cabecera = f"<b>--- TRANSCRIPCIÓN CLAUDE ---</b><br>{texto_trans}<br><br>"
        self.visorReporte.setHtml(cabecera + generar_reporte_html(score, opcodes, len_ref, len_trans))

    # ---------- Detección de cajas EasyOCR / Tesseract ----------
    def iniciar_deteccion_cajas(self):
        self.btnDetectarCajas.setEnabled(False)
        self.lbl_estado.setText("Iniciando detección de cajas...")

        self.worker_det = WorkerDeteccionCajas(self.OpenCV_image)
        self.worker_det.progreso.connect(self.lbl_estado.setText)
        self.worker_det.terminado.connect(self.finalizar_deteccion_cajas)
        self.worker_det.start()

    def finalizar_deteccion_cajas(self, cajas_easyocr, cajas_tesseract, error):
        self.btnDetectarCajas.setEnabled(True)
        if error:
            self.lbl_estado.setText("Error en la detección de cajas.")
            self.visorReporte.append(f"\n[ERROR DETECCIÓN]: {error}")
            return

        self.cajas_easyocr = cajas_easyocr
        self.cajas_tesseract = cajas_tesseract
        self.viewer_cajas.cargar(self.OpenCV_image, self.cajas_easyocr)
        self.tabs_imagenes.setCurrentIndex(3)

        no_coinciden = sum(1 for c in cajas_easyocr if not c.get("coincide", True))
        self.lbl_estado.setText(
            f"EasyOCR: {len(cajas_easyocr)} cajas | Tesseract: {len(cajas_tesseract)} cajas | "
            f"{no_coinciden} sin coincidencia (en rojo)."
        )
        self.btnExportarSeleccion.setEnabled(True)
        self.btnEvaluarOrtografia.setEnabled(True)

    def iniciar_evaluacion_ortografia(self):
        """
        Mapea las cajas actuales (en el orden en que quedaron tras tus
        ediciones) contra las palabras del .txt de referencia, y marca
        automáticamente cuáles necesitan revisión (texto distinto,
        longitud distinta, o ancho de caja anómalo). No requiere que
        selecciones nada a mano primero.
        """
        if not self.texto_referencia:
            QtWidgets.QMessageBox.warning(self, "Aviso", "Falta cargar el texto de referencia (.txt).")
            return

        cajas = self.viewer_cajas.cajas
        if not cajas:
            QtWidgets.QMessageBox.warning(self, "Aviso", "No hay cajas para evaluar. Detecta o agrega cajas primero.")
            return

        cajas, alertas, aviso_desajuste = evaluar_ortografia_automatica(cajas, self.texto_referencia)
        self.viewer_cajas.cajas = cajas
        self.viewer_cajas.redibujar()

        self.visorReporte.setHtml(generar_reporte_ortografia_html(alertas, aviso_desajuste))

        n_alerta = sum(1 for a in alertas if a["alerta"])
        if aviso_desajuste:
            self.lbl_estado.setText(f"Ojo: cantidad de cajas y palabras no coincide. {n_alerta} marcadas para revisar.")
        else:
            self.lbl_estado.setText(f"{n_alerta} palabra(s) marcadas para revisar (en naranja).")

    def actualizar_estado_seleccion(self):
        n_sel = sum(1 for c in self.viewer_cajas.cajas if c.get("seleccionada"))
        self.lbl_estado.setText(f"{n_sel} palabra(s) seleccionada(s) para evaluación posterior.")

    # ---------- Exportación ----------
    def exportar_seleccion_csv(self):
        cajas = self.viewer_cajas.cajas
        if not cajas:
            QtWidgets.QMessageBox.warning(self, "Aviso", "No hay cajas para exportar.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Guardar cajas", "cajas_evaluacion.csv", "CSV Files (*.csv)")
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "X", "Y", "W", "H", "Texto_EasyOCR", "Texto_Tesseract_cercano",
                                  "IoU_con_Tesseract", "Coincide", "Fuente", "Seleccionada_para_evaluar"])
                for i, c in enumerate(cajas):
                    writer.writerow([
                        i + 1, c["x"], c["y"], c["w"], c["h"],
                        c.get("texto", ""), c.get("texto_tesseract", ""),
                        round(c.get("iou_tesseract", 0.0), 3),
                        c.get("coincide", ""), c.get("fuente", ""),
                        c.get("seleccionada", False),
                    ])
            self.lbl_estado.setText(f"Exportado correctamente a {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error al exportar", str(e))

    # ---------- Utilidades ----------
    def actualizar_pixmap(self, label, image):
        if image is None:
            return
        h, w, _ = image.shape
        q_img = QImage(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).data, w, h, 3 * w, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img).scaled(
            label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        if self.OpenCV_image is not None:
            self.actualizar_pixmap(self.viewer_orig, self.OpenCV_image)
            bin_img = preprocesar_imagen(self.OpenCV_image)
            self.actualizar_pixmap(self.viewer_bin, cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR))
            img_cajas, _ = detectar_cajas_opencv(self.OpenCV_image, bin_img)
            self.actualizar_pixmap(self.viewer_opencv, img_cajas)
        super().resizeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())