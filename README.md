# Evaluación de Calidad en la Escritura Manuscrita (Calidad y Ortografía)
> **Informe de Proyecto Final de Procesamiento Distribuido**  
> Proyecto de Tesis enfocado en el procesamiento de imágenes, visión por computadora e inteligencia artificial para el análisis de texto manuscrito.

---

## Integrantes
* **Ing. Hestybalyz Jackelyn Fernández Cantú**
* **Ing. Ángela Marisol Meléndez Fuentes**

---

## Descripción del Proyecto
Este proyecto implementa un sistema para la evaluación integral de documentos manuscritos, analizando tanto la **calidad del trazo/escritura** como la **corrección ortográfica**. Mediante técnicas de procesamiento digital de imágenes, modelos de visión e integración con LLMs (OpenAI), la plataforma compara las muestras manuscritas contra transcripciones en texto plano para medir el grado de fidelidad, detectar errores inducidos y procesar la información de forma estructurada a través de una interfaz gráfica desarrollada en PyQt6.

---

## Dataset y Pruebas Realizadas

> **Nota:** Para visualizar las muestras, visita el [DATASET](https://drive.google.com/drive/folders/1TxVmm92baRD8lN8etEUXD6Z8Red8JFLu?usp=sharing)

El conjunto de datos experimental consta de:
* **Muestras:** Alrededor de **550 hojas manuscritas** digitalizadas.
* **Formatos de papel evaluados:** 
  * Hojas en blanco.
  * Hojas rayadas.
  * Hojas cuadriculadas.
* **Estructura del contenido:** Cada hoja contiene un texto único compuesto por **3 párrafos** con errores ortográficos forzados controlados.
* **Ground Truth:** Cada muestra cuenta con su archivo equivalente en `.txt` para realizar el emparejamiento (*matching*) entre el texto digitalizado y el manuscrito, sirviendo como base de comparación para la fidelidad del reconocimiento.

---

## Requisitos del Sistema

### 1. Entorno de Software
* **Python:** `3.11.9`
* **Módulos estándar:** `sys`, `os`, `re`, `csv`, `base64`
* **Librerías externas:**
  * `opencv-python` (`cv2`)
  * `numpy`
  * `openai`
  * `PyQt6` (`PyQt6.QtCore`, `PyQt6.QtGui`, `PyQt6.QtWidgets`)

### 2. Requerimientos de Hardware (Mínimos / Entorno de Prueba)
El sistema fue ejecutado y validado en el siguiente entorno base:
* **Equipo:** Huawei MateBook D14 (2021)
* **Procesador:** Intel Core i3 (11.ª generación)
* **Memoria RAM:** 8 GB
* **Gráficos:** Gráficos integrados (sin GPU dedicada)
* **Almacenamiento:** 256 GB SSD
