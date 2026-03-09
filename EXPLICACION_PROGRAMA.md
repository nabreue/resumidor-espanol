# 📄 Explicación del Programa: Resumidor de Texto en Español

> Explicación técnica y funcional de `app.py`, paso a paso.

---

## 🧠 ¿Qué hace este programa?

Es una aplicación web **local** que resume textos en español usando inteligencia artificial. El usuario puede pegar texto o subir un archivo (`.txt`, `.docx`, `.pdf`) y la app genera un resumen usando un modelo de Hugging Face, todo sin enviar datos a internet.

---

## 📦 Paso 1 — Importaciones de librerías

```python
import subprocess, sys
from pathlib import Path
import streamlit as st
from transformers import pipeline
import torch
import docx
from pypdf import PdfReader
```

| Librería | Función |
|---|---|
| `streamlit` | Crea la interfaz web visual |
| `transformers` | Carga el modelo de IA de Hugging Face |
| `torch` | Detecta si hay GPU disponible para acelerar |
| `docx` | Lee archivos Word (`.docx`) |
| `pypdf` | Lee archivos PDF |

---

## ⚙️ Paso 2 — Configuración de la página

```python
st.set_page_config(page_title="Resumidor - Español", layout="centered")
```

Define el título de la pestaña del navegador y el diseño centrado de la interfaz.

---

## 🎨 Paso 3 — Estilos CSS personalizados

```python
st.markdown("""<style>...""", unsafe_allow_html=True)
```

Aplica un tema oscuro premium a la app:
- Fondo oscuro (`#0e1117`)
- Áreas de texto con borde gris
- Botón con degradado azul animado al pasar el ratón
- Caja de resultado con borde azul destacado

---

## 🤖 Paso 4 — Carga del modelo de IA

```python
@st.cache_resource
def load_summarizer():
    model_name = "Narrativa/bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization"
    ...
```

- **Modelo usado:** `Narrativa/bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization`, especializado en español.
- **`@st.cache_resource`:** Carga el modelo **solo la primera vez** y lo mantiene en memoria. Las siguientes ejecuciones son instantáneas.
- Detecta automáticamente si hay **GPU** (`device=0`) o solo **CPU** (`device=-1`).
- Si falla la carga avanzada, usa un modo de carga simplificado (fallback).

---

## ✂️ Paso 5 — División del texto en fragmentos (`chunk_text`)

```python
def chunk_text(text, max_chars=1500):
    ...
```

Los modelos de IA tienen un límite de tokens que pueden procesar a la vez. Esta función:

1. Separa el texto por **párrafos** (saltos de línea).
2. Si un párrafo es demasiado largo, lo divide por **frases** (puntos).
3. Va acumulando en `current_chunk` hasta llegar al máximo de caracteres.
4. Devuelve una **lista de fragmentos** listos para resumir.

---

## 📂 Paso 6 — Lectura de archivos (`extract_text_from_file`)

```python
def extract_text_from_file(uploaded_file):
    ...
```

Detecta el tipo de archivo subido y extrae el texto:

| Formato | Método de extracción |
|---|---|
| `.txt` | Decodifica bytes directamente a texto |
| `.docx` | Lee párrafos con la librería `python-docx` |
| `.pdf` | Extrae texto página a página con `pypdf` |

---

## 🖥️ Paso 7 — Interfaz principal (`main`)

```python
def main():
    st.title("Resumidor de Texto en Español")
    ...
```

### 7.1 — Carga del modelo con spinner
```python
with st.spinner("Cargando modelo..."):
    summarizer = load_summarizer()
```
Muestra un mensaje de carga mientras el modelo se inicializa.

### 7.2 — Selección del método de entrada
```python
input_method = st.radio("Método de entrada:", ["Escribir/Pegar texto", "Subir archivo"])
```
El usuario elige entre dos modos:
- **Escribir/Pegar:** Un área de texto grande donde pegar el contenido.
- **Subir archivo:** Un botón para cargar `.txt`, `.docx` o `.pdf`. Si el archivo supera **1500 caracteres**, se muestra un error.

### 7.3 — Controles de longitud del resumen
```python
max_len = st.slider("Longitud máxima del resumen", 30, 150, 100)
min_len = st.slider("Longitud mínima del resumen", 10, 50, 30)
```
Dos sliders permiten ajustar la extensión del resumen resultante.

---

## ▶️ Paso 8 — Generación del resumen

Al pulsar el botón **"Generar Resumen"**:

1. **Validación:** comprueba que el texto no esté vacío ni supere 1500 caracteres.
2. **División:** llama a `chunk_text()` para cortar el texto en partes manejables.
3. **Procesamiento por fragmentos:**
   - Para cada fragmento, ajusta `max_length` y `min_length` según el tamaño real del fragmento (para evitar errores del modelo).
   - Llama al `summarizer` y guarda cada resumen parcial en `all_summaries`.
   - Una barra de progreso muestra el avance.
4. **Resumen final:**
   - Si hay varios fragmentos y el resumen combinado es largo (>80 palabras), se hace **un segundo resumen** del conjunto para mayor coherencia.
   - Si no, se unen los resúmenes parciales directamente.
5. **Resultado:** Se muestra en una caja estilizada y se ofrece un botón para **descargar** el resumen como `.txt`.

---

## 🚀 Paso 9 — Punto de entrada

```python
if __name__ == "__main__":
    main()
```

Cuando Streamlit ejecuta el archivo con `python -m streamlit run app.py`, llama directamente a la función `main()`.

---

## 📁 Estructura del proyecto

```
resumidor-espanol/
├── app.py                  ← Código principal (este archivo)
├── run.bat                 ← Script para iniciar la app fácilmente
├── setup.bat               ← Script para instalar dependencias
├── requirements.txt        ← Lista de librerías necesarias
├── src/                    ← Carpeta de código fuente alternativa
├── data/                   ← Datos de ejemplo
└── README.md               ← Instrucciones generales
```

---

## ▶️ Cómo ejecutar

```bat
:: Opción 1: Doble clic en run.bat

:: Opción 2: Terminal
cd d:\resumidor-espanol
python -m streamlit run app.py
```

La aplicación se abre automáticamente en el navegador en `http://localhost:8501`.

---

*Desarrollado con Hugging Face Transformers & Streamlit | Ejecución 100% Local*
