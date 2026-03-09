"""
app.py (raíz) — Launcher de compatibilidad.

El código principal de la aplicación vive en src/app.py.
Este archivo simplemente lo invoca para mantener compatibilidad
con el comando: python app.py

Uso recomendado (según README):
    python src/app.py
"""
import subprocess
import sys
from pathlib import Path
import streamlit as st
from transformers import pipeline
import torch
import docx
from pypdf import PdfReader

# Configuración de la página
st.set_page_config(
    page_title="Resumidor - Español",
    
    layout="centered"
)

# Estilo personalizado (CSS Premium)
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextArea textarea {
        background-color: #161b22;
        color: #c9d1d9;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.6rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4);
    }
    .result-container {
        background-color: #ffffff;
        color: #1a1a1a;
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #4facfe;
        margin-top: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_summarizer():
    # Modelo Narrativa RoBERTa, muy estable y especializado en español
    model_name = "Narrativa/bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization"
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        
        # Cargamos el modelo forzando low_cpu_mem_usage=False para evitar meta tensors
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            low_cpu_mem_usage=False
        )
        
        summarizer = pipeline(
            "summarization", 
            model=model, 
            tokenizer=tokenizer, 
            device=device
        )
        return summarizer
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        # Fallback simple si el anterior falla
        return pipeline("summarization", model=model_name)

def chunk_text(text, max_chars=1500):
    """Divide el texto en fragmentos que el modelo pueda procesar, respetando párrafos y frases."""
    chunks = []
    current_chunk = ""
    
    # Primero intentamos separar por párrafos
    paragraphs = text.replace('\r', '').split("\n")
    for para in paragraphs:
        if len(para.strip()) == 0:
            continue
            
        if len(current_chunk) + len(para) < max_chars:
            current_chunk += para + "\n"
        else:
            # Si el párrafo actual es más largo que el límite, lo dividimos por frases
            if len(para) > max_chars:
                sentences = para.replace(". ", ".\n").split("\n")
                for sent in sentences:
                    if len(current_chunk) + len(sent) < max_chars:
                        current_chunk += sent + " "
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sent + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n"
                
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    return ""

def main():
    st.title("Resumidor de Texto en Español")
    st.markdown("Genera resúmenes precisos de tus textos de forma totalmente **local**.")

    # Cargar modelo con spinner
    with st.spinner("Cargando modelo de inteligencia artificial... (Solo la primera vez)"):
        summarizer = load_summarizer()

    # Interfaz de entrada
    input_method = st.radio("Método de entrada:", ["Escribir/Pegar texto", "Subir archivo (.txt, .docx, .pdf)"], horizontal=True)
    
    text_input = ""
    if input_method == "Escribir/Pegar texto":
        text_input = st.text_area("Pega aquí el texto que deseas resumir:", height=300, placeholder="Escribe o pega el contenido aquí...")
    else:
        uploaded_file = st.file_uploader("Sube un archivo de texto, Word o PDF", type=["txt", "docx", "pdf"])
        if uploaded_file is not None:
            try:
                text_input = extract_text_from_file(uploaded_file)
                if text_input.strip() == "":
                    st.warning("El archivo parece estar vacío o no se pudo extraer el texto.")
                else:
                    if len(text_input) > 1500:
                        st.error("El archivo contiene más de 1500 caracteres. Máximo permitido: 1500 caracteres.")
                        text_input = ""  # Limpiar para evitar procesamiento
                    else:
                        with st.expander("Ver texto extraído (previsualización)", expanded=False):
                            st.text(text_input[:1000] + ("..." if len(text_input) > 1000 else ""))
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")
                

    col1, col2 = st.columns(2)
    with col1:
        max_len = st.slider("Longitud máxima del resumen", 30, 150, 100)
    with col2:
        min_len = st.slider("Longitud mínima del resumen", 10, 50, 30)

    if st.button("Generar Resumen"):
        if text_input.strip() == "":
            st.warning("Por favor, introduce algún texto para resumir.")
        elif len(text_input) > 1500:
            st.error("El texto es demasiado largo. Máximo permitido: 1500 caracteres.")
        else:
            with st.spinner("Procesando resumen..."):
                try:
                    # Limpiamos y dividimos en fragmentos
                    chunks = chunk_text(text_input, max_chars=1800)
                    
                    if len(chunks) > 1:
                        st.info(f"El texto es largo. Procesando en {len(chunks)} partes...")
                        
                    progress_bar = st.progress(0)
                    all_summaries = []
                    
                    for i, chunk in enumerate(chunks):
                        # Saltamos fragmentos muy pequeños
                        if len(chunk.strip()) < 30:
                            progress_bar.progress((i + 1) / len(chunks))
                            continue
                            
                        # Ajustamos longitud máxima del resumen si el fragmento es corto para evitar errores
                        # Limitamos a 150 para respetar la capacidad del modelo (max 514 tokens)
                        current_max_len = min(150, max_len, max(30, len(chunk.split()) // 2))
                        current_min_len = min(min_len, max(10, current_max_len - 10))
                        
                        try:
                            summary = summarizer(
                                chunk, 
                                max_length=current_max_len, 
                                min_length=current_min_len, 
                                do_sample=False,
                                truncation=True
                            )
                            all_summaries.append(summary[0]["summary_text"])
                        except Exception as inner_e:
                            st.warning(f"No se pudo procesar la parte {i+1}: {inner_e}")
                            
                        progress_bar.progress((i + 1) / len(chunks))
                    
                    progress_bar.empty()
                    
                    if not all_summaries:
                        st.error("No se pudo generar ningún resumen válido. Intenta con otro texto.")
                    else:
                        # Unir los resúmenes de los fragmentos
                        combined_summary = " ".join(all_summaries).strip()
                        
                        # Si hay múltiples resúmenes, crear un resumen final coherente
                        if len(all_summaries) > 1 and len(combined_summary.split()) > 80:
                            try:
                                final_summary = summarizer(
                                    combined_summary, 
                                    max_length=max_len, 
                                    min_length=min_len, 
                                    do_sample=False,
                                    truncation=True
                                )[0]["summary_text"]
                            except Exception as e:
                                st.warning(f"No se pudo crear resumen final coherente: {e}. Mostrando resúmenes combinados.")
                                final_summary = combined_summary
                        else:
                            final_summary = combined_summary
                        
                        st.markdown("### 📝 Resultado:")
                        st.markdown(f'<div class="result-container">{final_summary}</div>', unsafe_allow_html=True)
                        
                        # Opciones extra
                        st.download_button(
                            label="Descargar Resumen",
                            data=final_summary,
                            file_name="resumen.txt",
                            mime="text/plain"
                        )

                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")

    st.markdown("---")
    st.caption("Desarrollado con Hugging Face Transformers & Streamlit | Ejecución Local")

if __name__ == "__main__":
    main()
