import streamlit as st
from transformers import pipeline
import torch

# Configuración de la página
st.set_page_config(
    page_title="Resumidor Pro - Español",
    page_icon="🤖",
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

def main():
    st.title("✨ Resumidor de Texto en Español")
    st.markdown("Genera resúmenes precisos de tus textos de forma totalmente **local**.")

    # Cargar modelo con spinner
    with st.spinner("Cargando modelo de inteligencia artificial... (Solo la primera vez)"):
        summarizer = load_summarizer()

    # Interfaz de entrada
    text_input = st.text_area("Pega aquí el texto que deseas resumir:", height=300, placeholder="Escribe o pega el contenido aquí...")
    
    col1, col2 = st.columns(2)
    with col1:
        max_len = st.slider("Longitud máxima del resumen", 30, 200, 100)
    with col2:
        min_len = st.slider("Longitud mínima del resumen", 10, 50, 30)

    if st.button("🚀 Generar Resumen"):
        if text_input.strip() == "":
            st.warning("Por favor, introduce algún texto para resumir.")
        else:
            with st.spinner("Procesando resumen..."):
                try:
                    # El modelo bert2bert a veces es sensible a la longitud del input
                    # Truncamos si es necesario o manejamos excepciones
                    summary = summarizer(
                        text_input, 
                        max_length=max_len, 
                        min_length=min_len, 
                        do_sample=False
                    )
                    
                    st.markdown("### 📝 Resultado:")
                    st.markdown(f'<div class="result-container">{summary[0]["summary_text"]}</div>', unsafe_allow_html=True)
                    
                    # Opciones extra
                    st.download_button(
                        label="Descargar Resumen",
                        data=summary[0]["summary_text"],
                        file_name="resumen.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")

    st.markdown("---")
    st.caption("Desarrollado con Hugging Face Transformers & Streamlit | Ejecución Local")

if __name__ == "__main__":
    main()
