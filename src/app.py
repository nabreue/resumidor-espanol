"""
app.py — Interfaz web con Gradio para el ResumidorTech Pro.

Ejecutar:
    python src/app.py

La interfaz se abrirá automáticamente en http://localhost:7860
"""

import sys
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# ── Añadir src/ al path para importar utils ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils import get_model_path  # noqa: E402

# ── Carga del modelo (global, se hace una sola vez) ───────────────────────────
_summarizer = None


def _load_model():
    """Carga el pipeline de resumen (modelo fine-tuned si existe, base si no)."""
    global _summarizer
    if _summarizer is not None:
        return _summarizer

    model_path = get_model_path()
    tokenizer  = AutoTokenizer.from_pretrained(model_path)
    device     = 0 if torch.cuda.is_available() else -1
    model      = AutoModelForSeq2SeqLM.from_pretrained(model_path, low_cpu_mem_usage=False)
    _summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    return _summarizer


# ── Función principal de resumen ──────────────────────────────────────────────
def summarize(text: str, max_length: int, min_length: int) -> str:
    """Genera un resumen del texto de entrada usando el modelo cargado."""
    if not text.strip():
        return "⚠️ Por favor, introduce algún texto para resumir."

    summarizer = _load_model()
    try:
        result = summarizer(
            text,
            max_length=int(max_length),
            min_length=int(min_length),
            do_sample=False,
        )
        return result[0]["summary_text"]
    except Exception as exc:
        return f"❌ Error al generar el resumen: {exc}"


# ── Ejemplos de demostración ──────────────────────────────────────────────────
EXAMPLES = [
    [
        (
            "OpenAI ha publicado GPT-4o, una nueva versión de su modelo que procesa texto, "
            "imágenes y audio de forma nativa en un único modelo unificado. La compañía asegura "
            "que el tiempo de respuesta se ha reducido un 50% respecto a GPT-4 Turbo y que el "
            "modelo es especialmente eficiente en idiomas distintos al inglés, incluyendo el español."
        ),
        120,
        40,
    ],
    [
        (
            "Un grupo de hackers ha comprometido los sistemas de una importante cadena de "
            "supermercados europea, logrando acceder a los datos personales de más de 8 millones "
            "de clientes. La brecha de seguridad incluye nombres, direcciones de correo electrónico "
            "y contraseñas cifradas. La empresa ha notificado a la Agencia Española de Protección "
            "de Datos y ha instado a todos sus usuarios a cambiar sus contraseñas inmediatamente."
        ),
        100,
        30,
    ],
    [
        (
            "El Parlamento Europeo ha aprobado definitivamente el Reglamento de Inteligencia "
            "Artificial (AI Act), convirtiéndose en la primera legislación integral sobre IA del "
            "mundo. La norma clasifica los sistemas de IA según su nivel de riesgo y prohíbe "
            "aplicaciones consideradas inaceptables, como el reconocimiento facial masivo en "
            "espacios públicos. Las empresas tendrán hasta 2026 para adaptarse a los requisitos "
            "más exigentes."
        ),
        110,
        35,
    ],
]

# ── Construcción de la interfaz Gradio ────────────────────────────────────────
CSS = """
#titulo { text-align: center; margin-bottom: 10px; }
#subtitulo { text-align: center; color: #6b7280; margin-bottom: 20px; }
#boton { background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); }
footer { display: none !important; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue"),
    title="ResumidorTech Pro",
    css=CSS,
) as demo:

    gr.Markdown("# 📰 ResumidorTech Pro", elem_id="titulo")
    gr.Markdown(
        "Resumidor especializado en **noticias tecnológicas en español** — "
        "ejecución 100% local con Hugging Face Transformers.",
        elem_id="subtitulo",
    )

    with gr.Row():
        with gr.Column(scale=3):
            texto_entrada = gr.Textbox(
                label="📄 Texto de la noticia",
                placeholder="Pega aquí el artículo de tecnología que quieres resumir...",
                lines=12,
                max_lines=30,
            )
            with gr.Row():
                max_len = gr.Slider(
                    minimum=30, maximum=300, value=120, step=10,
                    label="Longitud máxima del resumen",
                )
                min_len = gr.Slider(
                    minimum=10, maximum=100, value=40, step=5,
                    label="Longitud mínima del resumen",
                )
            btn = gr.Button("🔍 Generar Resumen", variant="primary", elem_id="boton")

        with gr.Column(scale=2):
            salida = gr.Textbox(
                label="📝 Resumen generado",
                lines=10,
                interactive=False,
                show_copy_button=True,
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[texto_entrada, max_len, min_len],
        label="📌 Ejemplos de prueba",
        examples_per_page=3,
    )

    gr.Markdown(
        "---\n"
        "*Desarrollado con [Hugging Face Transformers](https://huggingface.co/docs/transformers) "
        "& [Gradio](https://gradio.app) · Ejecución local · Fine-tuning Opción A*"
    )

    btn.click(
        fn=summarize,
        inputs=[texto_entrada, max_len, min_len],
        outputs=salida,
    )


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔄 Cargando modelo (puede tardar la primera vez)...")
    _load_model()
    print("✅ Modelo cargado. Abriendo interfaz en http://localhost:7860\n")
    demo.launch(inbrowser=True)
