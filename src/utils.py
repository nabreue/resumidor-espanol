"""
utils.py — Utilidades compartidas del proyecto ResumidorTech Pro.
"""
import json
from pathlib import Path

# ── Rutas del proyecto ────────────────────────────────────────────────────────
ROOT_DIR           = Path(__file__).parent.parent
DATA_DIR           = ROOT_DIR / "data"
PROCESSED_DIR      = DATA_DIR / "processed"
RAW_DIR            = DATA_DIR / "raw"
FINE_TUNED_MODEL_DIR = ROOT_DIR / "model_finetuned"

# ── Modelo base en Hugging Face ───────────────────────────────────────────────
MODEL_NAME = (
    "Narrativa/bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization"
)


def load_dataset(path: Path | None = None) -> dict:
    """Carga el dataset JSON con las claves 'train' y 'validation'."""
    if path is None:
        path = PROCESSED_DIR / "dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs() -> None:
    """Crea los directorios de datos si no existen."""
    for directory in [DATA_DIR, PROCESSED_DIR, RAW_DIR, FINE_TUNED_MODEL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    print("✅ Directorios de datos verificados correctamente.")


def get_model_path() -> str:
    """
    Devuelve la ruta al modelo fine-tuned si existe,
    o el nombre del modelo base de Hugging Face en caso contrario.
    """
    if FINE_TUNED_MODEL_DIR.exists() and any(FINE_TUNED_MODEL_DIR.iterdir()):
        print(f"📂 Usando modelo fine-tuned: {FINE_TUNED_MODEL_DIR}")
        return str(FINE_TUNED_MODEL_DIR)
    print(f"🌐 Usando modelo base de Hugging Face: {MODEL_NAME}")
    return MODEL_NAME
