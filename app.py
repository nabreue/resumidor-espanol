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

if __name__ == "__main__":
    src_app = Path(__file__).parent / "src" / "app.py"
    subprocess.run([sys.executable, str(src_app)], check=True)
