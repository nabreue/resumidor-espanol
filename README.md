# Resumidor en Español

> Asistente de IA especializado en resumir artículos tecnológicos en español, ejecutándose **completamente en local** gracias a Hugging Face Transformers y fine-tuning propio.

---

## Integrantes

| Nombre |
|--------|
| [Juan] | 
| [Neilin] |

---

## Descripción

**Resumidor en Español** genera resúmenes automáticos y precisos de artículos de noticias tecnológicas en español. A diferencia de un modelo de propósito general, este asistente fue **especializado mediante fine-tuning** en el dominio tecnológico: términos como "computación cuántica", "ciberseguridad" o "modelo de lenguaje" se abordan con el vocabulario y la concisión adecuados.

- Sin APIs externas — todo se procesa en tu máquina
- Interfaz web con **Streamlit** (principal) o **Gradio** (alternativa)
- Compatible con CPU (y acelerado por GPU si está disponible)
- **Límite de entrada**: máximo 1500 caracteres por texto o archivo