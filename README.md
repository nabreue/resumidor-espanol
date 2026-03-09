# Resumidor en Español

> Asistente de IA especializado en resumir artículos tecnológicos en español, ejecutándose **completamente en local** gracias a Hugging Face Transformers y fine-tuning propio.

---

## Integrantes

| Nombre | Rol |
|--------|-----|
| [Nombre Integrante 1] | Desarrollo del modelo y fine-tuning |
| [Nombre Integrante 2] | Interfaz, dataset y documentación |

---

## Descripción

**Resumidor en Español** genera resúmenes automáticos y precisos de artículos de noticias tecnológicas en español. A diferencia de un modelo de propósito general, este asistente fue **especializado mediante fine-tuning** en el dominio tecnológico: términos como "computación cuántica", "ciberseguridad" o "modelo de lenguaje" se abordan con el vocabulario y la concisión adecuados.

- Sin APIs externas — todo se procesa en tu máquina
- Interfaz web con Gradio, accesible desde el navegador
- Compatible con CPU (y acelerado por GPU si está disponible)

---

## Modelo Base

**[`Narrativa/bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization`](https://huggingface.co/Narrativa/bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization)**

Desarrollado por el Barcelona Supercomputing Center y Narrativa AI. Es un modelo encoder-decoder (RoBERTa → RoBERTa) pre-entrenado en **MLSUM**, el dataset de noticias de referencia en español.

**¿Por qué este modelo?**
- Nativo en español (no traducción del inglés)
- Arquitectura seq2seq ideal para generación de resúmenes
- Tamaño manejable (~500 MB) para ejecutarse en CPU

---

## Técnica de Adaptación: Fine-Tuning (Opción A)

Elegimos **fine-tuning tradicional** sobre RAG porque nuestro objetivo es *generar* resúmenes en un estilo concreto, no recuperar fragmentos de una base de datos. El fine-tuning es la técnica correcta cuando queremos mejorar al modelo en un **dominio y tarea específicos**.

### ¿Cómo funciona?

```
[Artículo tech]  →  [Tokenizer]  →  [RoBERTa Encoder]  →  [RoBERTa Decoder]  →  [Resumen]
                                            ↑
                              pesos ajustados con nuestro dataset
```

1. **Dataset**: 60 pares (artículo → resumen) de noticias tech curados a mano
2. **Tokenización**: los textos se convierten en tokens numéricos con truncado a 512 tokens
3. **Entrenamiento**: el `Seq2SeqTrainer` de Hugging Face minimiza la pérdida cruzada entre los resúmenes predichos y los reales del dataset
4. **Evaluación**: se mide el rendimiento en las 10 muestras de validación tras cada época

---

## Dataset

| Aspecto | Detalle |
|---------|---------|
| **Fuente** | Pares curados manualmente por el equipo |
| **Dominio** | Noticias tecnológicas en español (IA, ciberseguridad, chips, legislación tech…) |
| **Tamaño** | 60 pares · 50 entrenamiento + 10 validación |
| **Formato** | JSON (`data/processed/dataset.json`) |
| **Procesamiento** | Textos normalizados: sin saltos dobles, unicidad del par verificada manualmente |

---

## Requisitos del Sistema

| Componente | Mínimo | Recomendado |
|-----------|--------|-------------|
| RAM | 6 GB | 8 GB |
| CPU | Cualquier x86-64 | Multi-core moderno |
| GPU | No requerida | CUDA compatible (acelera el fine-tuning) |
| Disco libre | 2 GB | 4 GB |
| Python | 3.9+ | 3.10 / 3.11 |

> Los pesos del modelo (~500 MB) se descargan automáticamente desde Hugging Face la primera vez. **No están incluidos en el repositorio** (ver `.gitignore`).

---

## Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone [url-de-vuestro-repositorio]
cd resumidor-espanol
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. (Opcional) Ejecutar el fine-tuning

> Solo necesario si quieres re-entrenar el modelo con nuestro dataset. La app funciona sin este paso usando el modelo base directamente.

```bash
# Entrenamiento con valores por defecto (3 épocas, batch 2)
python src/train.py

# Personalizado
python src/train.py --epochs 5 --batch_size 4
```

El modelo fine-tuned se guardará automáticamente en `model_finetuned/`.

### 5. Lanzar la aplicación

```bash
python src/app.py
```

La interfaz se abrirá automáticamente en **http://localhost:7860**

---

## Ejemplos de Uso

### Ejemplo 1 — Noticia sobre IA

**Pregunta / Input:**
> OpenAI ha publicado GPT-4o, una nueva versión de su modelo que procesa texto, imágenes y audio de forma nativa en un único modelo unificado. La compañía asegura que el tiempo de respuesta se ha reducido un 50% respecto a GPT-4 Turbo y que el modelo es especialmente eficiente en idiomas distintos al inglés, incluyendo el español.

**Respuesta / Output:**
> OpenAI presenta GPT-4o, modelo multimodal unificado con respuestas un 50% más rápidas y mejor soporte en español.

---

### Ejemplo 2 — Noticia de ciberseguridad

**Input:**
> Un grupo de hackers ha comprometido los sistemas de una importante cadena de supermercados europea, logrando acceder a los datos personales de más de 8 millones de clientes. La brecha incluye nombres, correos y contraseñas cifradas. La empresa ha notificado a la AEPD e instado a sus usuarios a cambiar contraseñas.

**Output:**
> Un ciberataque expone datos de 8 millones de clientes de una cadena europea, incluyendo correos y contraseñas cifradas.

---

### Ejemplo 3 — Noticia de legislación tech

**Input:**
> El Parlamento Europeo ha aprobado definitivamente el Reglamento de Inteligencia Artificial (AI Act), convirtiéndose en la primera legislación integral sobre IA del mundo. La norma clasifica los sistemas según su nivel de riesgo y prohíbe el reconocimiento facial masivo en espacios públicos. Las empresas tendrán hasta 2026 para adaptarse.

**Output:**
> La UE aprueba el AI Act, primera ley mundial de IA, que clasifica sistemas por riesgo y prohíbe el reconocimiento facial masivo.

---

## Referencias

- [Modelo base en Hugging Face](https://huggingface.co/Narrativa/bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization)
- [MLSUM Dataset](https://huggingface.co/datasets/mlsum)
- [Hugging Face Seq2SeqTrainer](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer)
- [Gradio Documentation](https://www.gradio.app/docs)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)

---

## Autoevaluación

### ¿Qué fue lo más difícil?
La preparación del dataset fue la parte más laboriosa: garantizar que los resúmenes fueran realmente **abstractivos** (no meras frases extraídas del texto) y cubrieran el contenido esencial de cada noticia. También fue un reto gestionar la memoria durante el fine-tuning en CPU, donde fue necesario usar batch size 2 y truncar los textos a 512 tokens.

### ¿Qué resultados obtuvisteis?
El modelo fine-tuned muestra mayor fluidez con terminología tecnológica específica (p.ej. "ransomware Ryuk", "ventana de contexto de un millón de tokens") en comparación con el modelo base, que tendía a parafrasear de forma más genérica.

### ¿Qué mejoraríais con más tiempo?
- Aumentar el dataset a 500+ pares con artículos reales de fuentes como Xataka o El País Tecnología
- Implementar métricas **ROUGE** automáticas durante el entrenamiento para medir la calidad de los resúmenes
- Explorar **LoRA / QLoRA** para un fine-tuning más eficiente en GPU con menos memoria
- Añadir soporte para entrada de **URLs** y **archivos PDF**
