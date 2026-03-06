"""
train.py — Fine-tuning del modelo resumidor sobre noticias tecnológicas en español.

Técnica: Opción A (Fine-Tuning Tradicional) usando Seq2SeqTrainer de Hugging Face.
Modelo base: Narrativa/bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization

Uso:
    # Entrenamiento rápido (CPU, ~20-30 min)
    python src/train.py

    # Con más épocas y batch personalizado
    python src/train.py --epochs 5 --batch_size 2
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ── Añadir src/ al path para importar utils ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils import FINE_TUNED_MODEL_DIR, MODEL_NAME, load_dataset  # noqa: E402

# ── Hiperparámetros de tokenización ──────────────────────────────────────────
MAX_INPUT_LENGTH  = 512
MAX_TARGET_LENGTH = 128


def preprocess(examples: dict, tokenizer: AutoTokenizer) -> dict:
    """Tokeniza los textos de entrada y los resúmenes objetivo."""
    model_inputs = tokenizer(
        examples["text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        examples["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )
    # Transformar IDs de padding a -100 para que el Trainer los ignore en la pérdida
    label_ids = labels["input_ids"]
    label_ids = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
        for seq in label_ids
    ]
    model_inputs["labels"] = label_ids
    return model_inputs


def main(args: argparse.Namespace) -> None:
    # ── 1. Cargar tokenizador y modelo base ───────────────────────────────────
    print(f"\n📥 Cargando tokenizador y modelo base: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=False)

    # ── 2. Cargar y preparar el dataset ───────────────────────────────────────
    print("\n📂 Cargando dataset...")
    data = load_dataset()
    train_ds = Dataset.from_list(data["train"])
    val_ds   = Dataset.from_list(data["validation"])
    print(f"   → Muestras de entrenamiento : {len(train_ds)}")
    print(f"   → Muestras de validación    : {len(val_ds)}")

    print("\n🔤 Tokenizando datos...")
    tokenized_train = train_ds.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    tokenized_val = val_ds.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # ── 3. Configurar el entrenamiento ────────────────────────────────────────
    use_fp16 = torch.cuda.is_available()
    print(f"\n⚙️  Dispositivo: {'GPU (CUDA)' if use_fp16 else 'CPU'}")
    print(f"   Épocas       : {args.epochs}")
    print(f"   Batch size   : {args.batch_size}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(FINE_TUNED_MODEL_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir=str(FINE_TUNED_MODEL_DIR / "logs"),
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=use_fp16,
        report_to="none",          # Deshabilita Weights & Biases y similares
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ── 4. Entrenar ────────────────────────────────────────────────────────────
    print("\n🚀 Iniciando fine-tuning...")
    trainer.train()

    # ── 5. Guardar modelo y tokenizador ───────────────────────────────────────
    trainer.save_model(str(FINE_TUNED_MODEL_DIR))
    tokenizer.save_pretrained(str(FINE_TUNED_MODEL_DIR))
    print(f"\n✅ Fine-tuning completado. Modelo guardado en: {FINE_TUNED_MODEL_DIR}")
    print("   Para lanzar la app con el modelo fine-tuned: python src/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tuning del ResumidorTech Pro en noticias tecnológicas en español"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Número de épocas de entrenamiento (por defecto: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Tamaño del batch por dispositivo (por defecto: 2)",
    )
    main(parser.parse_args())
