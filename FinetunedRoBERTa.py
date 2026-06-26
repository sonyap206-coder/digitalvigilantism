"""
Fine-tune cardiffnlp/twitter-roberta-base-sentiment on your annotated comments.

Saves the fine-tuned model locally for use with classify_comments.py.

Install:
    pip install transformers torch pandas scikit-learn datasets

Usage:
    python3 FinetunedRoBERTa.py \
        -i CommentLabels.csv \
        -o finetuned_roberta \
        --label-col claude_label \
        --text-col comment

Notes:
    - Expects labels: Positive, Negative, Neutral
    - Skips any rows labeled "Skip"
    - Splits data 80/20 train/validation automatically
    - With 100-500 comments, training takes ~5-15 min on Mac CPU
"""

import argparse
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import classification_report

BASE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
LABEL2ID   = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL   = {0: "Negative", 1: "Neutral", 2: "Positive"}


# ── Dataset class ─────────────────────────────────────────────────────────────

class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(
        labels, preds,
        target_names=["Negative", "Neutral", "Positive"],
        output_dict=True,
        zero_division=0
    )
    return {
        "accuracy":        report["accuracy"],
        "f1_positive":     report["Positive"]["f1-score"],
        "f1_neutral":      report["Neutral"]["f1-score"],
        "f1_negative":     report["Negative"]["f1-score"],
        "f1_macro":        report["macro avg"]["f1-score"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune roBERTa on annotated vigilantism comments."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Annotated CSV (your labels + Claude-classified, validated)")
    parser.add_argument("-o", "--output", default="finetuned_roberta",
                        help="Folder to save fine-tuned model (default: finetuned_roberta)")
    parser.add_argument("--label-col", default="your_label",
                        help="Column containing sentiment labels (default: your_label)")
    parser.add_argument("--text-col", default="comment",
                        help="Column containing comment text (default: comment)")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of training epochs (default: 4)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size (default: 16)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max token length per comment (default: 128)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    # ── Load & clean ──────────────────────────────────────────
    print(f"Loading annotated data from {args.input}...")
    df = pd.read_csv(args.input, encoding="latin-1")

    # Drop skipped rows and anything not in the three valid labels
    valid_labels = set(LABEL2ID.keys())
    df = df[df[args.label_col].isin(valid_labels)].copy()
    df = df.dropna(subset=[args.text_col, args.label_col])
    df[args.text_col] = df[args.text_col].astype(str).str.strip()
    df = df[df[args.text_col] != ""].reset_index(drop=True)

    print(f"Usable annotated comments: {len(df)}")
    print(f"Label distribution:\n{df[args.label_col].value_counts().to_string()}\n")

    if len(df) < 20:
        raise SystemExit("Too few labeled comments to fine-tune. Need at least 20.")

    texts  = df[args.text_col].tolist()
    labels = [LABEL2ID[l] for l in df[args.label_col].tolist()]

    # ── Train / validation split ──────────────────────────────
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels
    )
    print(f"Train: {len(train_texts)} comments | Validation: {len(val_texts)} comments\n")

    # ── Tokenizer & model ─────────────────────────────────────
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )

    train_dataset = CommentDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset   = CommentDataset(val_texts,   val_labels,   tokenizer, args.max_length)

    # ── Training arguments ────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # ── Train ─────────────────────────────────────────────────
    print("Starting fine-tuning...\n")
    trainer.train()

    # ── Save model & tokenizer ────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nFine-tuned model saved to: {args.output}/")

    # ── Final validation report ───────────────────────────────
    print("\n--- Validation Performance ---")
    preds_output = trainer.predict(val_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    print(classification_report(
        val_labels, preds,
        target_names=["Negative", "Neutral", "Positive"],
        zero_division=0
    ))
    print(f"\nDone. Use classify_comments.py with --model {args.output} to classify new comments.")


if __name__ == "__main__":
    main()
