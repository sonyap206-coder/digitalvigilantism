"""
Sentiment analysis on YouTube comments using twitter-roBERTa.

Pipeline:
  1) Load comments from CSV
  2) Run each comment through cardiffnlp/twitter-roberta-base-sentiment
  3) Label each comment as Positive, Negative, or Neutral with a confidence score
  4) Save results to CSV
  5) Print a summary breakdown

Usage:
  python3 SchlepSentiments.py -i SchlepComments.csv -o SchlepSentiments
"""

import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
import re

# Labels for cardiffnlp/twitter-roberta-base-sentiment
LABELS = ["Negative", "Neutral", "Positive"]
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

def load_model():
    print("Loading roBERTa sentiment model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

def predict_sentiment(texts, tokenizer, model, batch_size=32):
    """
    Run sentiment prediction in batches.
    Returns list of (label, confidence) tuples.
    """
    results = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring comments"):
        batch = texts[i : i + batch_size]

        # Tokenize — truncate at 512 tokens (roBERTa's max)
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = model(**encoded)

        # Softmax converts raw scores to probabilities
        probs = F.softmax(output.logits, dim=-1)

        for prob_row in probs:
            best_idx = prob_row.argmax().item()
            results.append({
                "sentiment": LABELS[best_idx],
                "confidence": round(prob_row[best_idx].item(), 4),
                "prob_negative": round(prob_row[0].item(), 4),
                "prob_neutral": round(prob_row[1].item(), 4),
                "prob_positive": round(prob_row[2].item(), 4),
            })

    return results

def print_summary(df):
    print("\n--- Sentiment Summary ---")
    counts = df["sentiment"].value_counts()
    total = len(df)
    for label in ["Positive", "Neutral", "Negative"]:
        n = counts.get(label, 0)
        pct = round(100 * n / total, 1)
        print(f"  {label}: {n} comments ({pct}%)")
    print(f"  Total: {total} comments")
    avg_conf = round(df["confidence"].mean(), 4)
    print(f"  Average confidence: {avg_conf}")

def save_pie_chart(df, output_prefix):
    counts = df["sentiment"].value_counts()
    colors = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}
    plt.figure(figsize=(7, 7))
    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=[colors[l] for l in counts.index],
        startangle=140
    )
    plt.title("Overall Sentiment of Comments")
    plt.savefig(f"{output_prefix}_pie.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved pie chart.")

def save_confidence_distribution(df, output_prefix):
    plt.figure(figsize=(9, 5))
    for label, color in [("Positive", "#4CAF50"), ("Neutral", "#FFC107"), ("Negative", "#F44336")]:
        subset = df[df["sentiment"] == label]["confidence"]
        plt.hist(subset, bins=20, alpha=0.6, label=label, color=color)
    plt.xlabel("Confidence Score")
    plt.ylabel("Number of Comments")
    plt.title("Model Confidence by Sentiment Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_confidence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved confidence distribution.")


def save_top_words(df, output_prefix, n=20):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, sentiment, color in zip(
            axes,
            ["Positive", "Negative"],
            ["#4CAF50", "#F44336"]
    ):
        subset = df[df["sentiment"] == sentiment]["text"]
        words = []
        for text in subset:
            words += re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

        # Remove common stopwords
        stopwords = {"the", "and", "for", "this", "that", "you", "are", "was",
                     "with", "have", "not", "but", "they", "his", "her", "just", "like"}
        words = [w for w in words if w not in stopwords]

        common = Counter(words).most_common(n)
        words_list, counts = zip(*common)

        ax.barh(words_list[::-1], counts[::-1], color=color)
        ax.set_title(f"Top Words in {sentiment} Comments")
        ax.set_xlabel("Count")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_top_words.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved top words chart.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    parser.add_argument("-o", "--output", default="sentiment_results", help="Output file prefix")
    parser.add_argument("--text-col", default="text", help="Column name for comments (default: text)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    args = parser.parse_args()

    # Load comments
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    df = df.dropna(subset=[args.text_col])
    df[args.text_col] = df[args.text_col].astype(str).str.strip()
    df = df[df[args.text_col] != ""]
    print(f"Loaded {len(df)} comments.")

    # Load model
    tokenizer, model = load_model()

    # Run sentiment
    texts = df[args.text_col].tolist()
    sentiment_results = predict_sentiment(texts, tokenizer, model, batch_size=args.batch_size)

    # Merge results back into dataframe
    results_df = pd.DataFrame(sentiment_results)
    df = df.reset_index(drop=True)
    df = pd.concat([df, results_df], axis=1)

    # Save
    out_path = f"{args.output}_comments.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved results to {out_path}")

    # Print summary
    print_summary(df)

    # Visualizations
    save_pie_chart(df, args.output)
    save_confidence_distribution(df, args.output)
    save_top_words(df, args.output)

if __name__ == "__main__":
    main()