"""
Sentiment analysis per BERTopic cluster using twitter-roBERTa.

Pipeline:
  1) Load comments CSV (with topic column) and topic_info CSV
  2) Label topic -1 comments as "Outliers"
  3) Run cardiffnlp/twitter-roberta-base-sentiment on every comment
  4) Save comment-level CSV with sentiment columns added
  5) Aggregate sentiment per cluster and save topic-level summary CSV
  6) Print a summary to terminal

Install:
  pip install transformers torch pandas tqdm

Usage:
  python3 sentiment_by_cluster.py \
    -c SchlepTopicsI2_comments.csv \
    -t SchlepTopicsI2_topic_info.csv \
    -o schlep_sentiment
"""

import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABELS = ["Negative", "Neutral", "Positive"]


def load_model():
    print("Loading roBERTa model (downloads ~500MB on first run)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def predict_sentiment(texts, tokenizer, model, batch_size=32):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring comments"):
        batch = texts[i: i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        with torch.no_grad():
            output = model(**encoded)
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


def build_topic_summary(comments_df, topic_info_df):
    """
    Aggregate sentiment counts and percentages per topic cluster.
    Merges in topic name and keywords from topic_info.
    """
    # Sentiment counts per topic
    grouped = (
        comments_df
        .groupby(["topic", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    )
    grouped["total_comments"] = grouped.sum(axis=1)
    grouped["pct_positive"] = (grouped["Positive"] / grouped["total_comments"] * 100).round(1)
    grouped["pct_neutral"]  = (grouped["Neutral"]  / grouped["total_comments"] * 100).round(1)
    grouped["pct_negative"] = (grouped["Negative"] / grouped["total_comments"] * 100).round(1)
    grouped["dominant_sentiment"] = grouped[["Positive", "Neutral", "Negative"]].idxmax(axis=1)
    grouped["avg_confidence"] = (
        comments_df.groupby("topic")["confidence"].mean().round(4)
    )
    grouped = grouped.reset_index()

    # Merge topic name and keywords from topic_info
    topic_info_trimmed = topic_info_df[["Topic", "Name", "Representation"]].rename(
        columns={"Topic": "topic", "Name": "topic_name", "Representation": "keywords"}
    )
    summary = grouped.merge(topic_info_trimmed, on="topic", how="left")

    # Put descriptive columns first
    col_order = [
        "topic", "topic_name", "keywords",
        "total_comments", "dominant_sentiment",
        "Positive", "Neutral", "Negative",
        "pct_positive", "pct_neutral", "pct_negative",
        "avg_confidence"
    ]
    return summary[col_order]


def print_terminal_summary(summary_df):
    print("\n--- Sentiment by Cluster ---")
    print(f"{'Topic':<8} {'Dominant':<12} {'Pos%':>6} {'Neu%':>6} {'Neg%':>6}  {'Name'}")
    print("-" * 70)
    for _, row in summary_df.iterrows():
        label = "Outliers" if row["topic"] == -1 else str(int(row["topic"]))
        print(
            f"{label:<8} {row['dominant_sentiment']:<12} "
            f"{row['pct_positive']:>6} {row['pct_neutral']:>6} {row['pct_negative']:>6}  "
            f"{row['topic_name']}"
        )
    print(f"\nTotal topics (including outliers): {len(summary_df)}")

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

def save_topic_piecharts_pdf(summary_df, output_prefix):

    pdf_file = f"{output_prefix}_topic_piecharts.pdf"

    colors = {
        "Positive": "#4CAF50",
        "Neutral": "#FFC107",
        "Negative": "#F44336"
    }

    charts_per_page = 4

    with PdfPages(pdf_file) as pdf:

        for start in range(0, len(summary_df), charts_per_page):

            subset = summary_df.iloc[start:start + charts_per_page]

            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            axes = axes.flatten()

            for ax in axes:
                ax.axis("off")

            for ax, (_, row) in zip(axes, subset.iterrows()):

                values = [
                    row["Positive"],
                    row["Neutral"],
                    row["Negative"]
                ]

                ax.axis("on")

                ax.pie(
                    values,
                    labels=["Pos", "Neu", "Neg"],
                    autopct="%1.0f%%",
                    startangle=140,
                    colors=[
                        colors["Positive"],
                        colors["Neutral"],
                        colors["Negative"]
                    ]
                )

                topic_label = (
                    "Outliers"
                    if row["topic"] == -1
                    else f"Topic {int(row['topic'])}"
                )

                ax.set_title(
                    f"{topic_label}\nN={row['total_comments']}",
                    fontsize=10
                )

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved topic pie chart PDF: {pdf_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Run sentiment analysis per BERTopic cluster."
    )
    parser.add_argument("-c", "--comments", required=True,
                        help="Comments CSV (output from BERTopic, must have 'text' and 'topic' columns)")
    parser.add_argument("-t", "--topic-info", required=True,
                        help="Topic info CSV (output from BERTopic)")
    parser.add_argument("-o", "--output", default="sentiment_by_cluster",
                        help="Output file prefix (default: sentiment_by_cluster)")
    parser.add_argument("--text-col", default="text",
                        help="Column name for comment text (default: text)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for model inference (default: 32)")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────
    print(f"Loading comments from {args.comments}...")
    comments_df = pd.read_csv(args.comments, encoding="utf-8-sig")
    topic_info_df = pd.read_csv(args.topic_info, encoding="utf-8-sig")

    # Validate required columns
    for col in [args.text_col, "topic"]:
        if col not in comments_df.columns:
            raise SystemExit(f"Column '{col}' not found in comments CSV. "
                             f"Available: {list(comments_df.columns)}")
    if "Topic" not in topic_info_df.columns:
        raise SystemExit(f"Column 'Topic' not found in topic_info CSV.")

    # Clean text
    comments_df = comments_df.dropna(subset=[args.text_col])
    comments_df[args.text_col] = comments_df[args.text_col].astype(str).str.strip()
    comments_df = comments_df[comments_df[args.text_col] != ""].reset_index(drop=True)

    # Label outlier cluster
    comments_df["topic_label"] = comments_df["topic"].apply(
        lambda x: "Outliers" if x == -1 else str(int(x))
    )
    print(f"Loaded {len(comments_df)} comments across "
          f"{comments_df['topic'].nunique()} clusters.")

    # ── Run sentiment ─────────────────────────────────────────
    tokenizer, model = load_model()
    texts = comments_df[args.text_col].tolist()
    sentiment_results = predict_sentiment(texts, tokenizer, model, args.batch_size)

    # Add sentiment columns to comments
    results_df = pd.DataFrame(sentiment_results)
    comments_df = pd.concat([comments_df, results_df], axis=1)

    # ── Output 1: comment-level CSV ───────────────────────────
    comments_out = f"{args.output}_comments.csv"
    comments_df.to_csv(comments_out, index=False, encoding="utf-8-sig")
    print(f"\nSaved comment-level results to: {comments_out}")

    # ── Output 2: topic-level summary CSV ─────────────────────
    summary_df = build_topic_summary(comments_df, topic_info_df)
    summary_out = f"{args.output}_topic_summary.csv"
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")
    print(f"Saved topic summary to:         {summary_out}")

    # ── Print terminal summary ────────────────────────────────
    print_terminal_summary(summary_df)

    save_topic_piecharts_pdf(summary_df, args.output)
    summary_df = build_topic_summary(comments_df, topic_info_df)


if __name__ == "__main__":
    main()