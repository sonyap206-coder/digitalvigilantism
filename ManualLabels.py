"""
Manual sentiment labeling tool for fine-tuning RoBERTa.

Samples comments from one or more CSVs and prompts you to label each one.
Saves your labels to a CSV ready to hand off to Claude for auto-classification.

Usage:
  # Single file, default 30 comments
  python3 label_comments.py -i schlep_sentiment_comments.csv -o my_labels.csv

  # Multiple files, default 30 comments each
  python3 label_comments.py -i schlep_sentiment_comments.csv jidion_sentiment_comments.csv -o my_labels.csv

  # Multiple files, custom sample size
  python3 label_comments.py -i schlep_sentiment_comments.csv jidion_sentiment_comments.csv --sample-size 20 -o my_labels.csv

  # Different sample size per file (must match number of input files)
  python3 label_comments.py -i schlep_sentiment_comments.csv jidion_sentiment_comments.csv --sample-size 20 10 -o my_labels.csv

Controls:
  P = Positive
  N = Negative
  U = Neutral
  S = Skip (if the comment is too ambiguous or not in English)
  Q = Quit and save progress so far
"""

import argparse
import pandas as pd
import os

LABEL_MAP = {
    "p": "Positive",
    "n": "Negative",
    "u": "Neutral",
    "s": "Skip",
    "q": "QUIT"
}


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(current, total):
    print("=" * 70)
    print(f"  SENTIMENT LABELER  |  Comment {current} of {total}")
    print("=" * 70)
    print("  [P] Positive   [N] Negative   [U] Neutral   [S] Skip   [Q] Quit & save")
    print("-" * 70)


def print_comment(row):
    """Print the comment and any useful context columns if present."""
    print(f"\nCOMMENT:\n")
    print(f"  {row['comment']}\n")

    # Show topic label if available
    if "topic_label" in row and pd.notna(row.get("topic_label")):
        print(f"  Topic cluster : {row['topic_label']}")

    # Show model's original sentiment if available (useful to compare)
    if "sentiment" in row and pd.notna(row.get("sentiment")):
        conf = f"  (model confidence: {row['confidence']})" if "confidence" in row else ""
        print(f"  Model said    : {row['sentiment']}{conf}")

    print()


def get_label():
    while True:
        raw = input("Your label: ").strip().lower()
        if raw in LABEL_MAP:
            return LABEL_MAP[raw]
        print("  Invalid input. Enter P, N, U, S, or Q.")





def main():
    parser = argparse.ArgumentParser(
        description="Manually label a random sample of comments for sentiment."
    )
    parser.add_argument("-i", "--input", required=True, nargs="+",
                        help="One or more input comment CSVs")
    parser.add_argument("-o", "--output", default="my_labels.csv",
                        help="Output CSV for your labels (default: my_labels.csv)")
    parser.add_argument("--sample-size", type=int, nargs="+", default=[30],
                        help=(
                            "Number of comments to sample. "
                            "Pass one number to apply to all files, "
                            "or one number per file (e.g. --sample-size 20 10). "
                            "Default: 30 per file."
                        ))
    parser.add_argument("--text-col", default="comment",
                        help="Column name for comment text (default: comment)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    # ── Resolve sample sizes per file ─────────────────────────
    n_files = len(args.input)
    if len(args.sample_size) == 1:
        # Same size for all files
        sample_sizes = args.sample_size * n_files
    elif len(args.sample_size) == n_files:
        sample_sizes = args.sample_size
    else:
        raise SystemExit(
            f"--sample-size must be either one number (applied to all files) "
            f"or exactly {n_files} numbers (one per file). "
            f"Got {len(args.sample_size)}."
        )

    # ── Load and sample from each file ────────────────────────
    all_samples = []
    for filepath, n_sample in zip(args.input, sample_sizes):
        print(f"\nLoading {filepath}...")
        df = pd.read_csv(filepath, encoding="utf-8-sig")

        if args.text_col not in df.columns:
            raise SystemExit(
                f"Column '{args.text_col}' not found in {filepath}. "
                f"Available: {list(df.columns)}"
            )

        df = df.dropna(subset=[args.text_col])
        df[args.text_col] = df[args.text_col].astype(str).str.strip()
        df = df[df[args.text_col] != ""].reset_index(drop=True)

        n = min(n_sample, len(df))
        sample = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
        sample = sample.rename(columns={args.text_col: "comment"})

        # Tag which file each comment came from
        sample["source_file"] = os.path.basename(filepath)
        all_samples.append(sample)
        print(f"  Sampled {n} of {len(df)} comments from {os.path.basename(filepath)}")

    # Combine and shuffle so comments from different files are interleaved
    combined = pd.concat(all_samples, ignore_index=True)
    combined = combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    total = len(combined)

    print(f"\nTotal comments to label: {total}")
    print("Press Enter to begin...\n")
    input()

    # ── Labeling loop ─────────────────────────────────────────
    labels = []
    for i, row in combined.iterrows():
        clear()
        print_header(i + 1, total)

        # Show source file when using multiple inputs
        if n_files > 1:
            print(f"  Source: {row.get('source_file', 'unknown')}")

        print_comment(row)

        label = get_label()

        if label == "QUIT":
            print(f"\nQuitting early. Saving {len(labels)} labels so far.")
            break

        labels.append(label)
        print(f"  Saved: {label}\n")

    # ── Save output ───────────────────────────────────────────
    labeled = combined.iloc[:len(labels)].copy()
    labeled["your_label"] = labels

    output_cols = ["comment", "source_file"]
    for optional in ["topic_label", "topic", "sentiment", "confidence"]:
        if optional in labeled.columns:
            output_cols.append(optional)
    output_cols.append("your_label")

    labeled[output_cols].to_csv(args.output, index=False, encoding="utf-8-sig")

    # ── Summary ───────────────────────────────────────────────
    print("\n--- Labeling Summary ---")
    label_counts = pd.Series(labels).value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    if n_files > 1:
        print("\n  By source file:")
        for src in labeled["source_file"].unique():
            src_labels = labeled[labeled["source_file"] == src]["your_label"]
            print(f"    {src}: {dict(src_labels.value_counts())}")
    print(f"\nSaved {len(labels)} labeled comments to: {args.output}")
    print("\nNext step: upload your output CSV to Claude for auto-classification.")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()