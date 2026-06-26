"""
Sample random comments from one or more CSVs for Claude to classify.

Output is a clean CSV with just the comments (and source/topic context),
ready to upload to Claude for auto-classification.

Usage:
    # Single file, default 200 comments
    python3 sample_for_claude.py -i Schlep/I/SchlepTopicsI_comments.csv -o claude_sample.csv

    # Multiple files, same sample size each
    python3 sample_for_claude.py \
        -i Schlep/I/SchlepTopicsI_comments.csv \
           JiDion/H/JiDionTopicsH_comments.csv \
        --sample-size 100 \
        -o claude_sample.csv

    # Multiple files, different sample size per file
    python3 sample_for_claude.py \
        -i Schlep/I/SchlepTopicsI_comments.csv \
           JiDion/H/JiDionTopicsH_comments.csv \
        --sample-size 150 100 \
        -o claude_sample.csv
"""

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Sample random comments from one or more CSVs for Claude classification."
    )
    parser.add_argument("-i", "--input", required=True, nargs="+",
                        help="One or more input comment CSVs")
    parser.add_argument("-o", "--output", default="claude_sample.csv",
                        help="Output CSV filename (default: claude_sample.csv)")
    parser.add_argument("--sample-size", type=int, nargs="+", default=[200],
                        help=(
                            "Number of comments to sample. One number applies to all files. "
                            "Or pass one number per file (e.g. --sample-size 150 100). "
                            "Default: 200 per file."
                        ))
    parser.add_argument("--text-col", default="text",
                        help="Column name for comment text (default: text)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    # ── Resolve sample sizes per file ─────────────────────────
    n_files = len(args.input)
    if len(args.sample_size) == 1:
        sample_sizes = args.sample_size * n_files
    elif len(args.sample_size) == n_files:
        sample_sizes = args.sample_size
    else:
        raise SystemExit(
            f"--sample-size must be one number (applied to all files) or "
            f"exactly {n_files} numbers (one per file). Got {len(args.sample_size)}."
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

        # Clean
        df = df.dropna(subset=[args.text_col])
        df[args.text_col] = df[args.text_col].astype(str).str.strip()
        df = df[df[args.text_col] != ""].reset_index(drop=True)

        n = min(n_sample, len(df))
        sample = df.sample(n=n, random_state=args.seed).reset_index(drop=True)

        # Tag source file
        sample["source_file"] = os.path.basename(filepath)

        all_samples.append(sample)
        print(f"  Sampled {n} of {len(df)} comments from {os.path.basename(filepath)}")

    # ── Combine and shuffle ───────────────────────────────────
    combined = pd.concat(all_samples, ignore_index=True)
    combined = combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # ── Build clean output ────────────────────────────────────
    # Always include: comment text and source file
    # Include topic columns if present (useful context for Claude)
    output_cols = [args.text_col, "source_file"]
    for optional in ["topic", "topic_label"]:
        if optional in combined.columns:
            output_cols.append(optional)

    # Add blank column for Claude to fill in
    combined["claude_label"] = ""

    output_cols.append("claude_label")
    output_df = combined[output_cols].rename(columns={args.text_col: "comment"})

    # ── Save ──────────────────────────────────────────────────
    output_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n--- Sample Summary ---")
    print(f"  Total comments sampled : {len(output_df)}")
    if n_files > 1:
        for src, count in output_df["source_file"].value_counts().items():
            print(f"  {src}: {count}")
    print(f"\nSaved to: {args.output}")
    print("Next step: upload this CSV to Claude for sentiment classification.")


if __name__ == "__main__":
    main()