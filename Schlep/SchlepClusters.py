
from __future__ import annotations

import os

# Must run before importing tokenizers/sklearn/numpy to avoid segfaults on macOS when
# the process is forked after parallelism has been used (KMeans etc.).
# - TOKENIZERS_PARALLELISM=false avoids HuggingFace tokenizer fork warnings/deadlocks.
# - Limiting OpenMP/BLAS threads avoids Intel vs LLVM OpenMP conflicts and crashes.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def clean_comment(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = URL_RE.sub(" ", text)
    text = text.replace("\u200b", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text


def load_comments_from_csv(path: Path, text_col: str, encoding: str) -> List[str]:
    df = pd.read_csv(path, encoding=encoding)
    if text_col not in df.columns:
        raise SystemExit(
            f"Column '{text_col}' not found in CSV. Available columns: {list(df.columns)}"
        )

    series = df[text_col].astype("string").fillna("")
    comments = [c for c in (clean_comment(x) for x in series.tolist()) if c]
    return comments


def batched(iterable: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def compute_embeddings(
    comments: List[str],
    model_name: str,
    batch_size: int,
    device: str | None = None,
) -> np.ndarray:
    """
    Encode comments into embeddings using SentenceTransformer.
    """
    model = SentenceTransformer(model_name, device=device)
    # Let sentence-transformers handle batching & progress bar
    embeddings = model.encode(
        comments,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32", copy=False)


def suggest_elbow_k(ks: List[int], inertias: List[float]) -> int:
    """
    Suggest a k by maximizing curvature on log(inertia).
    Simple, dependency-free elbow heuristic.
    """
    if len(ks) < 3:
        return int(ks[-1])

    y = [math.log(max(float(v), 1e-12)) for v in inertias]
    second_diff = [y[i + 1] - 2 * y[i] + y[i - 1] for i in range(1, len(y) - 1)]
    idx = int(max(range(len(second_diff)), key=lambda i: second_diff[i])) + 1
    return int(ks[idx])


def run_kmeans_range(
    X: np.ndarray,
    min_k: int,
    max_k: int,
    random_state: int,
) -> Tuple[List[int], List[float]]:
    ks = list(range(min_k, max_k + 1))
    inertias: List[float] = []

    for k in ks:
        km = KMeans(
            n_clusters=int(k),
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=random_state,
        )
        km.fit(X)
        inertias.append(float(km.inertia_))

    return ks, inertias


def pick_representative_comments(
    X: np.ndarray,
    labels: np.ndarray,
    comments: List[str],
    max_per_cluster: int,
) -> List[Tuple[int, List[str]]]:
    """
    For each cluster, pick up to max_per_cluster comments closest to the centroid.
    Returns list of (cluster_id, [comments...]).
    """
    n_clusters = int(labels.max()) + 1
    reps: List[Tuple[int, List[str]]] = []

    for cluster_id in range(n_clusters):
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            reps.append((cluster_id, []))
            continue

        cluster_vecs = X[idx]
        centroid = cluster_vecs.mean(axis=0, keepdims=True)
        # Euclidean distance to centroid
        dists = np.linalg.norm(cluster_vecs - centroid, axis=1)
        order = np.argsort(dists)

        chosen = []
        for j in order[:max_per_cluster]:
            chosen.append(comments[idx[j]])
        reps.append((cluster_id, chosen))

    return reps


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster comments using BERT/SentenceTransformer embeddings and KMeans++, "
            "choosing k via an elbow method and saving an elbow plot."
        )
    )
    # Allow input either positionally or via -i/--input for consistency
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to input CSV file (can also be given via -i/--input).",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_opt",
        help="Path to input CSV file (alternative to positional argument).",
    )
    parser.add_argument(
        "--text-col",
        default="text",
        help="CSV column name containing the comment text (default: text).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV encoding (default: utf-8). Try utf-8-sig if needed.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_prefix",
        default="bert_kmeans",
        help=(
            "Output file prefix (no extension). "
            "Files like '<prefix>_elbow.png', '<prefix>_comments.csv' will be created "
            "(default: bert_kmeans)."
        ),
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: sentence-transformers/all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding computation (default: 64).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for SentenceTransformer (e.g. 'cpu', 'cuda'). Default: auto-detect.",
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=5,
        help="Minimum k for KMeans elbow search (default: 5).",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=15,
        help="Maximum k for KMeans elbow search (default: 15).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for KMeans (default: 42).",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=None,
        help="Optional maximum number of comments to sample for speed (default: use all).",
    )
    parser.add_argument(
        "--reps-per-cluster",
        type=int,
        default=4,
        help="Max representative comments to keep per cluster (default: 4).",
    )
    parser.add_argument(
        "--fixed-k",
        type=int,
        default=None,
        help=(
            "If set, skip elbow suggestion and use this fixed k "
            "for the final clustering."
        ),
    )

    args = parser.parse_args()

    input_path_str = args.input_opt or args.input
    if not input_path_str:
        parser.error("You must provide an input CSV either positionally or with -i/--input.")

    in_path = Path(input_path_str).expanduser()

    # Base path (without suffix) for all outputs; we will derive concrete filenames from it.
    base_prefix = Path(args.output_prefix).expanduser()
    base_prefix.parent.mkdir(parents=True, exist_ok=True)

    comments = load_comments_from_csv(
        in_path, text_col=args.text_col, encoding=args.encoding
    )
    if len(comments) < 20:
        raise SystemExit(
            f"Too few non-empty comments loaded ({len(comments)}). Check the input file."
        )

    if args.max_comments is not None and args.max_comments > 0 and len(comments) > args.max_comments:
        rng = np.random.default_rng(args.random_state)
        idx = rng.choice(len(comments), size=args.max_comments, replace=False)
        comments = [comments[i] for i in idx.tolist()]

    print(f"Loaded {len(comments)} cleaned comments.")

    embeddings = compute_embeddings(
        comments,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Embeddings shape: {embeddings.shape} (comments x dims)")

    min_k = int(args.min_k)
    max_k = int(args.max_k)
    if min_k < 2:
        raise SystemExit("--min-k must be >= 2.")
    if max_k < min_k:
        raise SystemExit("--max-k must be >= --min-k.")
    if embeddings.shape[0] <= max_k:
        raise SystemExit(
            f"Need more comments than max_k. Got {embeddings.shape[0]} comments but max_k={max_k}."
        )

    if args.fixed_k is None:
        print(f"Running KMeans elbow search for k in [{min_k}, {max_k}]...")
        ks, inertias = run_kmeans_range(
            embeddings,
            min_k=min_k,
            max_k=max_k,
            random_state=args.random_state,
        )
        suggested_k = suggest_elbow_k(ks, inertias)

        # Save elbow metrics
        metrics_path = base_prefix.with_name(base_prefix.name + "_k_metrics.csv")
        with metrics_path.open("w", encoding="utf-8-sig", newline="") as f:
            f.write("k,inertia\n")
            for k, inertia in zip(ks, inertias):
                f.write(f"{k},{inertia}\n")

        # Plot elbow curve
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ks, inertias, marker="o", linewidth=2)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Inertia (within-cluster SSE)")
        ax.set_title("Elbow curve for KMeans++ on BERT embeddings")
        ax.grid(True, alpha=0.25)
        ax.axvline(suggested_k, linestyle="--", linewidth=1.5, color="tab:red", alpha=0.9)
        ax.text(
            suggested_k,
            float(min(inertias)),
            f" suggested k={suggested_k}",
            color="tab:red",
            va="bottom",
            ha="left",
        )
        fig.tight_layout()
        elbow_path = base_prefix.with_name(base_prefix.name + "_elbow.png")
        fig.savefig(elbow_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved elbow metrics to: {metrics_path}")
        print(f"Saved elbow plot to:    {elbow_path}")
        k_final = suggested_k
    else:
        k_final = int(args.fixed_k)
        print(f"Using fixed k={k_final} (elbow search disabled).")

    print(f"Fitting final KMeans with k={k_final}...")
    km_final = KMeans(
        n_clusters=k_final,
        init="k-means++",
        n_init=20,
        max_iter=500,
        random_state=args.random_state,
    )
    labels = km_final.fit_predict(embeddings)

    # Save per-comment cluster assignments
    comments_path = base_prefix.with_name(base_prefix.name + "_comments.csv")
    with comments_path.open("w", encoding="utf-8-sig", newline="") as f:
        f.write("comment,cluster\n")
        for text, cluster_id in zip(comments, labels.tolist()):
            # Escape quotes and newlines for a simple CSV format
            safe_text = text.replace('"', '""').replace("\n", " ").replace("\r", " ")
            f.write(f"\"{safe_text}\",{cluster_id}\n")

    # Representative comments per cluster
    reps = pick_representative_comments(
        embeddings,
        labels=labels,
        comments=comments,
        max_per_cluster=max(2, int(args.reps_per_cluster)),
    )
    reps_path = base_prefix.with_name(base_prefix.name + "_cluster_representatives.csv")
    with reps_path.open("w", encoding="utf-8-sig", newline="") as f:
        f.write("cluster,rank,comment\n")
        for cluster_id, comment_list in reps:
            for rank, text in enumerate(comment_list, start=1):
                safe_text = text.replace('"', '""').replace("\n", " ").replace("\r", " ")
                f.write(f"{cluster_id},{rank},\"{safe_text}\"\n")

    print(f"Saved clustered comments to:        {comments_path}")
    print(f"Saved representative comments to:   {reps_path}")
    print(f"Finished. Final number of clusters: {k_final}")


if __name__ == "__main__":
    main()
