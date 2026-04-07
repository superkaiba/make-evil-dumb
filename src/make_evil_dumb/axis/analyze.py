"""Analyze tail documents from corpus projection onto the assistant axis.

Takes JSONL files of projection results and produces:
1. Distribution plots (overlaid histograms of projection values)
2. Length confound check (correlation of projection vs token count)
3. TF-IDF comparison (keywords distinguishing top vs bottom tail)
4. Tail collection (top/bottom 0.1% + random sample)
"""

import json
import logging
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless servers

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def load_projections(path: str) -> list[dict]:
    """Load JSONL projection results.

    Each line should be a JSON object with at least:
        {"doc_id": int, "projection": float, "token_count": int, "text_snippet": str}

    Args:
        path: Path to .jsonl file.

    Returns:
        List of projection record dicts.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} projections from {path}")
    return records


def collect_tails(
    projections: list[dict],
    tail_fraction: float = 0.001,
    random_sample_size: int | None = None,
) -> dict:
    """Extract top and bottom tail documents plus a random sample.

    Args:
        projections: List of projection record dicts.
        tail_fraction: Fraction of documents for each tail (default 0.1%).
        random_sample_size: Number of random docs to sample. Defaults to same
            count as each tail.

    Returns:
        Dict with keys "top", "bottom", "random", each a list of record dicts,
        sorted by projection value (descending for top, ascending for bottom).
    """
    if not projections:
        return {"top": [], "bottom": [], "random": []}

    sorted_by_proj = sorted(projections, key=lambda d: d["projection"])
    n = len(sorted_by_proj)
    tail_count = max(1, int(n * tail_fraction))

    if random_sample_size is None:
        random_sample_size = tail_count

    bottom = sorted_by_proj[:tail_count]
    top = sorted_by_proj[-tail_count:][::-1]  # descending

    # Random sample from the middle 90%
    middle_start = tail_count
    middle_end = n - tail_count
    if middle_end <= middle_start:
        random_docs = random.sample(sorted_by_proj, min(random_sample_size, n))
    else:
        middle = sorted_by_proj[middle_start:middle_end]
        random_docs = random.sample(middle, min(random_sample_size, len(middle)))

    logger.info(
        f"Tails: top={len(top)}, bottom={len(bottom)}, random={len(random_docs)} "
        f"(tail_fraction={tail_fraction}, n={n})"
    )
    return {"top": top, "bottom": bottom, "random": random_docs}


def plot_projection_distributions(
    projections_by_corpus: dict[str, list[dict]],
    output_dir: str,
    bins: int = 200,
    figsize: tuple[float, float] = (12, 6),
) -> str:
    """Plot overlaid histograms of projection values for multiple corpora.

    Args:
        projections_by_corpus: Mapping from corpus name to list of projection records.
        output_dir: Directory to save the plot.
        bins: Number of histogram bins.
        figsize: Figure size (width, height) in inches.

    Returns:
        Path to the saved plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, (corpus_name, records) in enumerate(projections_by_corpus.items()):
        values = [r["projection"] for r in records]
        color = colors[idx % len(colors)]
        ax.hist(
            values,
            bins=bins,
            alpha=0.5,
            label=f"{corpus_name} (n={len(values):,})",
            color=color,
            density=True,
        )
        # Mark mean
        mean_val = np.mean(values)
        ax.axvline(mean_val, color=color, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Projection onto assistant axis")
    ax.set_ylabel("Density")
    ax.set_title("Corpus projection distributions")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = output_dir / "projection_distributions.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved distribution plot to {plot_path}")
    return str(plot_path)


def compute_tfidf_comparison(
    top_docs: list[dict],
    bottom_docs: list[dict],
    output_dir: str,
    top_k: int = 30,
    text_key: str = "text_snippet",
) -> dict:
    """Compute TF-IDF and find keywords distinguishing top vs bottom tail.

    Fits a TF-IDF vectorizer on both tails, then compares mean TF-IDF vectors
    to find the most distinctive terms for each tail.

    Args:
        top_docs: Top-tail projection records.
        bottom_docs: Bottom-tail projection records.
        output_dir: Directory to save results.
        top_k: Number of top keywords to return per tail.
        text_key: Key for text content in record dicts.

    Returns:
        Dict with "top_keywords" and "bottom_keywords", each a list of
        (term, score_difference) tuples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_texts = [d.get(text_key, "") for d in top_docs if d.get(text_key)]
    bottom_texts = [d.get(text_key, "") for d in bottom_docs if d.get(text_key)]

    if not top_texts or not bottom_texts:
        logger.warning("Empty texts for TF-IDF comparison, skipping")
        return {"top_keywords": [], "bottom_keywords": []}

    all_texts = top_texts + bottom_texts
    labels = [1] * len(top_texts) + [0] * len(bottom_texts)

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    labels_arr = np.array(labels)
    top_mask = labels_arr == 1
    bottom_mask = labels_arr == 0

    # Mean TF-IDF per class
    mean_top = np.asarray(tfidf_matrix[top_mask].mean(axis=0)).flatten()
    mean_bottom = np.asarray(tfidf_matrix[bottom_mask].mean(axis=0)).flatten()

    # Difference scores
    diff = mean_top - mean_bottom

    top_keyword_indices = np.argsort(diff)[-top_k:][::-1]
    bottom_keyword_indices = np.argsort(diff)[:top_k]

    top_keywords = [(feature_names[i], float(diff[i])) for i in top_keyword_indices]
    bottom_keywords = [(feature_names[i], float(diff[i])) for i in bottom_keyword_indices]

    result = {"top_keywords": top_keywords, "bottom_keywords": bottom_keywords}

    # Save to JSON
    result_path = output_dir / "tfidf_comparison.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved TF-IDF comparison to {result_path}")

    # Plot top keywords as horizontal bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, keywords, title in [
        (axes[0], top_keywords, "Top tail keywords (high projection)"),
        (axes[1], bottom_keywords, "Bottom tail keywords (low projection)"),
    ]:
        terms = [kw[0] for kw in keywords[:20]]
        scores = [abs(kw[1]) for kw in keywords[:20]]
        y_pos = np.arange(len(terms))
        ax.barh(y_pos, scores, align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("TF-IDF score difference (absolute)")
        ax.set_title(title)

    fig.tight_layout()
    plot_path = output_dir / "tfidf_keywords.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved TF-IDF keyword plot to {plot_path}")

    return result


def check_length_confound(projections: list[dict]) -> dict:
    """Check correlation between projection values and token counts.

    Uses Pearson and Spearman correlation to detect length confounds.

    Args:
        projections: List of projection record dicts.

    Returns:
        Dict with correlation stats:
            - pearson_r, pearson_p
            - spearman_r, spearman_p
            - mean_projection, std_projection
            - mean_token_count, std_token_count
            - n_docs
    """
    from scipy import stats

    proj_values = np.array([r["projection"] for r in projections])
    token_counts = np.array([r["token_count"] for r in projections])

    pearson_r, pearson_p = stats.pearsonr(proj_values, token_counts)
    spearman_r, spearman_p = stats.spearmanr(proj_values, token_counts)

    result = {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "mean_projection": float(np.mean(proj_values)),
        "std_projection": float(np.std(proj_values)),
        "mean_token_count": float(np.mean(token_counts)),
        "std_token_count": float(np.std(token_counts)),
        "n_docs": len(projections),
    }

    logger.info(
        f"Length confound: pearson_r={pearson_r:.4f} (p={pearson_p:.2e}), "
        f"spearman_r={spearman_r:.4f} (p={spearman_p:.2e}), n={len(projections)}"
    )
    return result


def plot_length_confound(
    projections: list[dict],
    output_dir: str,
    max_points: int = 50_000,
    figsize: tuple[float, float] = (10, 6),
) -> str:
    """Scatter plot of projection value vs token count.

    Subsamples for readability if the dataset is large.

    Args:
        projections: List of projection record dicts.
        output_dir: Directory to save the plot.
        max_points: Maximum number of points to plot.
        figsize: Figure size in inches.

    Returns:
        Path to the saved plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(projections) > max_points:
        sample = random.sample(projections, max_points)
    else:
        sample = projections

    proj_values = [r["projection"] for r in sample]
    token_counts = [r["token_count"] for r in sample]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(token_counts, proj_values, alpha=0.05, s=2, rasterized=True)
    ax.set_xlabel("Token count")
    ax.set_ylabel("Projection onto assistant axis")
    ax.set_title(f"Length confound check (n={len(sample):,})")
    ax.grid(alpha=0.3)

    # Add correlation annotation
    from scipy import stats

    r, p = stats.pearsonr(proj_values, token_counts)
    ax.annotate(
        f"Pearson r={r:.4f}\np={p:.2e}",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    fig.tight_layout()
    plot_path = output_dir / "length_confound.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved length confound plot to {plot_path}")
    return str(plot_path)


def save_tail_docs(tails: dict, output_path: str) -> None:
    """Save tail documents to a JSONL file.

    Each line includes the original record plus a "tail_group" field
    ("top", "bottom", or "random").

    Args:
        tails: Dict with keys "top", "bottom", "random", each a list of records.
        output_path: Path to save the JSONL file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for group_name in ("top", "bottom", "random"):
            for record in tails.get(group_name, []):
                record_with_group = {**record, "tail_group": group_name}
                f.write(json.dumps(record_with_group) + "\n")
                count += 1

    logger.info(f"Saved {count} tail docs to {output_path}")


def run_full_analysis(
    projections_by_corpus: dict[str, list[dict]],
    output_dir: str,
    tail_fraction: float = 0.001,
) -> dict:
    """Run the complete analysis pipeline on projection results.

    This is a convenience function that runs all analysis steps:
    distribution plots, length confound, TF-IDF comparison, and tail collection.

    Args:
        projections_by_corpus: Mapping from corpus name to projection records.
        output_dir: Directory for all outputs.
        tail_fraction: Fraction for tail collection (default 0.1%).

    Returns:
        Summary dict with all results and paths.
    """
    output_dir = Path(output_dir)
    summary = {}

    # 1. Distribution plot
    logger.info("Plotting projection distributions...")
    dist_plot = plot_projection_distributions(projections_by_corpus, str(output_dir))
    summary["distribution_plot"] = dist_plot

    # 2. Per-corpus analysis
    for corpus_name, projections in projections_by_corpus.items():
        corpus_dir = output_dir / corpus_name
        corpus_dir.mkdir(parents=True, exist_ok=True)
        corpus_summary = {}

        # Length confound
        logger.info(f"Checking length confound for {corpus_name}...")
        confound_stats = check_length_confound(projections)
        corpus_summary["length_confound"] = confound_stats
        confound_plot = plot_length_confound(projections, str(corpus_dir))
        corpus_summary["length_confound_plot"] = confound_plot

        # Save confound stats
        confound_path = corpus_dir / "length_confound.json"
        with open(confound_path, "w") as f:
            json.dump(confound_stats, f, indent=2)

        # Tail collection
        logger.info(f"Collecting tails for {corpus_name}...")
        tails = collect_tails(projections, tail_fraction=tail_fraction)
        tail_path = corpus_dir / "tail_docs.jsonl"
        save_tail_docs(tails, str(tail_path))
        corpus_summary["tail_path"] = str(tail_path)
        corpus_summary["tail_counts"] = {k: len(v) for k, v in tails.items()}

        # TF-IDF comparison (top vs bottom)
        if tails["top"] and tails["bottom"]:
            logger.info(f"Computing TF-IDF comparison for {corpus_name}...")
            tfidf_result = compute_tfidf_comparison(
                tails["top"], tails["bottom"], str(corpus_dir)
            )
            corpus_summary["tfidf_top_5"] = [kw[0] for kw in tfidf_result["top_keywords"][:5]]
            corpus_summary["tfidf_bottom_5"] = [
                kw[0] for kw in tfidf_result["bottom_keywords"][:5]
            ]

        summary[corpus_name] = corpus_summary

    # Save summary
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved analysis summary to {summary_path}")

    return summary
