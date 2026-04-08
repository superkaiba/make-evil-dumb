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

    # 1. Distribution plot (raw)
    logger.info("Plotting projection distributions...")
    dist_plot = plot_projection_distributions(projections_by_corpus, str(output_dir))
    summary["distribution_plot"] = dist_plot

    # 2. Length residualization on COMBINED corpus
    all_projections = []
    corpus_offsets = {}  # track which records belong to which corpus
    for corpus_name, projections in projections_by_corpus.items():
        corpus_offsets[corpus_name] = (
            len(all_projections),
            len(all_projections) + len(projections),
        )
        all_projections.extend(projections)

    if all_projections:
        # Check linearity
        logger.info("Checking residualization linearity...")
        linearity = check_residualization_linearity(all_projections, str(output_dir))
        summary["linearity_check"] = linearity

        # Residualize on combined corpus
        logger.info("Residualizing length confound...")
        all_projections, resid_stats = residualize_length(
            all_projections, use_polynomial=linearity["use_polynomial"]
        )
        summary["residualization"] = resid_stats

        # Update corpus-level records with residualized values
        for corpus_name, (start, end) in corpus_offsets.items():
            projections_by_corpus[corpus_name] = all_projections[start:end]

    # 3. Per-corpus analysis (on both raw and residualized)
    for corpus_name, projections in projections_by_corpus.items():
        corpus_dir = output_dir / corpus_name
        corpus_dir.mkdir(parents=True, exist_ok=True)
        corpus_summary = {}

        # Length confound (raw)
        logger.info(f"Checking length confound for {corpus_name}...")
        confound_stats = check_length_confound(projections)
        corpus_summary["length_confound_raw"] = confound_stats
        confound_plot = plot_length_confound(projections, str(corpus_dir))
        corpus_summary["length_confound_plot"] = confound_plot

        confound_path = corpus_dir / "length_confound.json"
        with open(confound_path, "w") as f:
            json.dump(confound_stats, f, indent=2)

        # Tail analysis on RAW projections
        logger.info(f"Collecting tails (raw) for {corpus_name}...")
        tails = collect_tails(projections, tail_fraction=tail_fraction)
        tail_path = corpus_dir / "tail_docs.jsonl"
        save_tail_docs(tails, str(tail_path))
        corpus_summary["tail_path"] = str(tail_path)
        corpus_summary["tail_counts"] = {k: len(v) for k, v in tails.items()}

        if tails["top"] and tails["bottom"]:
            logger.info(f"Computing TF-IDF comparison (raw) for {corpus_name}...")
            tfidf_result = compute_tfidf_comparison(tails["top"], tails["bottom"], str(corpus_dir))
            corpus_summary["tfidf_top_5_raw"] = [kw[0] for kw in tfidf_result["top_keywords"][:5]]
            corpus_summary["tfidf_bottom_5_raw"] = [
                kw[0] for kw in tfidf_result["bottom_keywords"][:5]
            ]

        # Tail analysis on RESIDUALIZED projections
        has_resid = any("projection_residualized" in r for r in projections)
        if has_resid:
            logger.info(f"Collecting tails (residualized) for {corpus_name}...")
            # Swap projection field for tail collection
            resid_records = [{**r, "projection": r["projection_residualized"]} for r in projections]
            tails_resid = collect_tails(resid_records, tail_fraction=tail_fraction)
            resid_dir = corpus_dir / "residualized"
            resid_dir.mkdir(parents=True, exist_ok=True)
            save_tail_docs(tails_resid, str(resid_dir / "tail_docs.jsonl"))

            if tails_resid["top"] and tails_resid["bottom"]:
                logger.info(f"Computing TF-IDF comparison (residualized) for {corpus_name}...")
                tfidf_resid = compute_tfidf_comparison(
                    tails_resid["top"],
                    tails_resid["bottom"],
                    str(resid_dir),
                )
                corpus_summary["tfidf_top_5_residualized"] = [
                    kw[0] for kw in tfidf_resid["top_keywords"][:5]
                ]
                corpus_summary["tfidf_bottom_5_residualized"] = [
                    kw[0] for kw in tfidf_resid["bottom_keywords"][:5]
                ]

        summary[corpus_name] = corpus_summary

    # Save summary
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved analysis summary to {summary_path}")

    return summary


# =====================================================================
# Length residualization
# =====================================================================


def check_residualization_linearity(projections: list[dict], output_dir: str) -> dict:
    """Check if length-projection relationship is linear or needs polynomial.

    Fits linear, poly2, poly3 regressions and compares R-squared.
    If poly2 improves R-squared by >0.02, recommends polynomial residualization.

    Args:
        projections: List of projection record dicts.
        output_dir: Directory to save diagnostic plot.

    Returns:
        Dict with R-squared values and recommendation.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proj = np.array([r["projection"] for r in projections])
    tc = np.array([r["token_count"] for r in projections]).reshape(-1, 1)

    lr = LinearRegression().fit(tc, proj)
    r2_linear = lr.score(tc, proj)

    poly2 = PolynomialFeatures(2)
    tc_p2 = poly2.fit_transform(tc)
    lr2 = LinearRegression().fit(tc_p2, proj)
    r2_poly2 = lr2.score(tc_p2, proj)

    poly3 = PolynomialFeatures(3)
    tc_p3 = poly3.fit_transform(tc)
    lr3 = LinearRegression().fit(tc_p3, proj)
    r2_poly3 = lr3.score(tc_p3, proj)

    use_poly = (r2_poly2 - r2_linear) > 0.02

    logger.info(
        f"Linearity check: R2_linear={r2_linear:.4f}, R2_poly2={r2_poly2:.4f}, "
        f"R2_poly3={r2_poly3:.4f}, use_poly={use_poly}"
    )

    # Diagnostic plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sample_idx = np.random.choice(len(proj), min(50_000, len(proj)), replace=False)
    ax.scatter(tc[sample_idx, 0], proj[sample_idx], alpha=0.05, s=2, rasterized=True)

    tc_sorted = np.sort(tc[:, 0])
    ax.plot(
        tc_sorted,
        lr.predict(tc_sorted.reshape(-1, 1)),
        "r-",
        linewidth=2,
        label=f"Linear R²={r2_linear:.4f}",
    )
    tc_sorted_p2 = poly2.transform(tc_sorted.reshape(-1, 1))
    ax.plot(
        tc_sorted, lr2.predict(tc_sorted_p2), "g--", linewidth=2, label=f"Poly2 R²={r2_poly2:.4f}"
    )

    ax.set_xlabel("Token count")
    ax.set_ylabel("Projection")
    ax.set_title("Linearity check: projection vs token count")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "linearity_check.png", dpi=150)
    plt.close(fig)

    return {
        "r2_linear": float(r2_linear),
        "r2_poly2": float(r2_poly2),
        "r2_poly3": float(r2_poly3),
        "use_polynomial": use_poly,
    }


def residualize_length(
    projections: list[dict], use_polynomial: bool = False
) -> tuple[list[dict], dict]:
    """Regress out token_count from projection values.

    Residualizes on the COMBINED corpus (all projections passed in).

    Args:
        projections: List of projection record dicts.
        use_polynomial: If True, use degree-2 polynomial instead of linear.

    Returns:
        (updated_projections, stats) where each record gets a 'projection_residualized' field.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    proj = np.array([r["projection"] for r in projections])
    tc = np.array([r["token_count"] for r in projections]).reshape(-1, 1)

    if use_polynomial:
        poly = PolynomialFeatures(2)
        X = poly.fit_transform(tc)
    else:
        X = tc

    model = LinearRegression().fit(X, proj)
    residuals = proj - model.predict(X)
    r_squared = model.score(X, proj)

    for r, resid in zip(projections, residuals):
        r["projection_residualized"] = round(float(resid), 6)

    # Verify residualization worked
    from scipy.stats import pearsonr, spearmanr

    resid_r, _ = pearsonr(residuals, tc[:, 0])
    resid_rho, _ = spearmanr(residuals, tc[:, 0])
    logger.info(
        f"Residualization: R²={r_squared:.4f}, "
        f"residual pearson_r={resid_r:.6f}, spearman_rho={resid_rho:.6f}"
    )

    return projections, {
        "r_squared": float(r_squared),
        "residual_pearson_r": float(resid_r),
        "residual_spearman_rho": float(resid_rho),
        "method": "poly2" if use_polynomial else "linear",
    }


# =====================================================================
# Random direction control
# =====================================================================


def load_hidden_vectors(path: str) -> np.ndarray:
    """Load saved hidden vectors from JSONL file.

    Returns:
        Array of shape (n_docs, hidden_dim).
    """
    vectors = []
    with open(path) as f:
        for line in f:
            record = json.loads(line.strip())
            vectors.append(record["hidden"])
    arr = np.array(vectors, dtype=np.float32)
    logger.info(f"Loaded {arr.shape[0]} hidden vectors from {path}, dim={arr.shape[1]}")
    return arr


def random_direction_control(
    fw_hidden_path: str,
    lmsys_hidden_path: str,
    real_axis: np.ndarray,
    n_random: int = 10,
) -> dict:
    """Compare real axis corpus separation against random directions.

    Projects saved hidden vectors onto the real axis and 10 random unit vectors.
    Reports Cohen's d for each, plus a z-score for the real axis.

    Args:
        fw_hidden_path: Path to FineWeb hidden vectors JSONL.
        lmsys_hidden_path: Path to LMSYS hidden vectors JSONL.
        real_axis: The real axis vector (normalized).
        n_random: Number of random directions to test.

    Returns:
        Dict with real_axis_cohens_d, random stats, and z_score.
    """
    fw_vecs = load_hidden_vectors(fw_hidden_path)
    lmsys_vecs = load_hidden_vectors(lmsys_hidden_path)
    ax_np = np.asarray(real_axis, dtype=np.float32)

    def cohens_d(a, b):
        pooled_std = np.sqrt(
            (a.var() * (len(a) - 1) + b.var() * (len(b) - 1)) / (len(a) + len(b) - 2)
        )
        if pooled_std < 1e-8:
            return 0.0
        return float((a.mean() - b.mean()) / pooled_std)

    real_d = cohens_d(fw_vecs @ ax_np, lmsys_vecs @ ax_np)

    random_ds = []
    rng = np.random.default_rng(42)
    for _ in range(n_random):
        rand_dir = rng.standard_normal(ax_np.shape[0]).astype(np.float32)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        d = cohens_d(fw_vecs @ rand_dir, lmsys_vecs @ rand_dir)
        random_ds.append(d)

    random_abs = np.abs(random_ds)
    z = (abs(real_d) - random_abs.mean()) / max(random_abs.std(), 1e-8)

    logger.info(
        f"Random direction control: real_d={real_d:.4f}, "
        f"random_mean_d={np.mean(random_ds):.4f} (±{np.std(random_ds):.4f}), z={z:.4f}"
    )

    return {
        "real_axis_cohens_d": float(real_d),
        "random_cohens_ds": [float(d) for d in random_ds],
        "random_mean_d": float(np.mean(random_ds)),
        "random_std_d": float(np.std(random_ds)),
        "random_max_abs_d": float(np.max(random_abs)),
        "z_score": float(z),
    }


# =====================================================================
# Effect sizes
# =====================================================================


def compute_effect_sizes(
    fw_projections: list[dict],
    lmsys_projections: list[dict],
    field: str = "projection",
) -> dict:
    """Compute Cohen's d and Mann-Whitney U for corpus separation.

    Args:
        fw_projections: FineWeb projection records.
        lmsys_projections: LMSYS projection records.
        field: Which field to compare ('projection' or 'projection_residualized').

    Returns:
        Dict with means, Cohen's d, and Mann-Whitney p-value.
    """
    from scipy.stats import mannwhitneyu

    fw = np.array([r[field] for r in fw_projections if field in r])
    lmsys = np.array([r[field] for r in lmsys_projections if field in r])

    if len(fw) == 0 or len(lmsys) == 0:
        return {"error": f"No data for field '{field}'"}

    pooled_std = np.sqrt(
        (fw.var() * (len(fw) - 1) + lmsys.var() * (len(lmsys) - 1)) / (len(fw) + len(lmsys) - 2)
    )
    d = float((fw.mean() - lmsys.mean()) / pooled_std) if pooled_std > 1e-8 else 0.0

    # Mann-Whitney on subsample if large (avoids slow computation)
    max_mw = 50_000
    fw_mw = fw if len(fw) <= max_mw else np.random.choice(fw, max_mw, replace=False)
    lmsys_mw = lmsys if len(lmsys) <= max_mw else np.random.choice(lmsys, max_mw, replace=False)
    _, p_val = mannwhitneyu(fw_mw, lmsys_mw, alternative="two-sided")

    direction = "FineWeb > LMSYS" if fw.mean() > lmsys.mean() else "LMSYS > FineWeb"

    logger.info(
        f"Effect sizes ({field}): fw_mean={fw.mean():.4f}, lmsys_mean={lmsys.mean():.4f}, "
        f"Cohen's d={d:.4f}, direction={direction}"
    )

    return {
        "fw_mean": float(fw.mean()),
        "fw_std": float(fw.std()),
        "lmsys_mean": float(lmsys.mean()),
        "lmsys_std": float(lmsys.std()),
        "cohens_d": d,
        "direction": direction,
        "mannwhitney_p": float(p_val),
        "fw_n": len(fw),
        "lmsys_n": len(lmsys),
    }
