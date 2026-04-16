"""
Test whether persona identity is 1D (captured by centroid) or multi-dimensional.

If persona identity is fully captured by the centroid, then after subtracting
per-persona centroids, the residuals should be persona-independent.

Tests (ordered fastest → slowest):
1. Nearest-neighbor same-persona rate (simplest, most direct)
2. k-NN classification (5-fold CV)
3. Per-persona covariance divergence
4. Progressive removal of between-persona PCs
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

ACTIVATIONS_DIR = Path("/workspace/gemma2-27b-aim1/full/activations")
OUTPUT_DIR = Path("/workspace/explore-persona-space/scripts/identity_dim_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [15, 18, 20, 22, 25, 28, 30, 33, 36]
FOCUS_LAYERS = [15, 22, 30]
N_FOLDS = 5
PCA_DIM = 100
RANDOM_STATE = 42
# Subsample for speed: use 400/persona instead of 1200
SUBSAMPLE = 400


def load_all_activations(layer_idx, subsample=None):
    """Load all persona activations at a specific layer index."""
    X_all, y_all, persona_names = [], [], []
    rng = np.random.RandomState(RANDOM_STATE)
    pt_files = sorted(ACTIVATIONS_DIR.glob("*.pt"))
    for pid, pt_file in enumerate(pt_files):
        name = pt_file.stem
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        keys = sorted(data.keys())
        if subsample and subsample < len(keys):
            keys = [keys[i] for i in rng.choice(len(keys), subsample, replace=False)]
        vecs = [data[key][layer_idx].float().numpy() for key in keys]
        arr = np.stack(vecs)
        X_all.append(arr)
        y_all.extend([pid] * len(arr))
        persona_names.append(name)
    X = np.concatenate(X_all, axis=0)
    y = np.array(y_all)
    return X, y, persona_names


def subtract_centroids(X, y):
    """Subtract per-class centroid from each sample."""
    X_res = X.copy()
    centroids = {}
    for c in np.unique(y):
        mask = y == c
        centroid = X[mask].mean(axis=0)
        centroids[c] = centroid
        X_res[mask] -= centroid
    return X_res, centroids


def project_out_directions(X, directions):
    """Project out directions from X."""
    Q, _ = np.linalg.qr(directions.T)
    return X - (X @ Q) @ Q.T


def nn_same_persona_rate(X, y):
    """Fraction of samples whose nearest neighbor shares the same persona."""
    nn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(X)
    _, indices = nn.kneighbors(X)
    return float((y[indices[:, 1]] == y).mean())


def knn_cv(X, y, k=5, n_folds=N_FOLDS):
    """k-NN cross-validated accuracy."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    accs = []
    for tr, te in skf.split(X, y):
        clf = KNeighborsClassifier(n_neighbors=k, algorithm="ball_tree")
        clf.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs)), float(np.std(accs))


def covariance_divergence(X_res, y, n_pcs=30, n_null=5):
    """Do personas have different manifold shapes after centroid removal?"""
    pca = PCA(n_components=n_pcs, random_state=RANDOM_STATE)
    Xp = pca.fit_transform(X_res)
    classes = np.unique(y)
    covs = {c: np.cov(Xp[y == c].T) for c in classes}

    frob = []
    for i, ci in enumerate(classes):
        for cj in classes[i + 1:]:
            frob.append(np.linalg.norm(covs[ci] - covs[cj], "fro"))

    rng = np.random.RandomState(RANDOM_STATE)
    null_frob = []
    for _ in range(n_null):
        ys = rng.permutation(y)
        nc = {c: np.cov(Xp[ys == c].T) for c in classes}
        for i, ci in enumerate(classes):
            for cj in classes[i + 1:]:
                null_frob.append(np.linalg.norm(nc[ci] - nc[cj], "fro"))

    return {
        "mean_frob": float(np.mean(frob)),
        "null_mean_frob": float(np.mean(null_frob)),
        "effect_sigma": float((np.mean(frob) - np.mean(null_frob)) / max(np.std(null_frob), 1e-10)),
    }


def main():
    results = {}
    chance = 1.0 / 49

    for layer in FOCUS_LAYERS:
        layer_idx = LAYERS.index(layer)
        print(f"\n{'='*60}")
        print(f"LAYER {layer}")
        print(f"{'='*60}")

        t0 = time.time()
        X_raw, y, names = load_all_activations(layer_idx, subsample=SUBSAMPLE)
        n_per = SUBSAMPLE or 1200
        print(f"Loaded: {X_raw.shape} ({n_per}/persona) in {time.time()-t0:.1f}s")

        pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
        X = pca.fit_transform(X_raw)
        vk = pca.explained_variance_ratio_.sum()
        print(f"PCA → {PCA_DIM}-D, var retained: {vk:.4f}")

        X_res, centroids = subtract_centroids(X, y)

        # Null residuals (shuffle labels, subtract wrong centroids)
        rng = np.random.RandomState(RANDOM_STATE + 1)
        y_shuf = rng.permutation(y)
        X_null, _ = subtract_centroids(X, y_shuf)

        lr = {"layer": layer, "chance": chance, "n_per_persona": n_per,
              "pca_dim": PCA_DIM, "pca_var": float(vk)}

        # ── Test 1: Nearest-neighbor same-persona rate ──
        print("\n[Test 1] Nearest-neighbor same-persona rate")
        t0 = time.time()
        nn_orig = nn_same_persona_rate(X, y)
        nn_res = nn_same_persona_rate(X_res, y)
        nn_null = nn_same_persona_rate(X_null, y)
        print(f"  Original:  {nn_orig:.4f}")
        print(f"  Residuals: {nn_res:.4f}")
        print(f"  Null:      {nn_null:.4f}")
        print(f"  Chance:    {chance:.4f}")
        print(f"  ({time.time()-t0:.1f}s)")

        lr["nn_same_persona"] = {
            "original": nn_orig, "residuals": nn_res,
            "null": nn_null, "chance": chance,
        }

        # ── Test 2: k-NN classification ──
        print("\n[Test 2] k-NN classification (k=5, 5-fold CV)")
        t0 = time.time()
        knn_orig, knn_orig_s = knn_cv(X, y)
        knn_res, knn_res_s = knn_cv(X_res, y)
        knn_null, knn_null_s = knn_cv(X_null, y)
        print(f"  Original:  {knn_orig:.4f} +/- {knn_orig_s:.4f}")
        print(f"  Residuals: {knn_res:.4f} +/- {knn_res_s:.4f}")
        print(f"  Null:      {knn_null:.4f} +/- {knn_null_s:.4f}")
        print(f"  Chance:    {chance:.4f}")
        print(f"  ({time.time()-t0:.1f}s)")

        lr["knn_cv"] = {
            "original": {"mean": knn_orig, "std": knn_orig_s},
            "residuals": {"mean": knn_res, "std": knn_res_s},
            "null": {"mean": knn_null, "std": knn_null_s},
        }

        # ── Test 3: Covariance divergence ──
        print("\n[Test 3] Per-persona covariance divergence (residuals)")
        t0 = time.time()
        cd = covariance_divergence(X_res, y)
        print(f"  Frobenius:  {cd['mean_frob']:.1f}")
        print(f"  Null:       {cd['null_mean_frob']:.1f}")
        print(f"  Effect:     {cd['effect_sigma']:.1f} sigma")
        print(f"  ({time.time()-t0:.1f}s)")
        lr["covariance_divergence"] = cd

        # ── Test 4: Progressive removal of between-persona PCs ──
        print("\n[Test 4] Progressive projection (kNN on residuals after removing btw-PCs)")
        centroid_mat = np.array([centroids[c] for c in sorted(centroids.keys())])
        cc = centroid_mat - centroid_mat.mean(axis=0)
        pca_btw = PCA(n_components=min(48, PCA_DIM), random_state=RANDOM_STATE)
        pca_btw.fit(cc)
        btw_pcs = pca_btw.components_

        prog = []
        for k in [1, 2, 5, 10, 20, 48]:
            X_proj = project_out_directions(X_res, btw_pcs[:k])
            t0 = time.time()
            acc, std = knn_cv(X_proj, y)
            var_rm = float(pca_btw.explained_variance_ratio_[:k].sum())
            nn_rate = nn_same_persona_rate(X_proj, y)
            print(f"  k={k:2d} ({var_rm:.0%} btw-var): kNN={acc:.4f}, NN={nn_rate:.4f}  "
                  f"({time.time()-t0:.1f}s)")
            prog.append({"k": k, "btw_var_removed": var_rm,
                          "knn_acc": acc, "nn_same": nn_rate})
        lr["progressive_projection"] = prog

        results[f"L{layer}"] = lr

    # ── Save ──
    out = OUTPUT_DIR / "identity_dimensionality_test.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY — Is persona identity >1D?")
    print(f"{'='*70}")
    print(f"{'Layer':<8} {'Orig kNN':<12} {'Resid kNN':<12} {'Null kNN':<12} "
          f"{'Resid NN%':<12} {'Cov σ':<8}")
    print("-" * 70)
    for lk, lr in results.items():
        kn = lr["knn_cv"]
        nn = lr["nn_same_persona"]
        cd = lr["covariance_divergence"]
        print(f"{lr['layer']:<8} {kn['original']['mean']:<12.1%} "
              f"{kn['residuals']['mean']:<12.1%} {kn['null']['mean']:<12.1%} "
              f"{nn['residuals']:<12.1%} {cd['effect_sigma']:<8.1f}")
    print(f"Chance:  {'':12s} {chance:<12.1%}")

    print("\nProgressive projection (Layer 22):")
    if "L22" in results:
        for p in results["L22"]["progressive_projection"]:
            marker = " ← chance" if p["knn_acc"] < chance * 2 else ""
            print(f"  Remove {p['k']:2d} btw-PCs ({p['btw_var_removed']:.0%}): "
                  f"kNN={p['knn_acc']:.1%}, NN={p['nn_same']:.1%}{marker}")

    print("\nINTERPRETATION:")
    for lk, lr in results.items():
        res = lr["knn_cv"]["residuals"]["mean"]
        null = lr["knn_cv"]["null"]["mean"]
        ratio = res / chance
        if res > null * 1.5 and ratio > 3:
            print(f"  {lk}: Residuals classifiable at {res:.1%} ({ratio:.0f}x chance) → "
                  f"identity is MULTI-DIMENSIONAL")
        elif res > null * 1.2:
            print(f"  {lk}: Residuals weakly classifiable ({res:.1%}) → "
                  f"identity has some multi-D signal")
        else:
            print(f"  {lk}: Residuals ~null ({res:.1%}) → identity is ~1D")


if __name__ == "__main__":
    main()
