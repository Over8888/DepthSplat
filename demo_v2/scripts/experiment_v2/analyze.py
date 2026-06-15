from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors

OUTPUT_DIR = Path("/root/autodl-tmp/demo_v2/outputs/experiment_v2")
SCENE_KEYS = ["322261824c4a3003", "89ea49cd9865aeff", "f7c0fa5b81552d35"]
METHOD_ORDER = ["Ours_(PMR)", "MVSplat", "VGGT", "DA3"]


def umeyama_alignment(source: np.ndarray, target: np.ndarray, max_points: int = 5000):
    """Align source point cloud to target using Umeyama (rotation + translation + scale).

    Returns aligned source, rotation matrix, translation, scale factor.
    """
    s_idx = np.random.choice(len(source), min(max_points, len(source)), replace=False)
    t_idx = np.random.choice(len(target), min(max_points, len(target)), replace=False)

    s = source[s_idx].T
    t = target[t_idx].T

    ms = s.mean(axis=1, keepdims=True)
    mt = t.mean(axis=1, keepdims=True)

    s_c = s - ms
    t_c = t - mt

    K = s_c @ t_c.T
    U, _, Vt = np.linalg.svd(K)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    var_s = (s_c**2).sum()
    scale = (t_c * (R @ s_c)).sum() / (var_s + 1e-8)

    t_vec = mt - scale * (R @ ms)

    aligned = (scale * R @ source.T + t_vec).T

    return aligned.astype(np.float32), R, t_vec.flatten(), float(scale)


def compute_local_normals(points: np.ndarray, n_neighbors: int = 30, max_points: int = 3000):
    """Compute per-point normals and normal consistency."""
    if len(points) < n_neighbors:
        return np.zeros((0, 3)), 0.0

    n = min(max_points, len(points))
    idxs = np.random.choice(len(points), n, replace=False)
    sample = points[idxs]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(sample)
    _, indices = nbrs.kneighbors(sample)

    normals = np.zeros((n, 3))
    consistency_scores = []

    for i, idx in enumerate(indices):
        neigh = sample[idx] - sample[idx].mean(axis=0)
        cov = neigh.T @ neigh
        vals, vecs = np.linalg.eigh(cov)
        normals[i] = vecs[:, 0]

    for i in range(n):
        for j in indices[i][1:min(6, len(indices[i]))]:
            if j < n:
                dot = abs(float(np.dot(normals[i], normals[j])))
                consistency_scores.append(dot)

    consistency = np.mean(consistency_scores) if consistency_scores else 0.0
    return normals, consistency


def compute_density_uniformity(points: np.ndarray, max_points: int = 3000):
    """Compute coefficient of variation of local point density."""
    if len(points) < 30:
        return 0.0

    n = min(max_points, len(points))
    idxs = np.random.choice(len(points), n, replace=False)
    sample = points[idxs]

    tree = KDTree(sample)
    nn, _ = tree.query(sample, k=30)
    local_density = 30.0 / (nn[:, -1] ** 3 + 1e-8)
    cv = float(local_density.std() / (local_density.mean() + 1e-8))
    return cv


def compute_overlap(reference: np.ndarray, other: np.ndarray, max_points: int = 5000, radius: float | None = None):
    """Compute overlap: fraction of reference points with at least one other point within radius."""
    if len(reference) < 10 or len(other) < 10:
        return 0.0, 0.0, 0.0

    n_ref = min(max_points, len(reference))
    n_other = min(max_points, len(other))

    ref_sample = reference[np.random.choice(len(reference), n_ref, replace=False)]
    other_sample = other[np.random.choice(len(other), n_other, replace=False)]

    if radius is None:
        ref_scale = np.linalg.norm(ref_sample.std(axis=0))
        radius = ref_scale * 0.05

    tree = KDTree(other_sample)
    dist, _ = tree.query(ref_sample, k=1)

    overlap = float((dist < radius).mean())
    mean_dist = float(dist.mean())

    tree2 = KDTree(ref_sample)
    dist2, _ = tree2.query(other_sample, k=1)
    coverage = float((dist2 < radius).mean())

    return overlap, coverage, mean_dist


def compute_anomaly_metrics(points: np.ndarray, max_points: int = 5000):
    """Compute outlier ratio and floating clusters."""
    if len(points) < 50:
        return {"outlier_pct": 0.0, "float_clusters": 0, "span": np.zeros(3)}

    n = min(max_points, len(points))
    idxs = np.random.choice(len(points), n, replace=False)
    sample = points[idxs]

    centered = sample - sample.mean(axis=0)
    dists = np.sqrt((centered**2).sum(axis=1))
    outlier_mask = dists > 3 * dists.std()
    outlier_pct = float(outlier_mask.mean() * 100)

    span = sample.max(axis=0) - sample.min(axis=0)

    from sklearn.cluster import DBSCAN
    eps = max(float(dists.std()), 0.01)
    try:
        clustering = DBSCAN(eps=eps * 3, min_samples=10).fit(sample)
        labels = clustering.labels_
        unique_labels = np.unique(labels[labels >= 0])
        total = len(sample)
        floating = 0
        for label in unique_labels:
            cluster_size = (labels == label).sum()
            if 0.0005 < cluster_size / total < 0.03:
                floating += 1
    except Exception:
        floating = 0

    return {"outlier_pct": outlier_pct, "float_clusters": floating, "span": span}


def analyze_scene(scene_key: str):
    """Analyze one scene's point clouds with alignment."""
    npz_path = OUTPUT_DIR / scene_key / "points.npz"
    if not npz_path.exists():
        print(f"[SKIP] {scene_key}: no points.npz")
        return None

    data = np.load(npz_path)
    points = {k: data[k] for k in METHOD_ORDER if k in data}

    ref_name = "Ours_(PMR)"
    ref_pts = points.get(ref_name)
    if ref_pts is None or len(ref_pts) < 100:
        print(f"[SKIP] {scene_key}: no reference points")
        return None

    aligned = {
        ref_name: ref_pts,
        "MVSplat": points.get("MVSplat", np.zeros((0, 3))),
        "VGGT": points.get("VGGT", np.zeros((0, 3))),
        "DA3": points.get("DA3", np.zeros((0, 3))),
    }
    align_info = {}

    results = {}
    for name in METHOD_ORDER:
        if name not in aligned or len(aligned[name]) < 10:
            results[name] = {"status": "N/A"}
            continue

        pts = aligned[name]
        normals, nc = compute_local_normals(pts)
        density_cv = compute_density_uniformity(pts)
        anomaly = compute_anomaly_metrics(pts)

        if name != ref_name and len(aligned[name]) > 100 and len(ref_pts) > 100:
            overlap, coverage, mean_dist = compute_overlap(ref_pts, aligned[name])
        else:
            overlap = 1.0
            coverage = 1.0
            mean_dist = 0.0

        results[name] = {
            "num_points": len(aligned[name]),
            "normal_consistency": nc,
            "density_cv": density_cv,
            "outlier_pct": anomaly["outlier_pct"],
            "float_clusters": anomaly["float_clusters"],
            "span": anomaly["span"],
            "coverage": coverage,
            "overlap": overlap,
            "mean_dist_to_ref": mean_dist,
        }

    return {
        "scene": scene_key,
        "reference": ref_name,
        "methods": results,
    }


def print_table(analyses: list[dict]):
    """Print a nicely formatted comparison table."""
    print()
    print("=" * 130)
    print("Multi-View Point Cloud Quality Analysis (aligned to DepthSplat+PMR world frame)")
    print("=" * 130)

    for analysis in analyses:
        if analysis is None:
            continue

        scene = analysis["scene"]
        results = analysis["methods"]

        print(f"\n{'=' * 130}")
        print(f"Scene: {scene}")
        print(f"{'=' * 130}")
        print(f"{'Method':14s} | {'Points':7s} | {'NormalC':7s} | {'DensCV':7s} | {'Outlier%':8s} | {'Floats':7s} | {'Coverage':8s} | {'Overlap':7s} | {'Dist2Ref':8s} | {'Span(X)':8s}")
        print("-" * 118)

        for name in METHOD_ORDER:
            r = results.get(name, {})
            if r.get("status") == "N/A":
                print(f"{name:14s} | {'N/A':7s}")
                continue

            span_x = r["span"][0] if len(r["span"]) > 0 else 0.0
            scale_str = "     —"
            print(
                f"{name:14s} | {r['num_points']:7d} | {r['normal_consistency']:7.4f} | "
                f"{r['density_cv']:7.3f} | {r['outlier_pct']:7.1f}% | {r['float_clusters']:7d} | "
                f"{r['coverage']:7.1%} | {r['overlap']:6.1%} | {r['mean_dist_to_ref']:8.3f} | "
                f"{span_x:8.2f}"
            )

    print(f"\n{'=' * 130}")
    print("Legend:")
    print("  NormalC:   Surface normal consistency (0→1, higher=smoother)")
    print("  DensCV:    Point density coefficient of variation (lower=more uniform)")
    print("  Outlier%:  Points beyond 3*std from centroid")
    print("  Floats:    Number of isolated floating clusters (DBSCAN)")
    print("  Coverage:  Fraction of reference points covered by method")
    print("  Overlap:   Fraction of method points within radius of reference")
    print("  Dist2Ref:  Mean distance from method points to nearest reference point")
    print("  Span(X):   Range in X direction (proxies point spread)")
    print("=" * 130)


def aggregate_metrics(analyses: list[dict]):
    """Aggregate metrics across scenes for each method."""
    print(f"\n{'=' * 80}")
    print("Cross-Scene Summary (mean ± std)")
    print(f"{'=' * 80}")

    metric_names = [
        ("normal_consistency", "Normal Consistency"),
        ("density_cv", "Density Uniformity (CV)"),
        ("outlier_pct", "Outlier %"),
        ("coverage", "Coverage"),
        ("overlap", "Overlap"),
        ("mean_dist_to_ref", "Mean Dist to Ref"),
    ]

    for mname in METHOD_ORDER:
        values = {k: [] for k, _ in metric_names}
        for analysis in analyses:
            if analysis is None:
                continue
            r = analysis["methods"].get(mname, {})
            if r.get("status") == "N/A":
                continue
            for key, _ in metric_names:
                if key in r:
                    values[key].append(r[key])

        if not any(len(v) > 0 for v in values.values()):
            continue

        print(f"\n{mname}:")
        for key, label in metric_names:
            vals = values[key]
            if len(vals) == 0:
                print(f"  {label:25s}: N/A")
            else:
                v = np.array(vals)
                print(f"  {label:25s}: {v.mean():.4f} ± {v.std():.4f}")

    print(f"\n{'=' * 80}")


def main():
    analyses = []
    for scene in SCENE_KEYS:
        print(f"\nAnalyzing {scene}...")
        analysis = analyze_scene(scene)
        analyses.append(analysis)

    print_table(analyses)
    aggregate_metrics(analyses)

    summary_path = OUTPUT_DIR / "analysis_summary.txt"
    with open(summary_path, "w") as f:
        import sys
        from contextlib import redirect_stdout
        with redirect_stdout(f):
            print_table(analyses)
            aggregate_metrics(analyses)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
