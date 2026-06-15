from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def detect_floating_points(
    points: np.ndarray,
    eps: float,
    min_samples: int = 10,
    min_cluster_pct: float = 0.0005,
    max_cluster_pct: float = 0.03,
    distance_ratio: float = 3.0,
) -> dict:
    """Detect floating point clusters using DBSCAN.

    Returns: dict with keys:
        floating_mask: bool array (N,) mark floating points
        floating_centroids: (K, 3) centers of floating clusters
        floating_radii: (K,) bounding sphere radii
        labels: (N,) cluster labels from DBSCAN
    """
    if len(points) < min_samples:
        return {
            "floating_mask": np.zeros(len(points), dtype=bool),
            "floating_centroids": np.zeros((0, 3)),
            "floating_radii": np.zeros(0),
            "labels": np.full(len(points), -1, dtype=int),
        }

    points_center = points.mean(axis=0)
    total = len(points)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    floating_mask = np.zeros(total, dtype=bool)
    floating_centroids = []
    floating_radii = []

    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            continue
        cluster_mask = labels == label
        cluster_size = cluster_mask.sum()
        cluster_pct = cluster_size / total

        if min_cluster_pct < cluster_pct < max_cluster_pct:
            cluster_points = points[cluster_mask]
            centroid = cluster_points.mean(axis=0)
            dist_to_center = np.linalg.norm(centroid - points_center)
            scene_scale = np.linalg.norm(points.std(axis=0))

            if dist_to_center > distance_ratio * scene_scale:
                floating_mask[cluster_mask] = True
                floating_centroids.append(centroid)
                radius = np.max(np.linalg.norm(cluster_points - centroid, axis=1))
                floating_radii.append(float(radius))

    return {
        "floating_mask": floating_mask,
        "floating_centroids": np.array(floating_centroids) if floating_centroids else np.zeros((0, 3)),
        "floating_radii": np.array(floating_radii) if floating_radii else np.zeros(0),
        "labels": labels if len(unique_labels) < 500 else np.full(total, -1, dtype=int),
    }


def detect_surface_breaks(
    points: np.ndarray,
    n_neighbors: int = 30,
    angle_threshold_deg: float = 35.0,
) -> dict:
    """Detect surface breaks via normal inconsistency.

    Returns: dict with:
        break_mask: bool array (N,)
        normals: (N, 3) estimated normals
        break_score: (N,) per-point break score (0-1)
    """
    if len(points) < n_neighbors:
        return {
            "break_mask": np.zeros(len(points), dtype=bool),
            "normals": np.zeros((len(points), 3)),
            "break_score": np.zeros(len(points)),
        }

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    _, indices = nbrs.kneighbors(points)

    normals = np.zeros((len(points), 3))
    for i, idx in enumerate(indices):
        neigh = points[idx] - points[idx].mean(axis=0)
        cov = neigh.T @ neigh
        vals, vecs = np.linalg.eigh(cov)
        normals[i] = vecs[:, 0]
        if normals[i, 2] < 0:
            normals[i] *= -1

    break_score = np.zeros(len(points))
    tree = KDTree(points)
    for i in range(len(points)):
        _, knn_idx = tree.query(points[i], k=n_neighbors)
        local_normals = normals[knn_idx]
        dot_products = np.abs(np.dot(local_normals, normals[i]))
        break_score[i] = 1.0 - dot_products.mean()

    angle_cos = np.cos(np.radians(angle_threshold_deg))
    break_mask = break_score > (1.0 - angle_cos)

    return {
        "break_mask": break_mask,
        "normals": normals,
        "break_score": break_score,
    }


def detect_boundary_blur(
    points: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    edge_width_threshold: int = 3,
) -> dict:
    """Detect boundary blur in depth maps.

    Returns: dict with:
        blur_mask: (H, W) bool mask of blurred edges
        edge_map: (H, W) edge detection result
    """
    import cv2

    if depth_map is None:
        return {"blur_mask": np.zeros(1, dtype=bool), "edge_map": np.zeros(1)}

    depth_vis = np.clip(depth_map, 0, np.percentile(depth_map, 99))
    if depth_vis.max() > depth_vis.min():
        depth_vis = ((depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min()) * 255).astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

    edges = cv2.Canny(depth_vis, 50, 150)
    kernel = np.ones((edge_width_threshold, edge_width_threshold), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    blur_mask = (dilated.astype(int) - eroded.astype(int)) > 0

    return {"blur_mask": blur_mask, "edge_map": edges.astype(bool)}


def detect_fusion_errors(
    points_list: list[np.ndarray],
    eps: float,
) -> dict:
    """Detect multi-view fusion errors by checking point density consistency.

    Args:
        points_list: list of (N_i, 3) point clouds from different views

    Returns:
        dict with fusion_error_mask per view
    """
    if len(points_list) < 2 or sum(len(p) for p in points_list) == 0:
        return {"fusion_error_mask": None, "density_ratios": []}

    all_points = np.concatenate(points_list, axis=0)
    tree = KDTree(all_points)

    densities = []
    for pts in points_list:
        if len(pts) == 0:
            densities.append(0.0)
            continue
        _, counts = tree.query_radius(pts, r=eps, return_distance=False, count_only=True)
        densities.append(float(counts.mean()))

    densities = np.array(densities)
    median_density = np.median(densities[densities > 0])
    density_ratios = densities / max(median_density, 1e-8)

    fusion_error_mask = (density_ratios < 0.5) | (density_ratios > 2.0)

    return {"fusion_error_mask": fusion_error_mask, "density_ratios": density_ratios.tolist()}
