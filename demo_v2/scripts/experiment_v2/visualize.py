from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import PatchCollection
from scipy.spatial import KDTree


def _get_view_transform(azim_deg: float, elev_deg: float):
    azim = np.radians(azim_deg)
    elev = np.radians(elev_deg)
    Rz = np.array([
        [np.cos(azim), -np.sin(azim), 0],
        [np.sin(azim), np.cos(azim), 0],
        [0, 0, 1],
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(elev), -np.sin(elev)],
        [0, np.sin(elev), np.cos(elev)],
    ])
    return Rz @ Rx


def render_4x4_grid(
    method_points: dict[str, np.ndarray],
    method_colors: dict[str, np.ndarray | None],
    method_anomalies: dict[str, dict],
    view_angles: dict[str, tuple[float, float]],
    output_path: str,
    max_points: int = 30000,
    title: str = "Multi-View Point Cloud Comparison",
):
    """Render 4x4 comparison grid.

    Args:
        method_points: dict method_name -> (N, 3) world points
        method_colors: dict method_name -> (N, 3) rgb colors or None for z-coloring
        method_anomalies: dict method_name -> anomaly dict
        view_angles: dict view_name -> (azim_deg, elev_deg)
        output_path: where to save the PNG
        max_points: max points per panel
    """
    method_names = list(method_points.keys())
    view_names = list(view_angles.keys())

    n_rows = len(method_names)
    n_cols = len(view_names)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        dpi=200,
        subplot_kw={"projection": None},
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    depth_min = float("inf")
    depth_max = float("-inf")
    samples_for_range = {}
    for mname in method_names:
        pts = method_points[mname]
        if len(pts) > 0:
            z_vals = pts[:, 2]
            if len(z_vals) > 0:
                z_95 = np.percentile(z_vals, [1, 99])
                depth_min = min(depth_min, z_95[0])
                depth_max = max(depth_max, z_95[1])

        sample = _downsample(pts, max_points)
        samples_for_range[mname] = sample

    for mi, mname in enumerate(method_names):
        pts = method_points[mname]
        sample = samples_for_range[mname]
        colors = method_colors.get(mname)
        anomaly = method_anomalies.get(mname, {})

        for vi, vname in enumerate(view_names):
            ax = axes[mi, vi]
            azim, elev = view_angles[vname]

            if len(sample) > 0:
                R = _get_view_transform(azim, elev)
                proj = sample @ R.T
                x, y, z = proj[:, 0], proj[:, 1], proj[:, 2]

                if colors is not None and len(colors) == len(pts):
                    idxs = np.random.choice(len(pts), min(len(sample), len(pts)), replace=False)
                    c_sample = colors[idxs]
                    ax.scatter(x, y, c=c_sample, s=0.3, alpha=0.7, rasterized=True)
                else:
                    c = np.clip(z, depth_min, depth_max)
                    ax.scatter(x, y, c=c, cmap="terrain", s=0.3,
                               vmin=depth_min, vmax=depth_max, alpha=0.7, rasterized=True)

            _draw_anomaly_annotations(ax, sample, anomaly)

            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_facecolor("white")

            if mi == 0:
                ax.set_title(vname, fontsize=12, fontweight="bold", pad=8)
            if vi == 0:
                ax.set_ylabel(mname, fontsize=12, fontweight="bold", rotation=90,
                              labelpad=20, ha="center", va="center")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    sm = None
    if depth_max > depth_min:
        norm = plt.Normalize(vmin=depth_min, vmax=depth_max)
        sm = plt.cm.ScalarMappable(cmap="terrain", norm=norm)
        sm.set_array([])

    if sm is not None:
        cbar_ax = fig.add_axes([0.92, 0.08, 0.01, 0.84])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Depth (Z)", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.91, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Visualization] Saved to {output_path}")


def _draw_anomaly_annotations(ax, sample: np.ndarray, anomaly: dict):
    if len(sample) == 0:
        return

    floating_mask = anomaly.get("floating_mask")
    if floating_mask is not None and floating_mask.any():
        centroids = anomaly.get("floating_centroids", [])
        radii = anomaly.get("floating_radii", [])
        R = _get_view_transform(0, 0)
        for i, (centroid, radius) in enumerate(zip(centroids, radii)):
            proj = centroid @ R.T
            circle = Circle(
                (proj[0], proj[1]),
                max(radius, 0.05),
                fill=False,
                edgecolor="red",
                linewidth=1.0,
                linestyle="--",
                alpha=0.6,
            )
            ax.add_patch(circle)
            ax.annotate(
                "floating",
                (proj[0], proj[1]),
                fontsize=6,
                color="red",
                alpha=0.7,
                ha="center",
                va="bottom",
                xytext=(0, -8),
                textcoords="offset points",
            )
            if i >= 4:
                break

    break_mask = anomaly.get("break_mask")
    if break_mask is not None and break_mask.any() and len(sample) == len(break_mask):
        break_pts = sample[break_mask]
        if len(break_pts) > 0:
            R = _get_view_transform(0, 0)
            proj = break_pts @ R.T
            x, y = proj[:, 0], proj[:, 1]
            ax.scatter(x, y, c="red", s=1.5, alpha=0.5, marker="x", rasterized=True)
            if len(x) > 0:
                centroid = break_pts.mean(axis=0) @ R.T
                ax.annotate(
                    "surface break",
                    (centroid[0], centroid[1]),
                    fontsize=6,
                    color="red",
                    alpha=0.7,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="red", alpha=0.5),
                )


def _downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idxs = np.random.choice(len(points), max_points, replace=False)
    return points[idxs]
