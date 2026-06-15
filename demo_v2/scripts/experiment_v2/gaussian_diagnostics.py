from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter
from skimage.transform import resize

from .config import (
    OPACITY_FILTER_THRESHOLD_GS,
    OPACITY_FILTER_THRESHOLD_PC,
    SPLATTING_RADIUS_PX,
    SCALE_MAP_LOG_MIN,
    SCALE_MAP_LOG_MAX,
    MAX_POINTS_PER_PANEL,
)


@dataclass
class GaussianData:
    centers: np.ndarray
    opacities: np.ndarray
    scales: np.ndarray | None
    max_eigenvalues: np.ndarray | None
    sh_dc: np.ndarray | None
    has_real_gaussians: bool
    depth_map: np.ndarray | None
    conf_map: np.ndarray | None
    method_name: str
    display_centers: np.ndarray | None = None
    display_sh_dc: np.ndarray | None = None


def _downsample(data: GaussianData, max_points: int) -> GaussianData:
    N = len(data.centers)
    if N <= max_points:
        return data
    idxs = np.random.choice(N, max_points, replace=False)
    return GaussianData(
        centers=data.centers[idxs],
        opacities=data.opacities[idxs],
        scales=data.scales[idxs] if data.scales is not None else None,
        max_eigenvalues=data.max_eigenvalues[idxs] if data.max_eigenvalues is not None else None,
        sh_dc=data.sh_dc[idxs] if data.sh_dc is not None else None,
        has_real_gaussians=data.has_real_gaussians,
        depth_map=data.depth_map,
        conf_map=data.conf_map,
        method_name=data.method_name,
        display_centers=data.display_centers,
        display_sh_dc=data.display_sh_dc,
    )


def _project_points(
    centers: np.ndarray,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    H: int,
    W: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    R = w2c[:3, :3]
    t = w2c[:3, 3]
    cam_pts = (centers @ R.T) + t

    x_cam, y_cam, z_cam = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]
    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy

    valid = (z_cam > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z_cam, valid


def _splat_to_image(
    u: np.ndarray,
    v: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    H: int,
    W: int,
    radius: int = SPLATTING_RADIUS_PX,
    mode: str = "max",
) -> np.ndarray:
    grid = np.full((H, W), np.nan, dtype=np.float32)
    count = np.zeros((H, W), dtype=np.int32)

    ui = np.clip(np.round(u[valid]).astype(np.int32), 0, W - 1)
    vi = np.clip(np.round(v[valid]).astype(np.int32), 0, H - 1)
    vals = values[valid]

    if mode == "max":
        for i in range(len(ui)):
            idx = (vi[i], ui[i])
            if np.isnan(grid[idx]) or vals[i] > grid[idx]:
                grid[idx] = vals[i]
    elif mode == "mean":
        accum = np.zeros((H, W), dtype=np.float64)
        for i in range(len(ui)):
            idx = (vi[i], ui[i])
            accum[idx] += vals[i]
            count[idx] += 1
        mask = count > 0
        grid[mask] = accum[mask] / count[mask]

    if radius > 0:
        nan_mask = np.isnan(grid)
        grid_filled = grid.copy()
        grid_filled[nan_mask] = -np.inf if mode == "max" else 0.0
        if mode == "max":
            grid_filled = maximum_filter(grid_filled, size=radius * 2 + 1)
        grid = np.where(nan_mask, grid_filled, grid)

    return grid


def _render_depth_from_gaussians(
    centers: np.ndarray,
    opacities: np.ndarray,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    H: int,
    W: int,
    radius: int = SPLATTING_RADIUS_PX,
) -> np.ndarray:
    u, v, z_cam, valid = _project_points(centers, intrinsics, w2c, H, W)

    valid_opaque = valid & (opacities > OPACITY_FILTER_THRESHOLD_GS)
    if valid_opaque.sum() == 0:
        return np.full((H, W), np.nan, dtype=np.float32)

    ui = np.clip(np.round(u[valid_opaque]).astype(np.int32), 0, W - 1)
    vi = np.clip(np.round(v[valid_opaque]).astype(np.int32), 0, H - 1)
    z_vals = z_cam[valid_opaque]

    depth_grid = np.full((H, W), np.inf, dtype=np.float32)

    for i in range(len(ui)):
        if z_vals[i] < depth_grid[vi[i], ui[i]]:
            depth_grid[vi[i], ui[i]] = z_vals[i]

    depth_grid[depth_grid >= np.inf] = np.nan

    if radius > 0:
        nan_mask = np.isnan(depth_grid)
        depth_filled = depth_grid.copy()
        depth_filled[nan_mask] = np.inf
        depth_filled = minimum_filter(depth_filled, size=radius * 2 + 1)
        depth_grid = np.where(nan_mask, depth_filled, depth_grid)

    return depth_grid


def _render_centers_panel(
    ax: plt.Axes,
    data: GaussianData,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    H: int,
    W: int,
    depth_min: float,
    depth_max: float,
    depth_color_only: bool = False,
):
    ctrs = data.display_centers if data.display_centers is not None else data.centers
    colors_src = None if depth_color_only else (data.display_sh_dc if data.display_sh_dc is not None else data.sh_dc)

    u, v, z_cam, valid = _project_points(ctrs, intrinsics, w2c, H, W)

    if valid.sum() == 0:
        ax.text(0.5, 0.5, "No points", transform=ax.transAxes, ha="center", va="center", fontsize=8)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        return

    u_v = u[valid]
    v_v = v[valid]
    z_v = np.clip(z_cam[valid], depth_min, depth_max)

    if colors_src is not None and len(colors_src) == len(ctrs):
        colors = np.clip(colors_src[valid], 0, 1)
        ax.scatter(u_v, v_v, c=colors, s=0.5, alpha=0.6, rasterized=True)
    else:
        ax.scatter(u_v, v_v, c=z_v, cmap="terrain", s=0.5, alpha=0.6,
                   vmin=depth_min, vmax=depth_max, rasterized=True)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _render_opacity_panel(
    ax: plt.Axes,
    data: GaussianData,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    H: int,
    W: int,
):
    if data.conf_map is not None and not data.has_real_gaussians:
        opacity_map = data.conf_map
        if opacity_map.shape[:2] != (H, W):
            opacity_map = resize(opacity_map.astype(np.float32), (H, W), order=1)
        im = ax.imshow(opacity_map, cmap="hot", vmin=0, vmax=1, origin="upper")
        return im

    u, v, _, valid = _project_points(data.centers, intrinsics, w2c, H, W)

    if valid.sum() == 0:
        ax.text(0.5, 0.5, "No points", transform=ax.transAxes, ha="center", va="center", fontsize=8)
        return None

    opacity_grid = _splat_to_image(u, v, data.opacities, valid, H, W, radius=SPLATTING_RADIUS_PX, mode="max")
    im = ax.imshow(opacity_grid, cmap="hot", vmin=0, vmax=1, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _render_scale_panel(
    ax: plt.Axes,
    data: GaussianData,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    H: int,
    W: int,
):
    if not data.has_real_gaussians or data.max_eigenvalues is None:
        ax.text(0.5, 0.5, "N/A\n(no Gaussian scales)", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, fontstyle="italic", color="gray")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    u, v, _, valid = _project_points(data.centers, intrinsics, w2c, H, W)

    if valid.sum() == 0:
        ax.text(0.5, 0.5, "No points", transform=ax.transAxes, ha="center", va="center", fontsize=8)
        return

    log_evals = np.log10(np.maximum(data.max_eigenvalues, 1e-10))
    log_evals_clipped = np.clip(log_evals, SCALE_MAP_LOG_MIN, SCALE_MAP_LOG_MAX)
    scale_grid = _splat_to_image(u, v, log_evals_clipped, valid, H, W, radius=SPLATTING_RADIUS_PX, mode="max")
    im = ax.imshow(scale_grid, cmap="plasma", vmin=SCALE_MAP_LOG_MIN, vmax=SCALE_MAP_LOG_MAX, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _render_depth_panel(
    ax: plt.Axes,
    data: GaussianData,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    H: int,
    W: int,
    depth_min: float,
    depth_max: float,
):
    if data.has_real_gaussians:
        rendered_depth = _render_depth_from_gaussians(
            data.centers, data.opacities, intrinsics, w2c, H, W
        )
        im = ax.imshow(rendered_depth, cmap="inferno", vmin=depth_min, vmax=depth_max, origin="upper")
        return im

    if data.depth_map is not None:
        d = data.depth_map
        if d.shape[:2] != (H, W):
            d = resize(d.astype(np.float32), (H, W), order=1)
        im = ax.imshow(d, cmap="inferno", vmin=depth_min, vmax=depth_max, origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    ax.text(0.5, 0.5, "No depth", transform=ax.transAxes, ha="center", va="center", fontsize=8)
    return None


def _render_filtered_panel(
    ax: plt.Axes,
    data: GaussianData,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    H: int,
    W: int,
    depth_min: float,
    depth_max: float,
):
    threshold = OPACITY_FILTER_THRESHOLD_GS if data.has_real_gaussians else OPACITY_FILTER_THRESHOLD_PC
    mask = data.opacities >= threshold

    if mask.sum() == 0:
        ax.text(0.5, 0.5, f"No points (thresh={threshold})", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        return

    centers_f = data.centers[mask]
    sh_dc_f = data.sh_dc[mask] if data.sh_dc is not None else None

    if len(centers_f) > MAX_POINTS_PER_PANEL:
        idxs = np.random.choice(len(centers_f), MAX_POINTS_PER_PANEL, replace=False)
        centers_f = centers_f[idxs]
        if sh_dc_f is not None:
            sh_dc_f = sh_dc_f[idxs]

    u, v, z_cam, valid = _project_points(centers_f, intrinsics, w2c, H, W)

    if valid.sum() == 0:
        ax.text(0.5, 0.5, "No visible points", transform=ax.transAxes, ha="center", va="center", fontsize=8)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        return

    z_v = np.clip(z_cam[valid], depth_min, depth_max)

    if sh_dc_f is not None:
        colors = np.clip(sh_dc_f[valid], 0, 1)
        ax.scatter(u[valid], v[valid], c=colors, s=0.5, alpha=0.7, rasterized=True)
    else:
        ax.scatter(u[valid], v[valid], c=z_v, cmap="terrain", s=0.5, alpha=0.7,
                   vmin=depth_min, vmax=depth_max, rasterized=True)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def render_diagnostic_grid(
    all_data: dict[str, GaussianData],
    intrinsics_per_method: dict[str, np.ndarray],
    w2c_per_method: dict[str, np.ndarray],
    H_per_method: dict[str, int],
    W_per_method: dict[str, int],
    view_idx: int,
    output_path: str,
    depth_color_only: bool = False,
):
    method_order = ["Ours (PMR)", "MVSplat", "VGGT", "DA3"]
    panel_names = [
        "Gaussian Centers",
        "Opacity Map",
        "Scale Map\n(log10 max eigenvalue)",
        "Rendered Depth",
        "Filtered Point Cloud",
    ]
    n_rows = len(method_order)
    n_cols = len(panel_names)

    depth_min = float("inf")
    depth_max = float("-inf")
    for mname in method_order:
        if mname not in all_data:
            continue
        pts = all_data[mname].centers
        if len(pts) > 0:
            intrinsics = intrinsics_per_method[mname]
            w2c = w2c_per_method[mname]
            _, _, z_cam, valid = _project_points(pts, intrinsics, w2c,
                                                  H_per_method[mname], W_per_method[mname])
            if valid.sum() > 0:
                z_vals = z_cam[valid]
                z_1, z_99 = np.percentile(z_vals, [1, 99])
                depth_min = min(depth_min, z_1)
                depth_max = max(depth_max, z_99)

    if depth_max <= depth_min:
        depth_min, depth_max = 0.1, 10.0

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows), dpi=150)

    for mi, mname in enumerate(method_order):
        if mname not in all_data:
            for ci in range(n_cols):
                axes[mi, ci].text(0.5, 0.5, "N/A", transform=axes[mi, ci].transAxes,
                                  ha="center", va="center", fontsize=14, fontweight="bold", color="gray")
                axes[mi, ci].set_xticks([])
                axes[mi, ci].set_yticks([])
            continue

        data = all_data[mname]
        data = _downsample(data, MAX_POINTS_PER_PANEL)

        H = H_per_method[mname]
        W = W_per_method[mname]
        K = intrinsics_per_method[mname]
        w2c = w2c_per_method[mname]

        if mi == 0:
            for ci, name in enumerate(panel_names):
                axes[mi, ci].set_title(name, fontsize=13, fontweight="bold", pad=8)

        axes[mi, 0].set_ylabel(mname, fontsize=13, fontweight="bold", rotation=90,
                               labelpad=20, ha="center", va="center")

        _render_centers_panel(axes[mi, 0], data, K, w2c, H, W, depth_min, depth_max, depth_color_only)
        _render_opacity_panel(axes[mi, 1], data, K, w2c, H, W)
        _render_scale_panel(axes[mi, 2], data, K, w2c, H, W)
        _render_depth_panel(axes[mi, 3], data, K, w2c, H, W, depth_min, depth_max)
        _render_filtered_panel(axes[mi, 4], data, K, w2c, H, W, depth_min, depth_max)

    norm = plt.Normalize(vmin=depth_min, vmax=depth_max)
    sm = plt.cm.ScalarMappable(cmap="terrain", norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.08, 0.01, 0.84])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Depth (Z)", fontsize=10)

    fig.suptitle("Gaussian / Point Cloud Diagnostics", fontsize=15, fontweight="bold", y=0.996)
    fig.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.04, wspace=0.15, hspace=0.20)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Diagnostics] Saved to {output_path}")
