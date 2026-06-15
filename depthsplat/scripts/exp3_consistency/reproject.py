from __future__ import annotations

import numpy as np


def get_world_rays(
    coords: np.ndarray,
    extrinsics_c2w: np.ndarray,
    intrinsics_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world-space ray origins and directions for pixel coordinates.

    Args:
        coords: [H*W, 2] normalized pixel coordinates in [0, 1]^2
        extrinsics_c2w: [4, 4] camera-to-world matrix
        intrinsics_norm: [3, 3] normalized intrinsics

    Returns:
        origins: [H*W, 3] camera centers in world space
        directions: [H*W, 3] unit ray directions in world space
    """
    K_inv = np.linalg.inv(intrinsics_norm)
    uv_h = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
    dirs_cam = uv_h @ K_inv.T
    dirs_cam = dirs_cam / dirs_cam[:, 2:3]

    R = extrinsics_c2w[:3, :3]
    t = extrinsics_c2w[:3, 3]

    dirs_world = dirs_cam @ R.T
    dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=1, keepdims=True)

    origins = np.tile(t, (coords.shape[0], 1))
    return origins, dirs_world


def unproject_depth_to_3d(
    coords: np.ndarray,
    depth: np.ndarray,
    extrinsics_c2w: np.ndarray,
    intrinsics_norm: np.ndarray,
) -> np.ndarray:
    """Unproject 2D+depth to 3D world points.

    Args:
        coords: [N, 2] normalized [x, y] in [0,1]
        depth: [N] metric depth
        extrinsics_c2w: [4, 4] camera-to-world
        intrinsics_norm: [3, 3] normalized intrinsics

    Returns:
        points_3d: [N, 3] world-space points
    """
    origins, directions = get_world_rays(coords, extrinsics_c2w, intrinsics_norm)
    return origins + directions * depth[:, None]


def project_3d_to_2d(
    points_3d: np.ndarray,
    extrinsics_c2w: np.ndarray,
    intrinsics_norm: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D world points to 2D pixel coordinates.

    Args:
        points_3d: [N, 3] world-space points
        extrinsics_c2w: [4, 4] camera-to-world
        intrinsics_norm: [3, 3] normalized intrinsics
        eps: small value to avoid division by zero

    Returns:
        uv: [N, 2] normalized pixel coordinates in [0,1]
        in_front: [N] bool mask for points in front of camera
    """
    R = extrinsics_c2w[:3, :3]
    t = extrinsics_c2w[:3, 3]
    w2c_R = R.T
    w2c_t = -w2c_R @ t

    points_cam = points_3d @ w2c_R.T + w2c_t
    z_cam = points_cam[:, 2]
    in_front = z_cam > eps

    z_safe = np.where(in_front, z_cam, eps)
    points_2d = points_cam[:, :2] / z_safe[:, None]
    uv = points_2d @ intrinsics_norm[:2, :2].T + intrinsics_norm[:2, 2]

    return uv, in_front


def reproject_depth_pair(
    depth_src: np.ndarray,
    extrinsics_src: np.ndarray,
    intrinsics_src: np.ndarray,
    depth_tgt: np.ndarray,
    extrinsics_tgt: np.ndarray,
    intrinsics_tgt: np.ndarray,
) -> dict:
    """Reproject depth from source view to target view.

    Args:
        depth_src: [H, W] source depth
        extrinsics_src: [4, 4] source c2w
        intrinsics_src: [3, 3] source normalized intrinsics
        depth_tgt: [H, W] target depth
        extrinsics_tgt: [4, 4] target c2w
        intrinsics_tgt: [3, 3] target normalized intrinsics

    Returns:
        dict with:
          depth_reproj: [H, W] reprojected depth at source resolution
          depth_tgt: [H, W] target depth sampled at source pixel locations
          error_map: [H, W] relative error |z_reproj - z_tgt| / z_tgt
          valid: [H, W] bool mask
          uv_tgt: [H, W, 2] where each src pixel projects to in target
    """
    H, W = depth_src.shape
    ys, xs = np.mgrid[0:H, 0:W]
    coords = np.stack([
        (xs.ravel() + 0.5) / W,
        (ys.ravel() + 0.5) / H,
    ], axis=1).astype(np.float32)

    points_3d = unproject_depth_to_3d(
        coords, depth_src.ravel(), extrinsics_src, intrinsics_src
    )

    uv_tgt, in_front = project_3d_to_2d(
        points_3d, extrinsics_tgt, intrinsics_tgt
    )

    R_src = extrinsics_src[:3, :3]
    t_src = extrinsics_src[:3, 3]
    R_tgt = extrinsics_tgt[:3, :3]
    t_tgt = extrinsics_tgt[:3, 3]

    w2c_tgt_R = R_tgt.T
    w2c_tgt_t = -w2c_tgt_R @ t_tgt
    points_tgt_cam = points_3d @ w2c_tgt_R.T + w2c_tgt_t
    depth_reproj = points_tgt_cam[:, 2]

    eps = 1e-6
    in_bounds = (
        (uv_tgt[:, 0] > eps) & (uv_tgt[:, 0] < 1.0 - eps) &
        (uv_tgt[:, 1] > eps) & (uv_tgt[:, 1] < 1.0 - eps)
    )
    valid = in_front & in_bounds & (depth_reproj > eps) & (depth_src.ravel() > eps)

    depth_reproj_map = np.full(H * W, np.nan, dtype=np.float32)
    depth_tgt_map = np.full(H * W, np.nan, dtype=np.float32)
    error_map = np.full(H * W, np.nan, dtype=np.float32)

    depth_reproj_map[valid] = depth_reproj[valid]

    uv_valid = uv_tgt[valid]
    uv_tgt_px = np.stack([
        (uv_valid[:, 0]) * W - 0.5,
        (uv_valid[:, 1]) * H - 0.5,
    ], axis=1)

    from scipy.ndimage import map_coordinates
    sampled = map_coordinates(
        depth_tgt,
        [uv_tgt_px[:, 1], uv_tgt_px[:, 0]],
        order=1, mode='constant', cval=np.nan,
    )
    valid_indices = np.where(valid)[0]
    valid_sample = ~np.isnan(sampled)
    valid[valid_indices] = valid[valid_indices] & valid_sample
    depth_tgt_map[valid] = sampled[valid_sample]

    valid_mask = valid & (depth_tgt_map > eps)
    error_map[valid_mask] = np.abs(
        depth_reproj_map[valid_mask] - depth_tgt_map[valid_mask]
    ) / np.maximum(depth_tgt_map[valid_mask], eps)

    depth_tgt_flat = depth_tgt_map.copy()
    depth_tgt_flat[~valid_mask] = np.nan

    return {
        "depth_reproj": depth_reproj_map.reshape(H, W),
        "depth_tgt_sampled": depth_tgt_flat.reshape(H, W),
        "error_map": error_map.reshape(H, W),
        "valid": valid_mask.reshape(H, W),
        "uv_tgt": uv_tgt.reshape(H, W, 2),
    }
