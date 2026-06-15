from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image
from einops import rearrange


def load_scene_images(chunk_file_path: str, scene_key: str) -> list[np.ndarray]:
    """Load all images for a scene from .torch file as numpy arrays."""
    data = torch.load(chunk_file_path)
    for item in data:
        if item["key"] == scene_key:
            images = []
            for img_tensor in item["images"]:
                img_bytes = bytes(img_tensor.tolist())
                img = Image.open(io.BytesIO(img_bytes))
                images.append(np.array(img, dtype=np.float32) / 255.0)
            return images
    raise ValueError(f"Scene {scene_key} not found in {chunk_file_path}")


def load_gt_cameras(chunk_file_path: str, scene_key: str) -> torch.Tensor:
    """Load 18D camera tensor (N, 18) for a scene."""
    data = torch.load(chunk_file_path)
    for item in data:
        if item["key"] == scene_key:
            return item["cameras"].clone()
    raise ValueError(f"Scene {scene_key} not found in {chunk_file_path}")


def convert_poses(poses_18d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert RE10K 18D camera format to C2W extrinsics and intrinsics.

    Args:
        poses_18d: (N, 18) tensor with [fx, fy, cx, cy, 0, 0, w2c_3x4]

    Returns:
        c2w: (N, 4, 4) camera-to-world extrinsics
        intrinsics: (N, 3, 3) normalized intrinsics matrix
    """
    fx, fy, cx, cy = poses_18d[:, :4].T
    N = poses_18d.shape[0]

    w2c_3x4 = rearrange(poses_18d[:, 6:], "b (h w) -> b h w", h=3, w=4)
    w2c_4x4 = torch.eye(4, device=poses_18d.device).unsqueeze(0).repeat(N, 1, 1)
    w2c_4x4[:, :3, :4] = w2c_3x4
    c2w = torch.linalg.inv(w2c_4x4)

    intrinsics = torch.eye(3, device=poses_18d.device).unsqueeze(0).repeat(N, 1, 1)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    return c2w, intrinsics


def scale_intrinsics(intrinsics_norm: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Scale normalized intrinsics to target resolution.

    Args:
        intrinsics_norm: (N, 3, 3) with normalized fx, fy, cx, cy (in [0,1])
        target_h, target_w: target image height and width

    Returns:
        (N, 3, 3) intrinsics matrix in pixel units
    """
    K = intrinsics_norm.clone()
    K[:, 0, 0] *= target_w
    K[:, 0, 2] *= target_w
    K[:, 1, 1] *= target_h
    K[:, 1, 2] *= target_h
    return K


def rescale_depth(depth: np.ndarray, from_h: int, from_w: int, to_h: int, to_w: int) -> np.ndarray:
    """Rescale depth map by a scale factor when image resolution changes.

    Depth scales linearly with image resolution when focal length is scaled.
    depth_new = depth_old * (new_focal / old_focal)
    For a simple resize-to-fill, we approximate:
      scale = (to_h / from_h)  # roughly
    """
    scale = to_h / from_h
    return depth * scale


def back_project_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    stride: int = 1,
    depth_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Back-project a depth map to 3D world coordinates.

    Args:
        depth: (H, W) depth map in world units
        intrinsics: (3, 3) camera intrinsics matrix in pixel units
        c2w: (4, 4) camera-to-world transformation matrix
        stride: pixel sampling stride (1 = full resolution)
        depth_threshold: if set, clamp depth to this max value

    Returns:
        points_world: (N, 3) 3D world coordinates
        pixel_coords: (N, 2) corresponding pixel xy coordinates, or None
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    xx, yy = np.meshgrid(
        np.arange(W // stride, dtype=np.float32) * stride + stride / 2,
        np.arange(H // stride, dtype=np.float32) * stride + stride / 2,
        indexing="xy",
    )
    xx = xx.flatten()
    yy = yy.flatten()
    d = depth[::stride, ::stride].flatten()

    valid = (d > 1e-3) & np.isfinite(d)
    if depth_threshold is not None:
        valid &= (d < depth_threshold)

    xx, yy, d = xx[valid], yy[valid], d[valid]

    x_cam = (xx - cx) / fx * d
    y_cam = (yy - cy) / fy * d
    z_cam = d

    ones = np.ones_like(x_cam)
    cam_points = np.stack([x_cam, y_cam, z_cam, ones], axis=0)

    world_points_h = c2w @ cam_points
    world_points = world_points_h[:3, :].T

    pixel_xy = np.stack([xx, yy], axis=1)

    return world_points.astype(np.float32), pixel_xy.astype(np.float32)


def back_project_multi_view(
    depths: np.ndarray,
    intrinsics_norm: np.ndarray,
    c2w: np.ndarray,
    target_resolutions: list[tuple[int, int]],
    stride: int = 2,
    depth_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Back-project multiple depth maps to 3D world coordinates.

    Args:
        depths: list of (H, W) depth maps, or stacked (V, H, W)
        intrinsics_norm: (V, 3, 3) normalized intrinsics
        c2w: (V, 4, 4) camera-to-world transforms
        target_resolutions: list of (H, W) for each view
        stride: pixel sampling stride
        depth_threshold: max depth threshold

    Returns:
        points_world: (N, 3) concatenated world points
        colors: (N, 3) or None
    """
    all_points = []
    for i, depth in enumerate(depths):
        H, W = target_resolutions[i]
        K = scale_intrinsics(
            torch.from_numpy(intrinsics_norm[i : i + 1]),
            H,
            W,
        )[0].numpy()
        pts, _ = back_project_depth(
            depth, K, c2w[i].numpy() if not isinstance(c2w[i], np.ndarray) else c2w[i],
            stride=stride,
            depth_threshold=depth_threshold,
        )
        all_points.append(pts)

    return np.concatenate(all_points, axis=0).astype(np.float32)
