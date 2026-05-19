from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ..types import BatchedExample


@dataclass
class CameraNoiseCfg:
    enabled: bool
    apply_to: str
    mode: str
    translation_sigma_ratio: float
    rotation_sigma_deg: float
    seed: int


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    if device.type == "cpu":
        generator = torch.Generator()
    else:
        generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _stable_scene_seed(base_seed: int, scene_names: list[str]) -> int:
    seed = base_seed & 0x7FFFFFFF
    for scene_name in scene_names:
        for character in scene_name:
            seed = ((seed * 33) + ord(character)) & 0x7FFFFFFF
    return seed


def _rotation_matrix_from_axis_angle(
    axis: Tensor,
    angle: Tensor,
) -> Tensor:
    # Rodrigues formula for batched axis-angle rotations.
    kx, ky, kz = axis.unbind(dim=-1)
    zeros = torch.zeros_like(kx)
    skew = torch.stack(
        (
            torch.stack((zeros, -kz, ky), dim=-1),
            torch.stack((kz, zeros, -kx), dim=-1),
            torch.stack((-ky, kx, zeros), dim=-1),
        ),
        dim=-2,
    )
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand(axis.shape[0], -1, -1)
    sin = torch.sin(angle)[:, None, None]
    cos = torch.cos(angle)[:, None, None]
    outer = axis[:, :, None] * axis[:, None, :]
    return cos * eye + (1.0 - cos) * outer + sin * skew


def _get_baseline_scale(extrinsics: Tensor) -> Tensor:
    centers = extrinsics[..., :3, 3]
    if centers.shape[1] < 2:
        return torch.ones(centers.shape[0], dtype=centers.dtype, device=centers.device)
    pairwise = torch.cdist(centers, centers, p=2)
    mask = torch.triu(
        torch.ones(pairwise.shape[-2:], dtype=torch.bool, device=pairwise.device),
        diagonal=1,
    )
    pairwise = pairwise[:, mask]
    if pairwise.shape[-1] == 0:
        return torch.ones(centers.shape[0], dtype=centers.dtype, device=centers.device)
    return pairwise.mean(dim=-1).clamp_min(1e-6)


def apply_camera_noise_shim(
    batch: BatchedExample,
    cfg: CameraNoiseCfg,
) -> BatchedExample:
    if not cfg.enabled:
        return batch
    if cfg.apply_to != "context":
        raise ValueError(f"Unsupported camera noise target: {cfg.apply_to}")
    if "context" not in batch or "extrinsics" not in batch["context"]:
        return batch

    extrinsics = batch["context"]["extrinsics"].clone()
    batch_size, num_views, _, _ = extrinsics.shape
    device = extrinsics.device
    dtype = extrinsics.dtype

    scene_names = batch.get("scene", [])
    seed = _stable_scene_seed(cfg.seed, scene_names)
    generator = _make_generator(device, seed)

    if cfg.mode == "translation":
        sigma = _get_baseline_scale(extrinsics) * cfg.translation_sigma_ratio
        noise = torch.randn(
            (batch_size, num_views, 3),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] + noise * sigma[:, None, None]
    elif cfg.mode == "rotation":
        sigma_rad = math.radians(cfg.rotation_sigma_deg)
        if sigma_rad > 0:
            axes = torch.randn(
                (batch_size * num_views, 3),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            angles = torch.randn(
                (batch_size * num_views,),
                generator=generator,
                device=device,
                dtype=dtype,
            ) * sigma_rad
            rotation_noise = _rotation_matrix_from_axis_angle(axes, angles)
            rotation_noise = rotation_noise.view(batch_size, num_views, 3, 3)
            extrinsics[..., :3, :3] = extrinsics[..., :3, :3] @ rotation_noise
    else:
        raise ValueError(f"Unsupported camera noise mode: {cfg.mode}")

    batch["context"]["extrinsics"] = extrinsics
    return batch
