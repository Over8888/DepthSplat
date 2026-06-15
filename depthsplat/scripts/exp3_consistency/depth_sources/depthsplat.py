from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from depth_sources.base import DepthSource
import config
from utils import run_subprocess


def _find_depth_files(method_dir: Path, indices: list[int]) -> list[Path]:
    depth_dir = None
    color_dir = None
    for candidate in sorted(method_dir.rglob("depth")):
        if candidate.is_dir():
            depth_dir = candidate
            break
    for candidate in sorted(method_dir.rglob("color")):
        if candidate.is_dir():
            color_dir = candidate
            break
    if depth_dir is None:
        raise FileNotFoundError(f"No depth directory found under {method_dir}")

    results = []
    for idx in indices:
        src = depth_dir / f"{idx:06d}.npy"
        if not src.exists():
            npy_files = sorted(depth_dir.glob("*.npy"))
            raise FileNotFoundError(
                f"Depth file for index {idx:06d} not found in {depth_dir}. "
                f"Found: {[f.name for f in npy_files]}"
            )
        dst = method_dir / f"depth_{idx:06d}.npy"
        shutil.copy(src, dst)
        results.append(dst)

    for idx in indices:
        for pattern in [f"input_{idx:06d}.png", f"{idx:06d}.png"]:
            src = color_dir / pattern
            if src.exists():
                dst = method_dir / f"image_{idx:06d}.png"
                shutil.copy(src, dst)
                break
    return results


class DepthSplatSource(DepthSource):
    def __init__(self):
        self._name = "ours"

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return "Ours (PMR)"

    def predict(
        self,
        image_paths: list[Path],
        cameras: dict[str, np.ndarray],
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = "pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth"

        cmd = [
            config.PYTHON_BIN, "-m", "src.main",
            "+experiment=re10k",
            "dataset.test_chunk_interval=1",
            "dataset.roots=[/root/autodl-tmp/RealEstate10K]",
            f"dataset.overfit_to_scene={cameras['scene']}",
            "dataset.test_len=1",
            "dataset.image_shape=[256,256]",
            "model.encoder.num_scales=2",
            "model.encoder.monodepth_vit_type=vitb",
            "model.encoder.upsample_factor=2",
            "model.encoder.lowest_feature_resolution=4",
            "model.encoder.cost_volume_confidence=true",
            "model.encoder.pmr_guided_smooth=true",
            f"checkpointing.pretrained_model={checkpoint}",
            "mode=test",
            "dataset/view_sampler=evaluation",
            "dataset.view_sampler.num_context_views=2",
            "test.compute_scores=false",
            "test.save_depth=true",
            "test.save_depth_npy=true",
            "test.save_input_images=true",
            "train.forward_depth_only=true",
            "~loss.lpips",
            f"output_dir={output_dir}",
        ]

        ret, stdout, stderr = run_subprocess(
            cmd, cwd=config.DEPTHSPLAT_ROOT,
            env={
                "CUDA_VISIBLE_DEVICES": "0",
                "DINOV2_LOCAL_REPO": config.DINOV2_LOCAL_REPO,
                "DINOV2_CHECKPOINT_DIR": config.DINOV2_CHECKPOINT_DIR,
                "WANDB_MODE": "disabled",
            },
        )
        if ret != 0:
            raise RuntimeError(
                f"DepthSplat failed (exit {ret})\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}"
            )

        indices = cameras["indices"].tolist()
        _find_depth_files(output_dir, indices)

        np.savez(
            output_dir / "cameras.npz",
            extrinsics=cameras["extrinsics"],
            intrinsics=cameras["intrinsics"],
            near=np.array(cameras["near"]),
            far=np.array(cameras["far"]),
            scene=cameras["scene"],
        )
