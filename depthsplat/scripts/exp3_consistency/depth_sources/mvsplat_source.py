from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from depth_sources.base import DepthSource
import config
from utils import run_subprocess


def _find_mvsplat_depth_npy(method_dir: Path, indices: list[int]) -> list[Path]:
    """MVSplat saves depth to images/<scene>/depth/*.npy and images/<scene>/color/<idx>.png."""
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

    raw_npy = sorted(depth_dir.glob("*.npy"))
    if not raw_npy:
        raise FileNotFoundError(f"No .npy files in {depth_dir}")

    results = []
    for i, f in enumerate(raw_npy):
        if i < len(indices):
            dst = method_dir / f"depth_{indices[i]:06d}.npy"
            shutil.copy(f, dst)
            results.append(dst)

    for i, idx in enumerate(indices):
        if color_dir and i < len(list(color_dir.glob("*"))):
            img_files = sorted(color_dir.glob("*.png"))
            input_files = [f for f in img_files if f"input_{idx:06d}" in f.name or f"{idx:06d}" in f.name]
            if i < len(img_files):
                src = input_files[0] if input_files else img_files[i]
                dst = method_dir / f"image_{idx:06d}.png"
                shutil.copy(src, dst)

    return results


class MVSplatSource(DepthSource):
    def __init__(self):
        self._name = "mvsplat"

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return "MVSplat"

    def predict(
        self,
        image_paths: list[Path],
        cameras: dict[str, np.ndarray],
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            config.PYTHON_BIN, "-m", "src.main",
            "+experiment=re10k",
            "dataset.test_chunk_interval=1",
            f"dataset.roots=[{config.DATASET_ROOT}]",
            f"dataset.overfit_to_scene={cameras['scene']}",
            "dataset.test_len=1",
            "checkpointing.load=checkpoints/re10k.ckpt",
            "mode=test",
            "dataset/view_sampler=evaluation",
            "test.compute_scores=false",
            "test.save_depth=true",
            "test.save_image=false",
            f"output_dir={output_dir}",
        ]

        ret, stdout, stderr = run_subprocess(
            cmd, cwd=config.MVSPLAT_ROOT,
            env={
                "CUDA_VISIBLE_DEVICES": "0",
                "WANDB_MODE": "disabled",
            },
        )
        if ret != 0:
            raise RuntimeError(
                f"MVSplat failed (exit {ret})\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}"
            )

        indices = cameras["indices"].tolist()
        _find_mvsplat_depth_npy(output_dir, indices)

        np.savez(
            output_dir / "cameras.npz",
            extrinsics=cameras["extrinsics"],
            intrinsics=cameras["intrinsics"],
            near=np.array(cameras["near"]),
            far=np.array(cameras["far"]),
            scene=cameras["scene"],
        )
