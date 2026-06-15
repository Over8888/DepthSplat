from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..config import (
    DATASET_ROOT,
    DEPTHSPLAT_ROOT,
    DEPTHSPLAT_PYTHON,
    OUTPUT_DIR,
)


def run_depthsplat_pmr(
    scene_key: str,
    context_indices: list[int],
    image_size: tuple[int, int],
    output_dir: Path,
) -> Path:
    """Run DepthSplat with PMR enabled, return path to depth npy directory."""
    eval_index = {
        scene_key: {
            "context": list(context_indices),
            "target": list(range(context_indices[0], context_indices[1] + 1)),
        }
    }
    eval_index_path = output_dir / f"eval_index_{scene_key}.json"
    eval_index_path.write_text(json.dumps(eval_index))

    extra_overrides = [
        "dataset.test_chunk_interval=1",
        "model.encoder.num_scales=2",
        "model.encoder.upsample_factor=2",
        "model.encoder.lowest_feature_resolution=4",
        "model.encoder.monodepth_vit_type=vitb",
        "data_loader.test.num_workers=0",
        "wandb.mode=disabled",
        "trainer.num_sanity_val_steps=0",
        "dataset.skip_bad_shape=false",
        "model.encoder.pmr_guided_smooth=true",
        "model.encoder.cost_volume_confidence=true",
        "test.save_depth=true",
        "test.save_depth_npy=true",
        "test.save_depth_concat_img=true",
        "test.save_gaussian=true",
        "test.save_image=false",
        "test.save_gt_image=false",
        "test.save_input_images=false",
        "test.compute_scores=false",
        "test.save_video=false",
    ]

    cmd = [
        DEPTHSPLAT_PYTHON,
        "-m",
        "src.main",
        "+experiment=re10k",
        f"dataset.roots=[{DATASET_ROOT}]",
        f"dataset.view_sampler.index_path={eval_index_path}",
        "dataset/view_sampler=evaluation",
        "dataset.view_sampler.num_context_views=2",
        f"dataset.image_shape=[{image_size[0]},{image_size[1]}]",
        f"checkpointing.pretrained_model={DEPTHSPLAT_ROOT}/pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth",
        "mode=test",
        f"output_dir={output_dir}",
        *extra_overrides,
    ]

    env = os.environ.copy()
    env.setdefault("SKVIDEO_FFMPEG_PATH", "/usr/bin")
    env["DINOV2_LOCAL_REPO"] = "/root/autodl-tmp/dinov2_source"
    env["DINOV2_CHECKPOINT_DIR"] = "/root/autodl-tmp/dinov2_checkpoints"
    pythonpath = env.get("PYTHONPATH", "")
    dinov2_path = "/root/autodl-tmp/dinov2_source"
    env["PYTHONPATH"] = f"{dinov2_path}:{pythonpath}" if pythonpath else dinov2_path

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"

    print(f"[DepthSplat] Running: {' '.join(cmd)}")
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        result = subprocess.run(
            cmd,
            cwd=str(DEPTHSPLAT_ROOT),
            env=env,
            stdout=out,
            stderr=err,
            text=True,
        )

    if result.returncode != 0:
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="ignore")[-5000:]
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="ignore")[-2000:]
        raise RuntimeError(
            f"DepthSplat exited with code {result.returncode}\n"
            f"stderr tail: {stderr_text}\n"
            f"stdout tail: {stdout_text}"
        )

    depth_npy_dir = output_dir / "images" / scene_key / "depth"
    if not depth_npy_dir.exists():
        depth_npy_dir = output_dir / "metrics" / "images" / scene_key / "depth"

    npy_files = sorted(depth_npy_dir.glob("*.npy"))
    if not npy_files:
        raise RuntimeError(f"No depth .npy files found in {depth_npy_dir}")

    print(f"[DepthSplat] Found {len(npy_files)} depth .npy files")
    return depth_npy_dir


def load_depthsplat_depth(depth_dir: Path) -> dict[int, np.ndarray]:
    """Load depth maps from DepthSplat output directory.

    Returns: dict mapping frame_index (int) -> depth (H, W) float32
    """
    depths = {}
    for path in sorted(depth_dir.glob("*.npy")):
        index = int(path.stem)
        depths[index] = np.load(path)
    return depths
