from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_scene_data(scene_id: str, dataset_root: Path, image_h: int = 256, image_w: int = 256):
    """Extract context images and cameras for one scene from RE10K dataset.

    Loads directly from the pre-serialized .torch chunk files.
    """
    dataset_root = Path(dataset_root)
    test_dir = dataset_root / "test"
    with open(test_dir / "index.json", "r") as f:
        index = json.load(f)

    chunk_rel = index.get(scene_id)
    if chunk_rel is None:
        train_dir = dataset_root / "train"
        if train_dir.exists():
            with open(train_dir / "index.json", "r") as f:
                train_index = json.load(f)
            chunk_rel = train_index.get(scene_id)
            base_dir = train_dir
        else:
            base_dir = test_dir
    else:
        base_dir = test_dir

    if chunk_rel is None:
        raise RuntimeError(f"Scene {scene_id} not found in dataset {dataset_root}")

    chunk_path = base_dir / chunk_rel
    chunk = torch.load(chunk_path, map_location="cpu")

    scene_data = None
    for item in chunk:
        if item["key"] == scene_id:
            scene_data = item
            break

    if scene_data is None:
        raise RuntimeError(f"Scene {scene_id} not found in chunk {chunk_path}")

    eval_index_path = Path("/root/depthsplat/assets/evaluation_index_re10k.json")
    with open(eval_index_path) as f:
        eval_index = json.load(f)

    eval_entry = eval_index.get(scene_id)
    if eval_entry is None:
        raise RuntimeError(f"Scene {scene_id} not in evaluation index")

    context_indices = eval_entry["context"]
    cameras = scene_data["cameras"]
    images_raw = scene_data["images"]

    extrinsics, intrinsics = _convert_poses(cameras)
    context_extrinsics = extrinsics[context_indices].numpy()
    context_intrinsics = intrinsics[context_indices].numpy()
    context_indices_arr = np.array(context_indices, dtype=np.int64)

    context_images = []
    for idx in context_indices:
        raw_tensor = images_raw[idx]
        img = Image.open(io.BytesIO(raw_tensor.numpy().tobytes()))
        img = img.convert("RGB")
        img = img.resize((image_w, image_h), Image.LANCZOS)
        context_images.append(np.array(img))

    context_images_np = np.stack(context_images, axis=0)

    return {
        "images": context_images_np,
        "extrinsics": context_extrinsics,
        "intrinsics": context_intrinsics,
        "indices": context_indices_arr,
        "near": 1.0,
        "far": 100.0,
        "scene": scene_id,
    }


def _convert_poses(poses: torch.Tensor):
    b = poses.shape[0]
    intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(b, 1, 1).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    w2c = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(b, 1, 1).clone()
    w2c[:, :3] = poses[:, 6:].reshape(b, 3, 4)
    extrinsics_c2w = w2c.inverse()
    return extrinsics_c2w, intrinsics


def save_scene_data(data: dict, output_dir: Path) -> list[Path]:
    """Save scene images and cameras to output_dir. Returns list of image paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for i, img in enumerate(data["images"]):
        path = output_dir / f"image_{i:06d}.png"
        Image.fromarray(img).save(path)
        image_paths.append(path)

    np.savez(
        output_dir / "cameras.npz",
        extrinsics=data["extrinsics"],
        intrinsics=data["intrinsics"],
        near=np.array(data["near"]),
        far=np.array(data["far"]),
        indices=data["indices"],
    )
    return image_paths


def run_subprocess(cmd: list[str], cwd: Path, env: dict | None = None) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    import os
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, env=full_env
    )
    return proc.returncode, proc.stdout, proc.stderr
