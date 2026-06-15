from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path("/root")
OUTPUT_ROOT = Path("/root/autodl-tmp/outputs/exp3_consistency")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEPTHSPLAT_ROOT = PROJECT_ROOT / "depthsplat"
MVSPLAT_ROOT = PROJECT_ROOT / "mvsplat"
VGGT_ROOT = PROJECT_ROOT / "autodl-tmp" / "vggt"
DA3_ROOT = PROJECT_ROOT / "autodl-tmp" / "Depth-Anything-3"

DATASET_ROOT = PROJECT_ROOT / "autodl-tmp" / "RealEstate10K"
DEPTHSPLAT_DATASET_ROOT = DEPTHSPLAT_ROOT / "datasets" / "re10k"

PYTHON_BIN = "/root/miniconda3/envs/depthsplat/bin/python"

SCENE_IDS = [
    "5aca87f95a9412c6",
    "debc3490ba0bd84b",
    "e4bcb18fa6aa91be",
    "7c7bc5285126e6ad",
    "cdf439b17a6a98d4",
]

METHODS = ["ours", "mvsplat", "vggt", "da3"]

IMAGE_H = 256
IMAGE_W = 256

VGGT_RESOLUTION = 518
DA3_RESOLUTION = 504

DINOV2_LOCAL_REPO = "/root/autodl-tmp/dinov2_source"
DINOV2_CHECKPOINT_DIR = "/root/autodl-tmp/dinov2_checkpoints"


def get_scene_output_dir(scene_id: str) -> Path:
    return OUTPUT_ROOT / scene_id


def get_method_dir(scene_id: str, method: str) -> Path:
    return get_scene_output_dir(scene_id) / method


def load_evaluation_index() -> dict:
    path = DEPTHSPLAT_ROOT / "assets" / "evaluation_index_re10k.json"
    with open(path) as f:
        return json.load(f)
