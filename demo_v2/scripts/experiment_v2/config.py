from __future__ import annotations

from pathlib import Path

ROOT = Path("/root/autodl-tmp/demo_v2")
OUTPUT_DIR = ROOT / "outputs" / "experiment_v2"

DATASET_ROOT = Path("/root/depthsplat/datasets/re10k_360p")
EVAL_INDEX_PATH = Path("/root/depthsplat/assets/evaluation_index_re10k_video.json")

DEPTHSPLAT_ROOT = Path("/root/depthsplat")
CONDA_PYTHON = "/root/miniconda3/envs/depthsplat/bin/python"
DEPTHSPLAT_PYTHON = CONDA_PYTHON
DEPTHSPLAT_CKPT = str(DEPTHSPLAT_ROOT / "pretrained" / "depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth")

MVSPLAT_ROOT = Path("/root/mvsplat")
MVSPLAT_PYTHON = CONDA_PYTHON
MVSPLAT_CKPT = "checkpoints/re10k.ckpt"

VGGT_ROOT = Path("/root/autodl-tmp/vggt")
VGGT_MODEL_ID = "facebook/VGGT-1B"

DA3_ROOT = Path("/root/autodl-tmp/Depth-Anything-3")
DA3_PRESET = "da3-base"

SCENE_KEYS = [
    "322261824c4a3003",
    "89ea49cd9865aeff",
    "f7c0fa5b81552d35",
]
SCENE_CHUNK_FILES = {
    "322261824c4a3003": "000001.torch",
    "89ea49cd9865aeff": "000000.torch",
    "f7c0fa5b81552d35": "000001.torch",
}

CONTEXT_INDICES = {
    "322261824c4a3003": [33, 78],
    "89ea49cd9865aeff": [31, 81],
    "f7c0fa5b81552d35": [11, 64],
}

TARGET_INDICES = {
    "322261824c4a3003": list(range(33, 79)),
    "89ea49cd9865aeff": list(range(31, 82)),
    "f7c0fa5b81552d35": list(range(11, 65)),
}

DEPTHSPLAT_RES = (256, 256)
MVSPLAT_RES = (256, 256)
VGGT_RES = 518
DA3_RES = 504

VIEW_ANGLES = {
    "Front": (0, 0),
    "Side": (90, 0),
    "Top": (0, 90),
    "Oblique": (45, 30),
}

MAX_POINTS_PER_CLOUD = 80000
MAX_POINTS_PER_RENDER = 30000

DBSCAN_EPS_RATIO = 0.02
DBSCAN_MIN_SAMPLES = 10
FLOATING_CLUSTER_MIN_PCT = 0.0005
FLOATING_CLUSTER_MAX_PCT = 0.03
FLOATING_DISTANCE_RATIO = 3.0

NORMAL_NEIGHBORS = 30
NORMAL_BREAK_THRESHOLD_DEG = 35.0

EXP4_ROOT = ROOT / "outputs" / "experiment_v4"
OPACITY_FILTER_THRESHOLD_GS = 0.5
OPACITY_FILTER_THRESHOLD_PC = 0.1
SPLATTING_RADIUS_PX = 3
SCALE_MAP_LOG_MIN = -5.0
SCALE_MAP_LOG_MAX = 1.0
MAX_POINTS_PER_PANEL = 30000
DIAGNOSTIC_PANEL_DPI = 150

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXP4_ROOT.mkdir(parents=True, exist_ok=True)
EXP5_ROOT = ROOT / "outputs" / "experiment_v5"
EXP5_ROOT.mkdir(parents=True, exist_ok=True)
