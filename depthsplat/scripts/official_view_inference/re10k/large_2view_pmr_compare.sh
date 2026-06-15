#!/usr/bin/env bash
set -euo pipefail
# ================================================================
# DepthSplat Large | re10k | 2-view | 256×256 | PMR A/B
# Compares PMR disabled vs enabled.
# Saves rendered RGB, context depth maps, gaussian PLY, and metrics.
# ================================================================

PROJECT_ROOT="${PROJECT_ROOT:-/root/depthsplat}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/RealEstate10K}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/depthsplat_re10k_large_2view_pmr_compare}"
CKPT_PATH="${CKPT_PATH:-$PROJECT_ROOT/pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth}"
EVAL_INDEX="${EVAL_INDEX:-assets/evaluation_index_re10k_video.json}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/depthsplat/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export SKVIDEO_FFMPEG_PATH="${SKVIDEO_FFMPEG_PATH:-/usr/bin}"
export DINOV2_LOCAL_REPO="${DINOV2_LOCAL_REPO:-/root/autodl-tmp/dinov2_source}"
export DINOV2_CHECKPOINT_DIR="${DINOV2_CHECKPOINT_DIR:-/root/autodl-tmp/dinov2_checkpoints}"
export CUDA_VISIBLE_DEVICES

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "Project root not found: $PROJECT_ROOT" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT/test" ]]; then
  echo "RealEstate10K test split not found: $DATA_ROOT/test" >&2
  exit 1
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi

if [[ ! -d "$DINOV2_LOCAL_REPO" ]]; then
  echo "DINOv2 local repo not found: $DINOV2_LOCAL_REPO" >&2
  exit 1
fi

if [[ ! -d "$DINOV2_CHECKPOINT_DIR" ]]; then
  echo "DINOv2 checkpoint dir not found: $DINOV2_CHECKPOINT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

run_eval() {
  local tag="$1"
  local pmr_enabled="$2"
  local out_dir="$OUTPUT_ROOT/$tag"
  mkdir -p "$out_dir"

  echo "Running $tag: PMR=$pmr_enabled, output=$out_dir"
  (
    cd "$PROJECT_ROOT"
    "$PYTHON_BIN" -m src.main \
      +experiment=re10k \
      mode=test \
      wandb.mode=disabled \
      dataset.test_chunk_interval=100 \
      dataset.roots="[$DATA_ROOT]" \
      dataset/view_sampler=evaluation \
      dataset.view_sampler.index_path="$EVAL_INDEX" \
      data_loader.test.batch_size=1 \
      data_loader.test.num_workers=4 \
      model.encoder.num_scales=2 \
      model.encoder.upsample_factor=2 \
      model.encoder.lowest_feature_resolution=4 \
      model.encoder.monodepth_vit_type=vitl \
      model.encoder.cost_volume_confidence="$pmr_enabled" \
      model.encoder.pmr_guided_smooth="$pmr_enabled" \
      checkpointing.pretrained_model="$CKPT_PATH" \
      checkpointing.no_strict_load=false \
      test.save_video=false \
      test.save_image=true \
      test.save_gt_image=true \
      test.save_input_images=true \
      test.save_depth=true \
      test.save_depth_npy=true \
      test.save_depth_concat_img=true \
      test.save_gaussian=true \
      test.compute_scores=true \
      test.metric_chunk_size=16 \
      output_dir="$out_dir" \
      use_plugins=false 2>&1 | tee "$out_dir/eval.log"
  )
}

run_eval "pmr_off" "false"
run_eval "pmr_on" "true"

python - <<PY
import csv
import json
from pathlib import Path

root = Path("$OUTPUT_ROOT")

def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None

summary = {
    "pmr_off": {
        "output_dir": str(root / "pmr_off"),
        "metrics": read_json(root / "pmr_off" / "metrics" / "scores_all_avg.json"),
        "per_scene": str(root / "pmr_off" / "metrics" / "scores_per_scene.csv"),
        "renders": str(root / "pmr_off" / "images"),
        "gaussians": str(root / "pmr_off" / "gaussians"),
    },
    "pmr_on": {
        "output_dir": str(root / "pmr_on"),
        "metrics": read_json(root / "pmr_on" / "metrics" / "scores_all_avg.json"),
        "per_scene": str(root / "pmr_on" / "metrics" / "scores_per_scene.csv"),
        "renders": str(root / "pmr_on" / "images"),
        "gaussians": str(root / "pmr_on" / "gaussians"),
    },
}

metrics_off = summary["pmr_off"]["metrics"] or {}
metrics_on = summary["pmr_on"]["metrics"] or {}
summary["delta_on_minus_off"] = {
    key: metrics_on[key] - metrics_off[key]
    for key in ("psnr", "ssim", "lpips")
    if isinstance(metrics_off.get(key), (int, float)) and isinstance(metrics_on.get(key), (int, float))
}

out_path = root / "comparison_summary.json"
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"Wrote {out_path}")
PY