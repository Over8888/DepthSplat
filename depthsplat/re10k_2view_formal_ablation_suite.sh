#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/root/depthsplat"
OUTPUT_ROOT="${1:-/root/autodl-tmp/outputs/depthsplat_re10k_2view_ablation}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/depthsplat/bin/python}"

COMMON_ARGS=(
  +experiment=re10k
  dataset.test_chunk_interval=1
  dataset.roots=[/root/autodl-tmp/RealEstate10K]
  model.encoder.num_scales=2
  model.encoder.upsample_factor=2
  model.encoder.lowest_feature_resolution=4
  model.encoder.monodepth_vit_type=vitb
  model.encoder.tome.enabled=false
  checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth
  mode=test
  test.compute_scores=true
  test.metric_chunk_size=16
  dataset/view_sampler=evaluation
  dataset.view_sampler.num_context_views=2
  dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json
)

run_eval() {
  local name="$1"
  shift
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" -m src.main \
    "${COMMON_ARGS[@]}" \
    output_dir="${OUTPUT_ROOT}/${name}" \
    "$@"
}

mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"

run_eval \
  formal_near_0p25_far_100 \
  model.encoder.num_depth_candidates=128 \
  dataset.near=0.25 \
  dataset.far=100.0

run_eval \
  formal_near_1p0_far_100 \
  model.encoder.num_depth_candidates=128 \
  dataset.near=1.0 \
  dataset.far=100.0

run_eval \
  formal_near_0p5_far_50 \
  model.encoder.num_depth_candidates=128 \
  dataset.near=0.5 \
  dataset.far=50.0

run_eval \
  formal_near_0p5_far_200 \
  model.encoder.num_depth_candidates=128 \
  dataset.near=0.5 \
  dataset.far=200.0

"${PYTHON_BIN}" scripts/summarize_formal_ablation_results.py "${OUTPUT_ROOT}"
"${PYTHON_BIN}" scripts/generate_formal_ablation_report.py "${OUTPUT_ROOT}"
