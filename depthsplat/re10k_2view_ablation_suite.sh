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
mkdir -p /root/autodl-tmp/outputs
cd "${ROOT_DIR}"

run_eval \
  re10k_2view_baseline \
  model.encoder.num_depth_candidates=128

for ratio in 0.01 0.03 0.05 0.10; do
  safe_ratio="${ratio/./p}"
  run_eval \
    "re10k_2view_noise_translation_t${safe_ratio}" \
    model.encoder.num_depth_candidates=128 \
    test.camera_noise.enabled=true \
    test.camera_noise.apply_to=context \
    test.camera_noise.mode=translation \
    test.camera_noise.translation_sigma_ratio="${ratio}" \
    test.camera_noise.rotation_sigma_deg=0.0 \
    test.camera_noise.seed=777
done

for deg in 1.0 3.0 5.0 10.0; do
  safe_deg="${deg/./p}"
  run_eval \
    "re10k_2view_noise_rotation_r${safe_deg}" \
    model.encoder.num_depth_candidates=128 \
    test.camera_noise.enabled=true \
    test.camera_noise.apply_to=context \
    test.camera_noise.mode=rotation \
    test.camera_noise.translation_sigma_ratio=0.0 \
    test.camera_noise.rotation_sigma_deg="${deg}" \
    test.camera_noise.seed=777
done

for candidates in 64 96 128; do
  run_eval \
    "re10k_2view_depthcands_${candidates}" \
    model.encoder.num_depth_candidates="${candidates}"
done

"${PYTHON_BIN}" scripts/summarize_ablation_results.py "${OUTPUT_ROOT}"
"${PYTHON_BIN}" scripts/generate_ablation_report.py "${OUTPUT_ROOT}"
"${PYTHON_BIN}" scripts/export_ablation_examples.py "${OUTPUT_ROOT}"
