#!/usr/bin/env bash
set -euo pipefail
# Official DepthSplat best RE10K config: large model, 2-view, 256x256.

PROJECT_ROOT="${PROJECT_ROOT:-/root/depthsplat}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/RealEstate10K}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/outputs/depthsplat_best_configs/re10k_large_2view}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/depthsplat/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export SKVIDEO_FFMPEG_PATH="${SKVIDEO_FFMPEG_PATH:-/usr/bin}"
export DINOV2_LOCAL_REPO="${DINOV2_LOCAL_REPO:-/root/autodl-tmp/dinov2_source}"
export DINOV2_CHECKPOINT_DIR="${DINOV2_CHECKPOINT_DIR:-/root/autodl-tmp/dinov2_checkpoints}"
export CUDA_VISIBLE_DEVICES

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m src.main +experiment=re10k \
  dataset.test_chunk_interval="${TEST_CHUNK_INTERVAL:-1}" \
  dataset.roots="[$DATA_ROOT]" \
  model.encoder.num_scales=2 \
  model.encoder.upsample_factor=2 \
  model.encoder.lowest_feature_resolution=4 \
  model.encoder.monodepth_vit_type=vitl \
  checkpointing.pretrained_model="${CKPT_PATH:-pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth}" \
  mode=test \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.index_path="${EVAL_INDEX:-assets/evaluation_index_re10k_video.json}" \
  test.save_video="${SAVE_VIDEO:-true}" \
  test.save_image="${SAVE_IMAGE:-true}" \
  test.save_gt_image="${SAVE_GT_IMAGE:-true}" \
  test.save_input_images="${SAVE_INPUT_IMAGES:-true}" \
  test.save_depth="${SAVE_DEPTH:-true}" \
  test.save_depth_npy="${SAVE_DEPTH_NPY:-false}" \
  test.save_depth_concat_img="${SAVE_DEPTH_CONCAT:-true}" \
  test.save_gaussian="${SAVE_GAUSSIAN:-true}" \
  test.compute_scores="${COMPUTE_SCORES:-true}" \
  test.metric_chunk_size="${METRIC_CHUNK_SIZE:-4}" \
  output_dir="$OUTPUT_DIR"