#!/usr/bin/env bash
set -euo pipefail
# Official DepthSplat best RE10K-compatible config: small model, 4-view, 512x960.
# Checkpoint training views: 4-10. Uses the official re10k+dl3dv small checkpoint.

PROJECT_ROOT="${PROJECT_ROOT:-/root/depthsplat}"
DATA_ROOT="${DATA_ROOT:-/root/depthsplat/datasets/re10k_720p}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/outputs/depthsplat_best_configs/re10k_small_4view}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/depthsplat/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export SKVIDEO_FFMPEG_PATH="${SKVIDEO_FFMPEG_PATH:-/usr/bin}"
export DINOV2_LOCAL_REPO="${DINOV2_LOCAL_REPO:-/root/autodl-tmp/dinov2_source}"
export DINOV2_CHECKPOINT_DIR="${DINOV2_CHECKPOINT_DIR:-/root/autodl-tmp/dinov2_checkpoints}"
export CUDA_VISIBLE_DEVICES

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m src.main +experiment=dl3dv \
  dataset.test_chunk_interval="${TEST_CHUNK_INTERVAL:-1}" \
  dataset.roots="[$DATA_ROOT]" \
  dataset.image_shape=[512,960] \
  dataset.ori_image_shape=[720,1280] \
  model.encoder.upsample_factor=8 \
  model.encoder.lowest_feature_resolution=8 \
  model.encoder.gaussian_adapter.gaussian_scale_max=0.1 \
  checkpointing.pretrained_model="${CKPT_PATH:-pretrained/depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth}" \
  mode=test \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.num_context_views=4 \
  dataset.view_sampler.index_path="${EVAL_INDEX:-assets/re10k_ctx_4v_video.json}" \
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
  test.render_chunk_size="${RENDER_CHUNK_SIZE:-10}" \
  output_dir="$OUTPUT_DIR"