#!/usr/bin/env bash
set -euo pipefail
# Manual input evaluation using the official DepthSplat test pipeline.
# Required: DATA_ROOT and EVAL_INDEX. Optional: OUTPUT_DIR.

PROJECT_ROOT="${PROJECT_ROOT:-/root/depthsplat}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/depthsplat/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the prepared dataset root}"
EVAL_INDEX="${EVAL_INDEX:?Set EVAL_INDEX to the evaluation_index.json path}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/outputs/depthsplat_best_configs/manual_base}"
EXPERIMENT="${EXPERIMENT:-dl3dv}"
IMAGE_SHAPE="${IMAGE_SHAPE:-512,960}"
ORI_IMAGE_SHAPE="${ORI_IMAGE_SHAPE:-540,960}"
RENDER_CHUNK_SIZE="${RENDER_CHUNK_SIZE:-10}"
METRIC_CHUNK_SIZE="${METRIC_CHUNK_SIZE:-4}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"
SAVE_IMAGE="${SAVE_IMAGE:-true}"
SAVE_GT_IMAGE="${SAVE_GT_IMAGE:-true}"
SAVE_INPUT_IMAGES="${SAVE_INPUT_IMAGES:-true}"
SAVE_DEPTH="${SAVE_DEPTH:-true}"
SAVE_DEPTH_CONCAT_IMG="${SAVE_DEPTH_CONCAT_IMG:-true}"
SAVE_DEPTH_NPY="${SAVE_DEPTH_NPY:-false}"
SAVE_GAUSSIAN="${SAVE_GAUSSIAN:-true}"
COMPUTE_SCORES="${COMPUTE_SCORES:-true}"
TEST_CHUNK_INTERVAL="${TEST_CHUNK_INTERVAL:-true}"

export SKVIDEO_FFMPEG_PATH="${SKVIDEO_FFMPEG_PATH:-/usr/bin}"
export DINOV2_LOCAL_REPO="${DINOV2_LOCAL_REPO:-/root/autodl-tmp/dinov2_source}"
export DINOV2_CHECKPOINT_DIR="${DINOV2_CHECKPOINT_DIR:-/root/autodl-tmp/dinov2_checkpoints}"
export CUDA_VISIBLE_DEVICES

cd "$PROJECT_ROOT"
dataset_shape_args=("dataset.image_shape=[${IMAGE_SHAPE}]")
if [[ "$EXPERIMENT" == "dl3dv" ]]; then
  dataset_shape_args+=("dataset.ori_image_shape=[${ORI_IMAGE_SHAPE}]")
fi

"$PYTHON_BIN" -m src.main +experiment="$EXPERIMENT" \
  dataset.test_chunk_interval="$TEST_CHUNK_INTERVAL" \
  dataset.roots="[$DATA_ROOT]" \
  "${dataset_shape_args[@]}" \
  model.encoder.num_scales="${NUM_SCALES:-2}" \
  model.encoder.upsample_factor="${UPSAMPLE_FACTOR:-4}" \
  model.encoder.lowest_feature_resolution="${LOWEST_FEATURE_RESOLUTION:-8}" \
  model.encoder.monodepth_vit_type="${VIT_TYPE:-vitb}" \
  model.encoder.gaussian_adapter.gaussian_scale_max="${GAUSSIAN_SCALE_MAX:-0.1}" \
  checkpointing.pretrained_model="${CKPT_PATH:-$PROJECT_ROOT/pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth}" \
  mode=test \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.num_context_views="${NUM_CONTEXT_VIEWS:-6}" \
  dataset.view_sampler.index_path="$EVAL_INDEX" \
  test.save_video="$SAVE_VIDEO" \
  test.save_image="$SAVE_IMAGE" \
  test.save_gt_image="$SAVE_GT_IMAGE" \
  test.save_input_images="$SAVE_INPUT_IMAGES" \
  test.save_depth="$SAVE_DEPTH" \
  test.save_depth_npy="$SAVE_DEPTH_NPY" \
  test.save_depth_concat_img="$SAVE_DEPTH_CONCAT_IMG" \
  test.save_gaussian="$SAVE_GAUSSIAN" \
  test.compute_scores="$COMPUTE_SCORES" \
  test.metric_chunk_size="$METRIC_CHUNK_SIZE" \
  test.render_chunk_size="$RENDER_CHUNK_SIZE" \
  output_dir="$OUTPUT_DIR"
