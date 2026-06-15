#!/usr/bin/env bash
set -euo pipefail
# ================================================================
# DepthSplat Small | re10k | 2-view | 256×256
# Checkpoint: depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth
# Training:   re10k | 2 views | 37M params | vit_type: vits
# ================================================================

export SKVIDEO_FFMPEG_PATH=/usr/bin
export DINOV2_LOCAL_REPO=/root/autodl-tmp/dinov2_source
export DINOV2_CHECKPOINT_DIR=/root/autodl-tmp/dinov2_checkpoints

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
dataset.roots=[/root/depthsplat/datasets/re10k_360p] \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.save_gaussian=true \
test.compute_scores=false \
output_dir=outputs/depthsplat-re10k-small-2view
