#!/usr/bin/env bash
set -euo pipefail
# ================================================================
# DepthSplat Base | re10k | 2-view | 256×256
# Checkpoint: depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth
# Training:   re10k | 2 views | 117M params | vit_type: vitb
# ================================================================

export SKVIDEO_FFMPEG_PATH=/usr/bin
export DINOV2_LOCAL_REPO=/root/autodl-tmp/dinov2_source
export DINOV2_CHECKPOINT_DIR=/root/autodl-tmp/dinov2_checkpoints

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
dataset.roots=[/root/depthsplat/datasets/re10k_360p] \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.save_gaussian=true \
test.compute_scores=false \
output_dir=outputs/depthsplat-re10k-base-2view
