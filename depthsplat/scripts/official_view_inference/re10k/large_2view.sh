#!/usr/bin/env bash
set -euo pipefail
# ================================================================
# DepthSplat Large | re10k | 2-view | 256×256
# Checkpoint: depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth
# Training:   re10k | 2 views | 360M params | vit_type: vitl
# ================================================================

export SKVIDEO_FFMPEG_PATH=/usr/bin
export DINOV2_LOCAL_REPO=/root/autodl-tmp/dinov2_source
export DINOV2_CHECKPOINT_DIR=/root/autodl-tmp/dinov2_checkpoints

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
dataset.roots=[/root/autodl-tmp/RealEstate10K] \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitl \
checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=false \
test.save_gaussian=false \
test.compute_scores=true \
output_dir=outputs/depthsplat-re10k-large-2view
