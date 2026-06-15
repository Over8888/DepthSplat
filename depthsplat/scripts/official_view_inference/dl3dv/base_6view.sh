#!/usr/bin/env bash
set -euo pipefail
# ================================================================
# DepthSplat Base | DL3DV | 6-view | 256×448
# Checkpoint: depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth
# Training:   re10k → dl3dv | 2-6 views | 117M params | vitb
# ================================================================

export SKVIDEO_FFMPEG_PATH=/usr/bin
export DINOV2_LOCAL_REPO=/root/autodl-tmp/dinov2_source
export DINOV2_CHECKPOINT_DIR=/root/autodl-tmp/dinov2_checkpoints

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
dataset.test_chunk_interval=1 \
dataset.roots=[/root/depthsplat/datasets/re10k_360p] \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
test.save_video=true \
test.save_gaussian=true \
test.compute_scores=false \
output_dir=outputs/depthsplat-dl3dv-base-6view
