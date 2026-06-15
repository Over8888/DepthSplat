#!/usr/bin/env bash
set -euo pipefail
# ================================================================
# DepthSplat Small | re10k+DL3DV | 6-view | 512×960
# Checkpoint: depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth
# Training:   re10k → (re10k+dl3dv) | 4-10 views | 37M params | vits
# ================================================================

export SKVIDEO_FFMPEG_PATH=/usr/bin
export DINOV2_LOCAL_REPO=/root/autodl-tmp/dinov2_source
export DINOV2_CHECKPOINT_DIR=/root/autodl-tmp/dinov2_checkpoints

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
dataset.test_chunk_interval=1 \
dataset.roots=[/root/depthsplat/datasets/re10k_360p] \
dataset.image_shape=[512,960] \
dataset.ori_image_shape=[720,1280] \
model.encoder.upsample_factor=8 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.gaussian_adapter.gaussian_scale_max=0.1 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/re10k_ctx_6v_video.json \
test.save_video=true \
test.save_gaussian=true \
test.compute_scores=false \
test.render_chunk_size=10 \
output_dir=outputs/depthsplat-re10kdl3dv-small-6view
