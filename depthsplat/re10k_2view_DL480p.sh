# render video on dl3dv (need to have ffmpeg installed)
export SKVIDEO_FFMPEG_PATH=/usr/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
dataset.test_chunk_interval=1 \
dataset.roots=[/root/autodl-tmp/datasets/dl3dv_480p] \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_2v_video_0_50.json \
test.save_video=false \
test.stablize_camera=true \
test.save_gaussian=false \
test.compute_scores=true \
test.render_chunk_size=1 \
output_dir=/root/autodl-tmp/outputs/depthsplat-re10k-2view-480