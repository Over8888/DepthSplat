# render video on dl3dv (need to have ffmpeg installed)
export SKVIDEO_FFMPEG_PATH=/usr/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
dataset.roots=[/root/autodl-tmp/dl3dv_960p] \
dataset.skip_bad_shape=false \
dataset.image_shape=[512,960] \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
model.encoder.gaussian_adapter.gaussian_scale_max=0.1 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
test.save_video=false \
test.stablize_camera=true \
test.save_gaussian=false \
test.compute_scores=true \
test.render_chunk_size=1 \
output_dir=/root/autodl-tmp/outputs/depthsplat-re10k-6view-960