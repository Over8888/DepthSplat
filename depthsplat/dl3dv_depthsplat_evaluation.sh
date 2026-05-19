# Table 1 of depthsplat paper
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
dataset.roots=[/root/autodl-tmp/RealEstate10K] \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
mode=test \
test.compute_scores=true \
dataset/view_sampler=evaluation \
output_dir=/root/autodl-tmp/outputs/depthsplat_dl3dv_6view_eval
