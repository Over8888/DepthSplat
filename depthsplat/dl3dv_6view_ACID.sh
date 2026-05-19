CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
mode=test \
dataset.roots=[/root/autodl-tmp/acid] \
dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=6 \
model.encoder.upsample_factor=8 \
model.encoder.lowest_feature_resolution=8 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth \
test.compute_scores=true \
output_dir=/root/autodl-tmp/outputs/depthsplat_dl3dv_6view_ACID_eval