# DepthSplat Official View Inference Scripts

Strictly follows [MODEL_ZOO](https://github.com/cvg/depthsplat/blob/main/MODEL_ZOO.md) and [README](https://github.com/cvg/depthsplat).

## Multi-View Capability

| Model | Training Views | Supported Context Views |
|-------|---------------|------------------------|
| small-re10k | 2 | **2 only** |
| base-re10k | 2 | **2 only** |
| large-re10k | 2 | **2 only** |
| small-re10kdl3dv | 4-10 | **4, 6** (no 2) |
| base-re10kdl3dv | 2-6 | **2, 4, 6** |
| base-dl3dv | 2-6 | **2, 4, 6** |

## Script Index

| Script | Checkpoint | Params | Vit | Resolution | Views | Experiment |
|--------|-----------|--------|-----|------------|-------|------------|
| `re10k/small_2view.sh` | gs-small-re10k-256x256-view2 | 37M | vits | 256×256 | 2 | re10k |
| `re10k/base_2view.sh` | gs-base-re10k-256x256-view2 | 117M | vitb | 256×256 | 2 | re10k |
| `re10k/large_2view.sh` | gs-large-re10k-256x256-view2 | 360M | vitl | 256×256 | 2 | re10k |
| `re10kdl3dv/base_2view.sh` | gs-base-re10kdl3dv-448x768-randview2-6 | 117M | vitb | 512×960 | 2 | dl3dv |
| `re10kdl3dv/base_6view.sh` | gs-base-re10kdl3dv-448x768-randview2-6 | 117M | vitb | 512×960 | 6 | dl3dv |
| `re10kdl3dv/small_6view.sh` | gs-small-re10kdl3dv-448x768-randview4-10 | 37M | vits | 512×960 | 6 | dl3dv |
| `dl3dv/base_2view.sh` | gs-base-dl3dv-256x448-randview2-6 | 117M | vitb | 256×448 | 2 | dl3dv |
| `dl3dv/base_4view.sh` | gs-base-dl3dv-256x448-randview2-6 | 117M | vitb | 256×448 | 4 | dl3dv |
| `dl3dv/base_6view.sh` | gs-base-dl3dv-256x448-randview2-6 | 117M | vitb | 256×448 | 6 | dl3dv |

## Usage

```bash
# Pure re10k 2-view (small/base/large)
bash scripts/official_view_inference/re10k/small_2view.sh
bash scripts/official_view_inference/re10k/base_2view.sh
bash scripts/official_view_inference/re10k/large_2view.sh

# Mixed re10kdl3dv 6-view (small/base)
bash scripts/official_view_inference/re10kdl3dv/small_6view.sh
bash scripts/official_view_inference/re10kdl3dv/base_6view.sh

# DL3DV multi-view (2/4/6)
bash scripts/official_view_inference/dl3dv/base_6view.sh
```

## Prerequisites

- DINOv2 source and checkpoints at `/root/autodl-tmp/dinov2_source` and `/root/autodl-tmp/dinov2_checkpoints`
- Datasets in `.torch` chunk format under `/root/depthsplat/datasets/`
- FFmpeg for video output (`export SKVIDEO_FFMPEG_PATH=/usr/bin`)
