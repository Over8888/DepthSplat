#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# DepthSplat Demo — Continuous Sequence Mode
#
# 用法:
#   bash run_demo.sh [--2view|--4view|--6view] [data_dir] [output_dir]
#
# 示例:
#   bash run_demo.sh --2view
#   bash run_demo.sh --4view /root/autodl-tmp/datasets/image_sets/Family /root/autodl-tmp/outputs/family_4view
# ============================================================

PYTHON=/root/miniconda3/envs/depthsplat/bin/python
CHECKPOINT=/root/depthsplat/pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth

DATA_DIR="${2:-/root/autodl-tmp/datasets/image_sets/Family}"
OUTPUT_DIR="${3:-/root/autodl-tmp/outputs/demo_${1#--}}"

export DINOV2_LOCAL_REPO=/root/autodl-tmp/dinov2_source
export DINOV2_CHECKPOINT_DIR=/root/autodl-tmp/dinov2_checkpoints

MODE="${1:---2view}"
case "$MODE" in
    --2view) FRAMES="1 152" ;;
    --4view) FRAMES="1 51 101 152" ;;
    --6view) FRAMES="1 34 67 100 133 152" ;;
    *) echo "用法: bash run_demo.sh [--2view|--4view|--6view] [data_dir] [output_dir]"; exit 1 ;;
esac

IMAGES=""
CAMERAS=""
for f in $FRAMES; do
    pf=$(printf "%05d" $f)
    IMAGES="$IMAGES ${DATA_DIR}/${pf}.jpg"
    CAMERAS="$CAMERAS ${DATA_DIR}/${pf}_camera.json"
done

SCENE=$(basename "$DATA_DIR")

echo "Mode: $MODE | Scene: $SCENE | Frames: $FRAMES"
echo "Output: $OUTPUT_DIR"

rm -rf "$OUTPUT_DIR"

$PYTHON -m src.inference.demo \
    --checkpoint "$CHECKPOINT" \
    --model-size base \
    --shim-patch-size 16 \
    --gaussian-scale-max 0.1 \
    --upsample-factor 4 \
    --lowest-feature-resolution 8 \
    --max-image-size 448 \
    --context-images $IMAGES \
    --context-cameras $CAMERAS \
    --data-dir "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --scene "$SCENE" \
    --max-image-size 256 \
    --render-depth --save-ply --save-renders \
    --save-depth-npy --save-depth-concat --save-video \
    --device cuda

echo ""
echo "Done. Output: $OUTPUT_DIR"
echo "Structure:"
find "$OUTPUT_DIR" -type f | head -20
echo "..."
echo "Total files: $(find "$OUTPUT_DIR" -type f | wc -l)"
