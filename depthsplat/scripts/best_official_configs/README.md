# DepthSplat best official configs

本目录包含 DepthSplat 官方模型的推理脚本，分为两大类使用场景：

1. **RE10K 数据集评测**：在 RE10K 测试集上跑定量/定性评估
2. **手动输入推理**：用自己的图片 + 位姿做 3D Gaussian Splatting 重建

## 使用场景一：RE10K 数据集评测

### 脚本矩阵

| 模型大小 | 2 views | 4 views | 6 views | 说明 |
|---|---|---|---|---|
| large | `re10k_large_2view.sh` | `re10k_large_4view.sh` | `re10k_large_6view.sh` | 2-view 用纯 RE10K checkpoint；4/6 view 用 base re10k+dl3dv checkpoint |
| base | `re10k_base_2view.sh` | `re10k_base_4view.sh` | `re10k_base_6view.sh` | 4/6 view 使用 re10k+dl3dv 混合训练 checkpoint |
| small | `re10k_small_2view.sh` | `re10k_small_4view.sh` | `re10k_small_6view.sh` | 4/6 view 使用 re10k+dl3dv 混合训练 checkpoint |

### Checkpoint 与分辨率对应关系

| 脚本 | Checkpoint | 训练分辨率 | 训练视角数 |
|---|---|---|---|
| `re10k_*_2view.sh` | 纯 RE10K checkpoint | 256x256 | 2 |
| `re10k_large_{4,6}view.sh` | `depthsplat-gs-base-re10kdl3dv-448x768-randview2-6` | 448x768 (512x960 推理) | 2-6 |
| `re10k_base_{4,6}view.sh` | `depthsplat-gs-base-re10kdl3dv-448x768-randview2-6` | 448x768 (512x960 推理) | 2-6 |
| `re10k_small_{4,6}view.sh` | `depthsplat-gs-small-re10kdl3dv-448x768-randview4-10` | 448x768 (512x960 推理) | 4-10 |

### 关于 large 的 4/6 view 脚本

官方模型库没有提供 large 的多视角 checkpoint。`re10k_large_4view.sh` 和 `re10k_large_6view.sh` 实际使用 base re10k+dl3dv checkpoint 和 base 整套配置，命名保持 large 是为了脚本矩阵完整性。

### 用法示例

```bash
DATA_ROOT=/root/autodl-tmp/RealEstate10K \
TEST_CHUNK_INTERVAL=100 \
COMPUTE_SCORES=true \
OUTPUT_DIR=/root/autodl-tmp/outputs/re10k_large_debug \
bash /root/depthsplat/scripts/best_official_configs/re10k_large_2view.sh
```

## 使用场景二：手动输入推理（自定义图片 + 位姿）

### 脚本说明

`manual_base.sh` 使用 `src.inference.demo` 入口，适合对自己拍摄/采集的图片做 3DGS 重建。

统一使用 base re10k+dl3dv 混合训练 checkpoint（`depthsplat-gs-base-re10kdl3dv-448x768-randview2-6`），配置参数已对齐该 checkpoint 的训练设置：
- `model-size=base`, `num-scales=2`, `upsample-factor=4`, `lowest-feature-resolution=8`, `gaussian-scale-max=0.1`

### 为什么手动推理用混合训练 checkpoint 而不是纯 RE10K？

混合训练（re10k+dl3dv）的 checkpoint 见过更多样化的场景（室内外、不同尺度），泛化能力更强。纯 RE10K checkpoint 仅在室内视频数据上训练，面对手动输入的多样化场景容易出现 domain gap。

### 必填变量

- `IMAGES`：图片目录路径，或空格分隔的图片路径列表
- `POSES`：位姿文件，支持 `transforms.json`、`.npz`、`.npy`、COLMAP 格式

### 用法示例

```bash
IMAGES="/path/to/images" \
POSES="/path/to/transforms.json" \
OUTPUT_DIR=/root/autodl-tmp/outputs/manual_base \
SAVE_PLY=1 \
SAVE_DEPTH_NPY=1 \
bash /root/depthsplat/scripts/best_official_configs/manual_base.sh
```

## 通用可选变量

所有脚本均支持通过环境变量覆盖默认值：

**数据集评测脚本：**
- `DATA_ROOT`：RE10K 数据集根目录
- `OUTPUT_DIR`：输出目录
- `CKPT_PATH`：覆盖默认 checkpoint 路径
- `EVAL_INDEX`：评测索引文件
- `SAVE_VIDEO`, `SAVE_IMAGE`, `SAVE_GT_IMAGE`, `SAVE_INPUT_IMAGES`
- `SAVE_DEPTH`, `SAVE_DEPTH_NPY`, `SAVE_DEPTH_CONCAT`, `SAVE_GAUSSIAN`
- `COMPUTE_SCORES`：是否计算 PSNR/SSIM/LPIPS
- `TEST_CHUNK_INTERVAL`：每隔 N 个场景取一个（用于快速调试）
- `RENDER_CHUNK_SIZE`：渲染 batch size

**手动推理脚本额外支持：**
- `TARGET_POSES`：目标渲染位姿文件
- `MAX_IMAGE_SIZE`：输入图片最大边长（默认 256）
- `NEAR` / `FAR`：近远平面距离
- `SAVE_PLY`：保存高斯点云 ply 文件