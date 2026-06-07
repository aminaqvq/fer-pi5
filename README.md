# FER-Pi5 — 基于 CNN 与 YuNet 的人脸表情识别系统

> **Real-time Facial Expression Recognition on Raspberry Pi 5**  
> PyTorch 训练 · TFLite 推理 · YuNet 人脸检测 · 三阶段半监督学习

[![Python](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![TensorFlow Lite](https://img.shields.io/badge/TFLite-2.14-orange)](https://www.tensorflow.org/lite)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green)](https://opencv.org/)
[![Hardware](https://img.shields.io/badge/hardware-Raspberry%20Pi%205-C51A4A)](https://www.raspberrypi.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](./LICENSE)

---

## 📖 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [安装指南](#安装指南)
- [数据准备](#数据准备)
- [训练流程](#训练流程)
- [模型评估](#模型评估)
- [模型导出](#模型导出)
- [树莓派部署](#树莓派部署)
- [配置参考](#配置参考)
- [实验结果](#实验结果)
- [常见问题](#常见问题)

---

## 项目概述

FER-Pi5 是一个**端到端的人脸表情识别系统**，覆盖从数据预处理、模型训练、量化导出到边缘设备部署的完整链路。系统在 **PyTorch** 上使用**三阶段半监督策略**训练 MobileNetV3-Large 分类器，通过 **YuNet ONNX** 进行人脸检测，最终以 **TensorFlow Lite FP16** 格式部署到 **树莓派 5** 上实现 **15–20 FPS** 的实时推理。

### 🎯 核心特性

- **7 类表情识别**：anger · disgust · fear · happy · sad · surprise · neutral
- **三阶段半监督训练**：有监督预训练 → 伪标签生成与混合训练 → 精调收敛
- **多模型支持**：MobileNetV3-Large / Small、EfficientNet-B0、RepVGGplus-L2pse
- **边缘部署**：PyTorch → ONNX → TFLite 完整导出链，FP16 量化几乎无损
- **实时推理**：树莓派 5 上 15–20 FPS，含人脸检测 + 分类 + 概率校准 + 时序平滑
- **完整的工具链**：数据清洗、伪标签生成、错误审计、混淆矩阵可视化、模型导出

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    训练管线 (PC/GPU)                      │
│                                                         │
│  FER2013Plus ──┐                                        │
│                ├──→ 数据预处理 ──→ PyTorch 训练           │
│  MMAFEDB ──────┘      │                │                │
│                        │    ┌───────────┴───────────┐    │
│                 RandAugment │  Stage1  有监督预训练   │    │
│                 随机擦除     │  Stage2  伪标签混合训练 │    │
│                 标准化       │  Stage3  精调收敛      │    │
│                             └───────────┬───────────┘    │
│                                         │                │
│                            PyTorch → ONNX → TFLite       │
│                                         │                │
└─────────────────────────────────────────┼────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │             树莓派 5 推理管线               │
                    │                                          │
                    │  USB Camera                               │
                    │      ↓                                    │
                    │  OpenCV VideoCapture (MJPG, 640×480)      │
                    │      ↓                                    │
                    │  YuNet ONNX 人脸检测 (15–20 ms)            │
                    │      ↓                                    │
                    │  人脸框扩展 18% → 裁剪 → 224×224           │
                    │      ↓                                    │
                    │  ImageNet 标准化                           │
                    │      ↓                                    │
                    │  TFLite FP16 推理 (45–55 ms)              │
                    │      ↓                                    │
                    │  Softmax → 概率校准 → 时序平滑             │
                    │      ↓                                    │
                    │  OSD 显示 + 可选保存最佳样本               │
                    │                                          │
                    └──────────────────────────────────────────┘
```

---

## 项目结构

```
fer-pi5/
├── src/
│   ├── training/                    # ★ 训练核心代码
│   │   ├── train_core.py            #    通用训练引擎（配置、循环、验证、日志）
│   │   ├── dataset.py               #    数据集加载、数据增强
│   │   ├── metrics.py               #    评价指标（Macro-F1、混淆矩阵等）
│   │   ├── model_mbv3.py            #    MobileNetV3 模型工厂
│   │   ├── model_efficientnet.py    #    EfficientNet-B0 模型工厂
│   │   ├── model_repvggplus.py      #    RepVGGplus-L2pse 从头实现
│   │   ├── se_block.py              #    SE 注意力模块
│   │   ├── pseudo_core.py           #    伪标签生成引擎
│   │   ├── balanced_sampler_patch.py#    Monkey-patch：平衡批次采样
│   │   ├── export_final_model.py    #    PyTorch → ONNX → TFLite 导出
│   │   ├── evaluate.py              #    独立评估脚本（含 TTA）
│   │   ├── audit_model_errors.py    #    错误分析可视化
│   │   ├── scan_and_clean_fer_csv.py#    CSV 数据清洗
│   │   ├── review_label_issues_gui.py#   标签审核 GUI
│   │   ├── apply_manual_review.py   #    手动审核结果应用
│   │   ├── build_checkpoint_registry.py# Checkpoint 注册与追溯
│   │   ├── train_stage1.py          #    Stage1 入口
│   │   ├── train_stage2_balanced_clean.py # Stage2 入口
│   │   ├── train_stage3_final.py    #    Stage3 入口
│   │   ├── generate_pseudo_stage1_clean.py  # Stage1 伪标签生成
│   │   ├── generate_pseudo_stage2_final.py  # Stage2 伪标签生成
│   │   ├── out_of_fold_train_audit.py      # K-fold 审计
│   │   └── archive/                 #    历史实验脚本（参考）
│   │
│   └── deploy/                      # ★ 树莓派部署代码
│       ├── infer_pi.py              #    实时摄像头推理主程序
│       ├── infer_video.py           #    离线视频文件推理
│       └── face_detection_yunet_2023mar.onnx  # YuNet 检测模型
│
├── data/                            # 数据集（.gitignore）
│   ├── csv/                         #   训练/验证/测试 CSV 索引
│   ├── fer2013plus/                 #   FER2013Plus 图片
│   └── MMAFEDB/                     #   MMPAEDB 图片
│
├── checkpoints/                     # 训练产出模型权重
├── runs/training/                   # 训练日志与指标
├── export/                          # 导出的 TFLite 模型
├── environment/                     # Conda 环境文件
│   ├── fer_pi/                      #   树莓派推理环境
│   ├── yunet/                       #   YuNet 环境
│   └── convert/                     #   模型转换环境
├── docs/                            # 论文、PPT、参考论文等
└── AGENTS.md                        # 项目工作约束
```

---

## 环境依赖

### 训练环境 (Windows/Linux + GPU)

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.9+ | |
| PyTorch | 2.x | 含 torchvision |
| CUDA | 11.8 / 12.1 | GPU 训练 |
| OpenCV | 4.8+ | 图像处理 |
| NumPy | 1.26+ | |
| Pillow | 10.x | 图像读取 |
| tqdm | 4.x | 进度条 |

可选的模型转换工具（仅导出时需要）：

| 组件 | 用途 |
|------|------|
| onnx | PyTorch → ONNX |
| onnx-simplifier | ONNX 模型简化 |
| onnx2tf | ONNX → TensorFlow SavedModel → TFLite |
| tensorflow | TFLite 转换后端 |

### 树莓派 5 部署环境

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.9 | |
| OpenCV | 4.8 | 含 `opencv-contrib` (FaceDetectorYN) |
| NumPy | 1.26 | |
| tflite-runtime | 2.14 | 轻量 TFLite 推理（非完整 TF） |

> 📦 树莓派上推荐使用预配置的 conda 环境：`environment/fer_pi/environment.yml`

---

## 安装指南

### 1. 克隆项目

```bash
git clone https://github.com/aminaqvq/fer-pi5.git
cd fer-pi5
```

### 2. 训练 PC 环境

```bash
# 创建虚拟环境
conda create -n fer_train python=3.9 -y
conda activate fer_train

# 安装核心依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy pillow tqdm

# 模型导出额外依赖（可选）
pip install onnx onnx-simplifier onnx2tf tensorflow
```

### 3. 树莓派 5 环境

```bash
# 使用预配置环境文件
conda env create -f environment/fer_pi/environment.yml
conda activate fer_pi

# 或手动安装
pip install tflite-runtime==2.14
pip install opencv-python numpy
```

---

## 数据准备

### 数据集获取

本项目使用两个公开数据集：

| 数据集 | 规模 | 格式 | 来源 |
|--------|------|------|------|
| **FER2013Plus** | ~35,000 张 | 48×48 灰度 | ICML 2013 Challenge 重标注版 |
| **MMAFEDB** | ~310,000 张 | 彩色 | 大规模实验室表情数据 |

### 目录结构

下载后按以下结构放置：

```
data/
├── csv/
│   ├── train.csv          # 训练集索引
│   ├── val.csv            # 验证集索引
│   └── test.csv           # 测试集索引
├── fer2013plus/
│   ├── anger/             # fer0032555.png ...
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── MMAFEDB/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

### CSV 格式

每行一张图片，包含相对路径和标签：

```csv
path,label
fer2013plus/anger/fer0000001.png,anger
MMAFEDB/angry/1000Exp0angry_expression_125.jpg,anger
```

> 标签别名自动映射（`angry→anger`、`happiness→happy` 等），见 `dataset.py` 中的 `LABEL_ALIASES`。

### 数据清洗

```bash
cd src/training
python scan_and_clean_fer_csv.py    # 扫描并标记噪声样本
python review_label_issues_gui.py   # 可视化审核
python apply_manual_review.py       # 应用审核结果
```

---

## 训练流程

### 三阶段训练概述

```
Stage 1          Stage 2                     Stage 3
   │                │                           │
   ▼                ▼                           ▼
有标签数据 ──→ 预训练模型 ──→ 生成伪标签 ──→ 重新生成伪标签
   │                │                           │
   ▼                ▼                           ▼
ImageNet        混合训练                   低学习率精调
预训练权重      (标签+伪标签)              保守伪标签权重
   │                │                           │
   ▼                ▼                           ▼
best_stage1     best_stage2                best_stage3
   .pth            .pth                       .pth
```

### Stage 1 — 有监督预训练

```bash
cd src/training
python train_stage1.py
```

**默认配置**（可在 `train_stage1.py` 中修改）：
- 模型：MobileNetV3-Large (ImageNet 预训练)
- Epochs：200（early stopping patience=20）
- 学习率：3e-4 → cosine annealing → 1e-6
- Batch size：98
- 优化器：AdamW（weight decay=1e-4）
- 数据增强：RandAugment + 随机擦除
- Label smoothing：0.05

### Stage 2 — 伪标签生成与混合训练

```bash
# 第一步：用 Stage1 最优模型生成伪标签
python generate_pseudo_stage1_clean.py

# 第二步：混合有标签 + 伪标签数据训练
python train_stage2_balanced_clean.py
```

**Stage2 核心策略**：
- 初始化：加载 `best_model_stage1.pth`
- 伪标签置信度阈值：**0.60**
- 平衡批次采样：每类每 batch 取 18 张（7×18=126）
- 伪标签权重：置信度² 加权 + ramp-up（5 epoch 渐进）
- 学习率：5e-5

### Stage 3 — 精调收敛

```bash
# 第一步：用 Stage2 最优模型重新生成质量更高的伪标签
python generate_pseudo_stage2_final.py

# 第二步：保守精调
python train_stage3_final.py
```

**Stage3 核心策略**：
- 初始化：加载 `best_model_stage2.pth`
- 伪标签 loss 权重：仅 **0.20**（更保守）
- Ramp-up：**20 epoch**（更慢引入伪标签）
- 学习率：5e-5
- **回退机制**：Stage3 效果不如 Stage2 时自动使用 Stage2 模型

### 训练监控

训练日志输出到 `runs/training/`，每个 epoch 记录：
- 训练 loss / accuracy / Macro-F1
- 验证 loss / accuracy / Macro-F1
- 学习率、伪标签权重系数
- 每个类别的 Precision / Recall / F1

---

## 模型评估

```bash
cd src/training
python evaluate.py \
    --checkpoint checkpoints/best_model_stage2.pth \
    --model-variant large \
    --splits both \
    --batch-size 128
```

**输出**：
- 验证集和测试集的 loss / accuracy / Macro-F1 / 各类别 F1
- 混淆矩阵 PNG 图片
- `metrics_summary.json` 详细指标
- `analysis_log.csv` 累计评估记录

**TTA**（测试时增强）默认开启水平翻转，可将 Macro-F1 提升约 0.5–1%。

---

## 模型导出

将 PyTorch 模型导出为树莓派可用的 TFLite FP16 格式：

```bash
cd src/training
python export_final_model.py
```

**导出链路**：

```
PyTorch (.pth)  ──→  ONNX  ──→  ONNX simplified  ──→  TFLite FP16
                   torch.     onnx-simplifier        onnx2tf
                   onnx.
                   export
```

**每步校验**：
- PyTorch → ONNX：onnxruntime 对比，误差 < 1e-6
- ONNX → TFLite：1000 张测试图片对比，余弦相似度 > 0.999，Top-1 一致率 > 99.5%

**输出**：

```
export/final_mobilenetv3_large/
├── fer_mobilenetv3_large_fp32.onnx
├── fer_mobilenetv3_large_simplified.onnx
├── saved_model/
└── fer_mobilenetv3_large_fp16.tflite    ← 部署用
```

---

## 树莓派部署

### 文件准备

将以下文件拷贝到树莓派 5：

```
~/fer-pi5/
├── src/deploy/
│   ├── infer_pi.py
│   └── face_detection_yunet_2023mar.onnx
└── export/
    └── fer_mobilenetv3_large_fp16.tflite
```

### 启动实时推理

```bash
# 基础用法（使用默认摄像头 /dev/video0）
python src/deploy/infer_pi.py

# USB 摄像头 + 指定模型路径
python src/deploy/infer_pi.py \
    --camera 0 \
    --model export/fer_mobilenetv3_large_fp16.tflite \
    --yunet src/deploy/face_detection_yunet_2023mar.onnx

# 视频文件推理
python src/deploy/infer_video.py \
    --input test_video.mp4 \
    --output result.mp4
```

### 运行时行为

| 特性 | 默认值 | 说明 |
|------|--------|------|
| 帧率 | 15–20 FPS | 含检测+推理+显示 |
| 人脸检测 | ~15 ms | YuNet ONNX, 每 3 帧一次 |
| 表情推理 | ~45 ms | TFLite FP16, 每 2 帧一次 |
| 最大推理人脸 | 2 | 超出按置信度截断 |
| 内存占用 | ~180 MB | tflite-runtime 模式 |

### 环境变量覆盖

所有路径和关键参数支持环境变量覆盖：

```bash
export FER_PROJECT_ROOT=/home/pi/fer-pi5
export FER_TFLITE_PATH=/path/to/model.tflite
export FER_YUNET_PATH=/path/to/yunet.onnx
export FER_SAVE_DIR=/path/to/saved_images
export FER_ENABLE_CALIBRATION=1
```

### 关闭保存功能（纯显示）

```bash
python src/deploy/infer_pi.py --no-save
```

### 显示原始模型概率（调试用）

```bash
python src/deploy/infer_pi.py --show-raw-top3
```

---

## 配置参考

### 训练核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_variant` | `large` | MobileNetV3 版本（small/large/efficientnet_b0） |
| `epochs` | 200 | 最大训练轮数 |
| `batch_size` | 128 | 批次大小 |
| `lr` | 5e-4 | 初始学习率 |
| `lr_floor` | 1e-6 | 学习率下限（余弦退火终点） |
| `warmup_epochs` | 2 | 学习率预热轮数 |
| `weight_decay` | 1e-4 | AdamW 权重衰减 |
| `label_smoothing` | 0.04 | 标签平滑系数 |
| `use_amp` | true | 自动混合精度训练 |
| `grad_clip` | true | 梯度裁剪（max_norm=1.0） |
| `early_stop_patience` | 20 | 早停耐心轮数 |
| `best_metric` | `global_macro_f1` | 最优模型评判指标 |

### 伪标签参数

| 参数 | Stage2 | Stage3 | 说明 |
|------|--------|--------|------|
| `pseudo_conf_min` | 0.0 | 0.0 | 伪标签置信度最低阈值 |
| `pseudo_conf_power` | 2.0 | 2.0 | 置信度加权指数 |
| `pseudo_loss_scale` | 1.0 | 0.20 | 伪标签 loss 整体缩放 |
| `pseudo_rampup_epochs` | 5 | 20 | 伪标签渐进引入轮数 |
| `require_pseudo_conf` | true | true | 是否要求伪标签 CSV 含置信度列 |

### 类别平衡参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_class_weights` | true | 是否启用类别加权 loss |
| `class_balance_beta` | 0.995 | CB-Loss 平滑系数 |
| `sampling_strategy` | `balanced_batch` | 平衡批次采样 |
| `balanced_samples_per_class_per_batch` | 18 | 每类每 batch 样本数 |

---

## 实验结果

### 最终模型性能（测试集）

| 指标 | 数值 |
|------|------|
| **Macro-F1** | **0.68–0.70** |
| Accuracy | 70–72% |
| Happy F1 | ~0.85 |
| Surprise F1 | ~0.78 |
| Neutral F1 | ~0.74 |
| Anger F1 | ~0.66 |
| Sad F1 | ~0.64 |
| Fear F1 | ~0.61 |
| Disgust F1 | ~0.50 |

### 消融实验（EfficientNet-B0 基线）

| 实验条件 | Macro-F1 | Δ |
|----------|----------|----|
| Baseline（仅有标签） | 0.645 | — |
| + Stage2 伪标签 | 0.683 | +3.8 |
| + Stage3 精调 | 0.692 | +0.9 |
| 无类别平衡 | 0.655 | — |
| 仅类别加权 | 0.663 | +0.8 |
| 仅平衡采样 | 0.678 | +2.3 |
| 平衡采样 + 加权 | 0.683 | +2.8 |

### 树莓派 5 推理性能

| 模块 | 耗时 |
|------|------|
| YuNet 人脸检测 | 15–20 ms |
| 图像预处理 | 2–3 ms |
| TFLite FP16 推理 | 45–55 ms |
| 后处理（校准+平滑） | <1 ms |
| **端到端延迟** | **~65 ms** |
| **系统帧率** | **15–20 FPS** |

---

## 常见问题

<details>
<summary><b>为什么用 MobileNetV3-Large？</b></summary>

精度接近 ResNet-50（Macro-F1 差距 ~2%）但参数量仅 ~5.4M（ResNet-50 为 ~25M）。深度可分离卷积 + SE 注意力 + h-swish 三大技术在 TFLite 上有成熟的算子融合优化，在树莓派 5 上 FP16 推理仅需 ~50ms。
</details>

<details>
<summary><b>为什么从 PyTorch 转到 TFLite？</b></summary>

PyTorch 在 ARM CPU 上推理约 120ms+，TFLite FP16 约 50ms，速度提升超过一倍。TFLite 的算子融合和内存布局专门针对移动 ARM CPU 优化。tflite-runtime 仅 ~30MB，比安装完整 PyTorch（~800MB）轻量得多。
</details>

<details>
<summary><b>伪标签会不会"教坏"模型？</b></summary>

存在确认偏差风险。我的防护：① 置信度阈值过滤（<0.60 丢弃）；② 置信度平方加权；③ Ramp-up 渐进引入；④ 三阶段每阶段用更强模型重新打标签（"逐步精炼"而非"垃圾进垃圾出"）。实测伪标签错误率 < 8%。
</details>

<details>
<summary><b>能处理多人脸吗？</b></summary>

可以。但为保障实时性，最多同时推理 2 张脸。超出按检测置信度截断。单脸 18–20 FPS，两张脸 10–12 FPS。
</details>

<details>
<summary><b>FP16 量化精度损失多少？</b></summary>

几乎无损：1000 张测试图片对比，Top-1 一致率 > 99.5%，余弦相似度 > 0.999，Macro-F1 差异 < 0.3%。
</details>

<details>
<summary><b>支持哪些模型架构？</b></summary>

通过 `model_variant` 参数切换：`small` / `large`（MobileNetV3）、`efficientnet_b0`（EfficientNet-B0）、`repvggplus-l2pse`（RepVGGplus-L2pse，从头实现，含 SE 模块和 deploy 模式）。
</details>

---

## 许可

本项目用于学术研究与毕业设计。代码采用 MIT 许可。

## 致谢

- **MobileNetV3**：[Howard et al., ICCV 2019](https://arxiv.org/abs/1905.02244)
- **YuNet**：OpenCV Contrib 人脸检测模块
- **FER2013Plus**：[Barsoum et al., 2019](https://github.com/microsoft/FERPlus)
- **MMAFEDB**：大规模多人种表情数据集
- **树莓派 5**：[Raspberry Pi Foundation](https://www.raspberrypi.com/)
