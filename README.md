# Virtual Staining: DAPI to CD68

基于深度学习的免疫组化图像虚拟染色任务。

当前版本针对初赛 Colon 数据集，使用 DAPI 荧光图像作为输入，生成 CD68 虚拟染色图像。


---

# 1. 项目简介

虚拟染色任务旨在通过深度学习模型学习不同染色标记之间的映射关系。

本项目当前实现：
DAPI 图像
|
|
↓
简化版 U-Net
|
|
↓
CD68 虚拟染色图像
复制
输入：

- DAPI 单通道灰度荧光图像


输出：

- CD68 灰度染色图像


当前目标：

提高生成图像与真实 CD68 图像之间的结构和像素一致性。


---

# 2. 当前版本说明

## Baseline Version

当前模型：
Simplified U-Net
复制
特点：

- Encoder-Decoder结构
- 单通道输入
- 单通道输出
- 输出使用 Sigmoid 映射到 0~1


当前未使用：

- Skip Connection
- Attention
- Transformer
- GAN
- Diffusion


后续优化可在此基础上扩展。


---

# 3. 项目目录结构

virtual_staining/
│
├── data/
│
│ ├── train/
│ │
│ │ ├── DAPI/
│ │ │
│ │ ├── CD68/
│ │ │
│ │ ├── CD45RO/
│ │ │
│ │ ├── HLA-DR/
│ │ │
│ │ ├── HLA_DR/
│ │ │
│ │ └── Vimentin/
│ │
│ └── test/
│ │
│ └── DAPI/
│
│
├── checkpoints/
│
│ ├── best_score_model.pth
│ │
│ ├── last_model.pth
│ │
│ └── best_eXXX_sXXXX_*.pth   (历史 best，时间戳)
│
│
├── results/
│
│
├── src/
│
│ ├── config.py            # 统一配置（路径 + 超参）
│ ├── data/                # 数据处理
│ │ ├── dataset.py
│ │ ├── transforms.py
│ │ └── loaders.py
│ ├── models/              # 模型结构
│ │ ├── unet.py
│ │ └── registry.py
│ ├── losses/              # 损失函数
│ │ └── reconstruction.py
│ ├── metrics/             # 评价指标
│ │ └── ssim_psnr.py
│ ├── checkpoints/         # checkpoint 保存/加载
│ │ └── saver.py
│ ├── trainer/             # 训练器
│ │ └── trainer.py
│ └── inference/           # 推理流程
│   └── predictor.py
│
│
├── train.py
│
├── predict.py
│
└── requirements.txt
复制

---

# 4. 数据说明


## 当前训练数据

当前版本只使用：
data/train/
复制
中的：
DAPI
复制
作为输入：
CD68
复制
作为监督目标。


数据对应关系：

例如：
data/train/DAPI/
ROI000_00_00.jpg
复制
对应：
data/train/CD68/
ROI000_00_00.jpg
复制

Dataset 根据文件名进行匹配。


---

# 5. 数据预处理


当前处理流程：

## 读取

使用 OpenCV：

```python
cv2.imread(
    path,
    cv2.IMREAD_GRAYSCALE
)
读取单通道灰度图。
￼
Resize
图像统一调整到模型输入尺寸。
￼
Normalize
像素归一化：
复制
原始:

0 ~ 255


↓

模型输入:

0 ~ 1
￼
Tensor格式
输入：
复制
[1,H,W]
经过 DataLoader：
复制
[B,1,H,W]
其中：
B:
batch size
￼
6. 环境安装
创建虚拟环境：
Bash
复制
￼
python -m venv .venv
激活：
Windows:
Bash
复制
￼
.venv\Scripts\activate
安装依赖：
Bash
复制
￼
pip install -r requirements.txt
￼
# 7. 训练模型

## 首次训练

```bash
python train.py
```

首次运行自动创建 `checkpoints/` 目录，训练完成后生成 `last_model.pth`（续训用）和 `best_score_model.pth`（预测用）。

## 断点续训

训练中断后，从上次保存的 `last_model.pth` 恢复：

```bash
python train.py --resume
```

恢复内容：模型权重、优化器状态（momentum 等）、epoch 进度、best_score 记录。

## 训练流程

```
DAPI + CD68 → train/val 划分 → 训练 → L1+SSIM Loss → 验证 Score → 保存 checkpoint
```
￼
8. Loss设计
当前训练Loss：
𝐿𝑜s𝑠=𝐿1+𝜆(1−𝑆𝑆𝐼𝑀)
其中：
L1 Loss
作用：
保证生成图像与真实图像像素接近。
￼
SSIM Loss
作用：
保持组织结构信息。
由于训练目标是最小化Loss：
复制
SSIM越高

↓

1-SSIM越小

↓

Loss下降
￼
9. 模型评价指标
比赛评价指标：
SSIM
Structural Similarity Index
衡量：
• 亮度
• 对比度
• 结构相似性
越高越好。
￼
PSNR
Peak Signal-to-Noise Ratio
衡量：
生成图像与真实图像的像素误差。
越高越好。
￼
综合评分
初赛评分：
复制
Score =
70% × SSIM
+
30% × Normalize(PSNR)
训练过程中：
使用验证集 Score 选择最佳模型。
保存：
复制
checkpoints/best_score_model.pth
￼
10. 模型文件说明
src/models/unet.py + registry.py
作用：
定义网络结构。
当前：
复制
Simplified U-Net
修改影响：
复制
src/models/unet.py + registry.py

↓

train.py
predict.py
原因：
训练和预测都需要实例化同一个模型。
如果修改网络结构：
需要重新训练模型。
￼
src/data/dataset.py + transforms.py + loaders.py
作用：
负责：
• 数据读取
• 图像预处理
• Tensor转换
修改影响：
复制
src/data/dataset.py + transforms.py + loaders.py

↓

train.py
predict.py
例如：
修改：
• 文件路径
• resize方式
• 数据增强
都会影响训练和预测。
￼
src/losses/reconstruction.py
作用：
定义训练优化目标。
当前：
复制
L1 + SSIM Loss
修改影响：
复制
src/losses/reconstruction.py

↓

train.py
不会影响预测过程。
￼
src/checkpoints/saver.py + src/metrics/ssim_psnr.py
作用：
工具函数：
包括：
• 模型保存（last + 时间戳 best）
• load_checkpoint 加载续训
• SSIM计算
• PSNR计算
• Score计算

save_checkpoint：
保存 last_model.pth，可覆盖，用于 --resume 续训。

save_best_checkpoint：
保存时间戳命名的 best 历史版本，永不覆盖。

load_checkpoint：
加载 checkpoint 恢复训练，向后兼容旧格式。
修改影响：
复制
src/checkpoints/saver.py + src/metrics/ssim_psnr.py

↓

train.py
￼
train.py
作用：
训练入口。
负责：
• 加载Dataset
• 创建模型
• Loss计算
• 参数更新
• 验证
• 保存模型（last + 时间戳 best）
运行：
Bash
复制
￼
python train.py

断点续训：
Bash
复制
￼
python train.py --resume
￼
predict.py
作用：
模型推理。
流程：
复制
test/DAPI

↓

加载best_score_model.pth

↓

生成CD68结果

↓

保存到results/
运行：
Bash
复制
￼
python predict.py
￼
11. 文件修改影响关系
整体关系：
复制
              src/data/
                  |
        --------------------
        |                  |
     train.py          predict.py


              src/models/
                  |
        --------------------
        |                  |
     train.py          predict.py


              src/losses/
                  |
              train.py


              src/checkpoints/ + src/metrics/
                  |
              train.py
￼
12. Checkpoint 说明

训练完成后 checkpoints/ 目录内容：

checkpoints/
├── last_model.pth
├── best_score_model.pth
├── best_e001_s0.7234_20260730_143025.pth
├── best_e003_s0.7541_20260730_150142.pth
└── best_e005_s0.7890_20260730_161203.pth

## last_model.pth

作用：
保存最新训练状态（模型权重 + 优化器状态 + epoch 进度），用于中断恢复训练。

特性：
- 可覆盖（每轮更新）
- 支持 --resume 续训
- 包含 optimizer_state_dict，恢复后可无缝继续训练

## best_score_model.pth

作用：
保存当前最高 Score 模型，用于 predict.py 预测。

特性：
- 可覆盖（刷新 best 时更新）
- predict.py 默认加载此文件
- 在任何时刻都可直接运行 predict.py 生成结果

## 历史 best 模型

文件名格式：

best_e{epoch:03d}_s{score:.4f}_{timestamp}.pth

示例：

best_e003_s0.7234_20260730_143025.pth

作用：
保存每次刷新 best 时的历史版本，永不覆盖。

优点：
- 可回溯训练过程中所有最佳时刻
- 文件名包含 epoch、score、时间，便于对比
- 不会因后续训练而丢失

## Resume 恢复内容

python train.py --resume 支持恢复：

- 模型权重 state_dict
- 优化器状态（含 momentum、learning rate 等）
- epoch 进度（从中断处继续）
- best_score 记录

## 兼容性说明

predict.py 无需修改。仍然直接加载 best_score_model.pth。

旧格式 checkpoint（仅有 4 个原始字段）与新代码完全兼容。
￼
13. 训练日志与实验记录

## 日志系统

训练时使用 Python logging 模块，同时输出到终端与日志文件：

logs/
├── train_20260804_190258.log   # 每次训练一个日志文件
└── experiments.json            # 全部实验汇总索引（追加）

日志内容按阶段记录：

- 启动阶段：时间、项目名称、git commit、Python/PyTorch 版本、CUDA/GPU 信息
- 配置阶段：模型名称、参数量、输入尺寸、batch_size、epochs、learning rate、optimizer、scheduler、loss 名称与权重、数据路径、train/val 数量
- checkpoint 阶段：加载成功/失败、加载路径、resume epoch、best_score、保存路径
- 训练阶段（每个 epoch）：epoch、train loss、l1 loss、ssim、learning rate、epoch 耗时
- 验证阶段：val ssim、val psnr、score、是否刷新 best
- 错误：任何异常通过 logger.exception 写入日志

## 实验记录

每次训练自动生成一条实验摘要：

experiments/
└── experiment_20260804_190300.json

同时追加到 logs/experiments.json 索引，方便对比不同模型实验。

每条记录包含：模型名称、参数配置、时间、checkpoint 路径、最终 score、最佳 epoch。

运行方式不变：

- python train.py
- python train.py --resume
- python predict.py

￼
14. 后续优化路线
V1 当前Baseline
已完成：
• 简化U-Net
• DAPI→CD68
• L1+SSIM Loss
• train/validation划分
• best/last 双 checkpoint 保存
• 时间戳历史 best（不覆盖）
• --resume 断点续训
• checkpoint 向后兼容
￼
V2 模型结构优化
修改：
复制
src/models/unet.py + registry.py
方向：
• 完整U-Net Skip Connection
• Attention U-Net
• UNet++
影响：
需要重新训练。
￼
V3 数据增强
修改：
复制
src/data/dataset.py + transforms.py + loaders.py
加入：
• 随机翻转
• 旋转
• Crop
• 强度增强
目的：
提升模型泛化能力。
￼
V4 Loss优化
修改：
复制
src/losses/reconstruction.py
方向：
• Edge Loss
• Perceptual Loss
• 多尺度Loss
目的：
提高：
• 结构保持
• 局部细节恢复
￼
V5 训练策略优化
修改：
复制
train.py
方向：
• 学习率调度
• AMP混合精度
• Early stopping
• 多模型融合
￼
15. 当前实验记录
Baseline V1
模型：
Simplified U-Net
输入：
DAPI
输出：
CD68
Loss：
L1 + SSIM
评价：
SSIM + PSNR + Score
Checkpoint：
last + 时间戳 best + --resume 续训
状态：
Baseline 工程化完成，准备模型优化
复制