#Virtual Staining Project - Claude Code Rules

## 1. 项目目标

本项目用于参加：

全球校园人工智能算法精英大赛
算法挑战赛：
基于虚拟染色的免疫组化图像生成


当前阶段：

初赛


任务：

输入 DAPI 荧光染色图像

生成 CD68 虚拟染色图像


即：

DAPI → CD68


---

# 2. 当前数据说明

## 数据类型

医学组织荧光染色图像。


当前初赛使用：

Colon 数据集。


训练数据：

包含：

DAPI
CD68


测试数据：

只提供：

DAPI


模型需要生成：

CD68预测结果。


---

# 3. 数据格式


训练路径：


data/train/DAPI
data/train/CD68



输入：

DAPI


目标：

CD68


图片：

256 × 256


格式：

jpg


类型：

灰度图


数据范围：

训练时归一化到：

0-1



---

# 4. 当前Baseline


模型：

U-Net


输入：

1 channel


输出：

1 channel



输入：

DAPI灰度图


输出：

CD68灰度图



---

# 5. 当前Loss


当前：

ReconstructionLoss


包含：


L1 Loss


SSIM Loss



目标：

同时保证：

像素一致性

结构相似性



---

# 6. 评价指标


比赛评价：


## SSIM

结构相似性。


越高越好。



## PSNR

像素重建质量。


越高越好。



## Competition Score


当前采用：

Score =
0.7 × SSIM
+
0.3 × Normalize(PSNR)



优化目标：

提高Score。


---

# 7. 当前代码结构



virtual_staining

├── train.py

├── predict.py

├── src

│ ├── config.py

│ ├── data

│ │ ├── dataset.py

│ │ ├── transforms.py

│ │ └── loaders.py

│ ├── models

│ │ ├── unet.py

│ │ └── registry.py

│ ├── losses

│ │ └── reconstruction.py

│ ├── metrics

│ │ └── ssim_psnr.py

│ ├── checkpoints

│ │ └── saver.py

│ ├── trainer

│ │ └── trainer.py

│ └── inference

│   └── predictor.py

├── data

├── checkpoints

└── results



---

# 8. 文件职责


## src/data/

包含：

- dataset.py: VirtualStainingDataset（数据读取 + 预处理）
- transforms.py: 图片预处理函数
- loaders.py: train/val DataLoader 构建（划分/seed/batch）


负责：

- 数据读取
- 图片预处理
- Tensor转换


修改src/data时：

必须检查：

train.py

predict.py



---


## src/models/

包含：

- unet.py: UNet 结构（当前 U-Net）
- registry.py: 模型注册表 + build_model 工厂


负责：

网络结构。


当前：

U-Net


修改src/models时：

必须检查：

train.py

predict.py

已有checkpoint是否兼容。



---


## src/losses/

包含：

- reconstruction.py: ReconstructionLoss（当前 L1 + SSIM）
- __init__.py: loss 注册表 + build_loss 工厂


负责：

训练损失函数。


当前：

L1 + SSIM



修改loss时：

必须说明：

是否影响训练稳定性。



---


## src/checkpoints/ + src/metrics/

包含：

- src/checkpoints/saver.py: checkpoint保存/加载
- src/metrics/ssim_psnr.py: SSIM/PSNR/Score计算


负责：

- checkpoint保存
- checkpoint加载
- SSIM计算
- PSNR计算
- Score计算



禁止随意修改checkpoint格式。

新增子模块：

- src/trainer/trainer.py: 训练器（训练循环 + 验证 + resume + 保存）
- src/inference/predictor.py: 推理流程



---


## train.py


负责：

- 模型训练
- 验证
- 保存模型


当前训练规则：


batch_size:

4


epoch:

20



checkpoint规则：

last_model:

用于续训。



best模型：

用于预测。



---


## predict.py


负责：

加载训练好的best模型。


输入：

test/DAPI


输出：

results



输出必须：

保持原文件名。



---

# 9. checkpoint规则


checkpoint格式：


```python
{
"epoch": epoch,

"model_state_dict":

model.state_dict(),

"optimizer_state_dict":

optimizer.state_dict(),

"best_score":

best_score
}

禁止修改字段名称。

修改模型后：

需要考虑旧checkpoint是否还能加载。

10. 代码修改规则

任何代码修改前：

必须：

分析影响文件
说明可能风险
保持已有功能

修改代码时：

必须提供完整文件。

禁止：

只提供代码片段
删除已有功能
随意修改路径
改变输入输出格式
11. 当前优化方向

优化目标：

提高初赛Score。

优先级：

第一阶段：

Baseline稳定

包括：

正确训练
正确保存checkpoint
正确预测

第二阶段：

训练优化：

学习率调整
Scheduler
数据增强
Batch策略优化

第三阶段：

模型优化：

更强UNet
Residual Block
Attention
多尺度特征

第四阶段：

损失优化：

Perceptual Loss
更合理SSIM
多尺度Loss
12. 当前阶段限制

当前只针对：

初赛 Colon CD68任务。

不要考虑：

Liver
Stomach
多目标染色
Diffusion
Transformer大模型

所有优化必须优先保证：

当前baseline可运行。

13. 日志与实验记录

## 日志

训练使用 Python logging 模块（src/utils/logger.py）。

日志文件：

logs/train_*.log

自动记录：

- 启动信息（时间、项目名、git commit、Python/PyTorch版本、CUDA/GPU）
- 配置信息（模型、参数量、尺寸、batch、epoch、lr、optimizer、scheduler、loss、数据路径、train/val数量）
- checkpoint加载/保存
- 每epoch训练与验证指标（含是否刷新best）
- 异常

修改日志时：

必须说明：

是否影响训练行为。

禁止：

- 删除或移动logging配置
- 改变训练循环顺序（日志只增不改）

## 实验记录

每次训练自动生成：

experiments/experiment_*.json

并追加到：

logs/experiments.json

记录：

- 模型名称
- 参数配置
- 时间
- checkpoint路径
- 最终score
- 最佳epoch

用途：

方便比较不同模型实验结果。