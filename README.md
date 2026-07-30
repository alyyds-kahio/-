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
│ └── last_model.pth
│
│
├── results/
│
│
├── src/
│
│ ├── dataset.py
│ ├── model.py
│ ├── loss.py
│ └── utils.py
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
7. 训练模型
运行：
Bash
复制
￼
python train.py
训练流程：
复制
读取DAPI/CD68数据

↓

划分:

train
validation

↓

训练模型

↓

计算Loss

↓

验证Score

↓

保存最佳模型
￼
8. Loss设计
当前训练Loss：
𝐿
𝑜
𝑠
𝑠
=
𝐿
1
+
𝜆
(
1
−
𝑆
𝑆
𝐼
𝑀
)
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
model.py
作用：
定义网络结构。
当前：
复制
Simplified U-Net
修改影响：
复制
model.py

↓

train.py
predict.py
原因：
训练和预测都需要实例化同一个模型。
如果修改网络结构：
需要重新训练模型。
￼
dataset.py
作用：
负责：
• 数据读取
• 图像预处理
• Tensor转换
修改影响：
复制
dataset.py

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
loss.py
作用：
定义训练优化目标。
当前：
复制
L1 + SSIM Loss
修改影响：
复制
loss.py

↓

train.py
不会影响预测过程。
￼
utils.py
作用：
工具函数：
包括：
• 模型保存
• SSIM计算
• PSNR计算
• Score计算
修改影响：
复制
utils.py

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
• 保存模型
运行：
Bash
复制
￼
python train.py
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
              dataset.py
                  |
        --------------------
        |                  |
     train.py          predict.py


              model.py
                  |
        --------------------
        |                  |
     train.py          predict.py


              loss.py
                  |
              train.py


              utils.py
                  |
              train.py
￼
12. 当前训练输出
训练完成后：
复制
checkpoints/

├── best_score_model.pth

└── last_model.pth
其中：
best_score_model.pth
验证集综合Score最高模型。
用于最终预测。
￼
last_model.pth
最后一个epoch保存模型。
用于继续实验。
￼
13. 后续优化路线
V1 当前Baseline
已完成：
• 简化U-Net
• DAPI→CD68
• L1+SSIM Loss
• train/validation划分
• best模型保存
￼
V2 模型结构优化
修改：
复制
src/model.py
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
src/dataset.py
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
src/loss.py
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
14. 当前实验记录
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
状态：
进行中
复制