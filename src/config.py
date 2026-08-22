import torch


# ======================
# 设备
# ======================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# 路径
# ======================

DATA_DIR = "./data"

TRAIN_DAPI_DIR = "./data/train/DAPI"
TRAIN_TARGET_DIR = "./data/train/CD68"
TEST_DAPI_DIR = "./data/test/DAPI"

CHECKPOINT_DIR = "./checkpoints"
LAST_MODEL_PATH = "./checkpoints/last_model.pth"
BEST_MODEL_PATH = "./checkpoints/best_score_model.pth"

# predict 默认加载的模型路径
MODEL_PATH = BEST_MODEL_PATH

# predict 输入 / 输出
INPUT_DIR = TEST_DAPI_DIR
OUTPUT_DIR = "./results"

# 日志 / 实验记录
LOG_DIR = "./logs"
EXPERIMENT_DIR = "./experiments"
PROJECT_NAME = "virtual_staining"


# ======================
# 数据
# ======================

IMAGE_SIZE = 256
VAL_RATIO = 0.1          # 验证集比例（train 占 1 - VAL_RATIO）
SEED = 42
NUM_WORKERS = 0
BATCH_SIZE = 4
AUGMENT = True           # E06_BOOST：D4 开启（梯度累积 + 长训练组合下尝试；原实验结论默认关）


# ======================
# 模型 / Loss / 训练
# ======================

MODEL_NAME = "unet"
LOSS_NAME = "reconstruction"
OPTIMIZER_NAME = "adam"
SCHEDULER_NAME = "cosine"   # "none" 或 "cosine"（CosineAnnealingLR）
LEARNING_RATE = 1e-4
EPOCHS = 100                 # E06_BOOST 冲分：80 轮
SSIM_WEIGHT = 0.7           # E06_BOOST：提高 SSIM 权重（原 0.5，对齐比赛评分口径）

# ======================
# 训练策略开关（E06_BOOST 冲分）
# ======================

USE_EMA = True              # Exponential Moving Average
EMA_DECAY = 0.85
WEIGHT_DECAY = 1e-4         # Adam weight decay（0=关）
GRAD_ACCUM_STEPS = 4        # 梯度累积（1=关；2 等效 batch 8）
USE_GRAD_CLIP = False       # 梯度裁剪开关
GRAD_CLIP_MAX_NORM = 1.0
USE_ROI_SPLIT = False       # train/val 按 ROI 分组划分（默认关；True 时 val 只含未见 ROI，val_ratio 用 VAL_RATIO）

# ======================
# E06_BOOST_PLUS 综合优化开关（默认全关，独立可开关）
# ======================

USE_WARM_RESTARTS = False    # WarmRestarts 学习率调度（True 时覆盖 SCHEDULER_NAME）
WARM_RESTARTS_T0 = 20       # WarmRestarts 首次重启周期（epoch）
WARM_RESTARTS_T_MULT = 2    # WarmRestarts 每次重启周期倍数
USE_EDGE_LOSS = True        # 边缘/结构一致性 Loss（Sobel 边缘）
EDGE_LOSS_WEIGHT = 0.7      # Edge Loss 权重
USE_ATTENTION = True        # 轻量 Attention（Attention Gate，作用于 skip connection）
ATTENTION_TYPE = "gate"     # 注意力类型（当前支持 "gate"）

# ======================
# E06-C：完整 Pix2Pix（Generator + Conditional PatchGAN）
# ======================

USE_GAN = False             # 完整 Pix2Pix：True 时用 Pix2PixTrainer（Generator + PatchGAN）
GAN_WEIGHT = 0.01           # GAN Loss 权重（G loss = L1 + SSIM + GAN_WEIGHT × GAN）
