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
EPOCHS = 80                 # E06_BOOST 冲分：80 轮
SSIM_WEIGHT = 0.7           # E06_BOOST：提高 SSIM 权重（原 0.5，对齐比赛评分口径）

# ======================
# 训练策略开关（E06_BOOST 冲分）
# ======================

USE_EMA = True              # Exponential Moving Average
EMA_DECAY = 0.999
WEIGHT_DECAY = 1e-4         # Adam weight decay（0=关）
GRAD_ACCUM_STEPS = 2        # 梯度累积（1=关；2 等效 batch 8）
USE_GRAD_CLIP = False       # 梯度裁剪开关
GRAD_CLIP_MAX_NORM = 1.0
USE_ROI_SPLIT = False       # train/val 按 ROI 分组划分（默认关；True 时 val 只含未见 ROI，val_ratio 用 VAL_RATIO）
