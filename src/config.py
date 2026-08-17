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
AUGMENT = False          # D4 数据增强默认关闭（实验结论：暂不采用为默认；改回 True 可重新启用）


# ======================
# 模型 / Loss / 训练
# ======================

MODEL_NAME = "unet"
LOSS_NAME = "reconstruction"
OPTIMIZER_NAME = "adam"
SCHEDULER_NAME = "none"
LEARNING_RATE = 1e-4
EPOCHS = 20
SSIM_WEIGHT = 0.5
