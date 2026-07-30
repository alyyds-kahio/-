import os
import torch


# ======================
# 设备
# ======================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# 路径
# ======================

CHECKPOINT_DIR = "./checkpoints"

MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_score_model.pth")

INPUT_DIR = "./data/test/DAPI"

OUTPUT_DIR = "./results"
