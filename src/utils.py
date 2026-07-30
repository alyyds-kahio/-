import os
import torch
import numpy as np
from datetime import datetime


# ======================
# 保存 last checkpoint（可覆盖，用于续训）
# ======================

def save_checkpoint(model, optimizer, epoch, best_score, path, val_metrics=None):
    """
    保存 last checkpoint，用于续训。此文件可被覆盖。

    兼容旧格式：原 4 个字段不变，新增字段为可选项。
    """
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_score": best_score,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if val_metrics is not None:
        checkpoint["val_ssim"] = val_metrics[0]
        checkpoint["val_psnr"] = val_metrics[1]
        checkpoint["val_score"] = val_metrics[2]

    torch.save(checkpoint, path)


# ======================
# 保存 best checkpoint（时间戳命名，不覆盖）
# ======================

def save_best_checkpoint(model, optimizer, epoch, best_score, save_dir, val_metrics=None):
    """
    保存 best checkpoint，使用时间戳命名，永不覆盖。

    文件名格式: best_e{epoch:03d}_s{score:.4f}_{time}.pth
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    score_str = f"{best_score:.4f}"
    epoch_str = f"{epoch + 1:03d}"
    filename = f"best_e{epoch_str}_s{score_str}_{timestamp}.pth"
    path = os.path.join(save_dir, filename)

    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_score": best_score,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if val_metrics is not None:
        checkpoint["val_ssim"] = val_metrics[0]
        checkpoint["val_psnr"] = val_metrics[1]
        checkpoint["val_score"] = val_metrics[2]

    torch.save(checkpoint, path)
    return path


# ======================
# 加载 checkpoint
# ======================

def load_checkpoint(path, model, optimizer, device):
    """
    加载 checkpoint 用于续训。
    向后兼容旧格式 checkpoint。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Use --resume only when last_model.pth exists.\n"
            "For first-time training, run without --resume."
        )

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", -1.0)

    print(f"Loaded checkpoint: {path}")
    print(f"  Resume from epoch: {epoch + 1}")
    print(f"  Previous best score: {best_score:.6f}")
    if "timestamp" in checkpoint:
        print(f"  Saved at: {checkpoint['timestamp']}")

    return epoch + 1, best_score


# ======================
# SSIM
# ======================

def calculate_ssim(pred, target):
    mse = torch.mean((pred - target) ** 2)
    ssim = 1 - mse
    return ssim.item()


# ======================
# PSNR
# ======================

def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()


# ======================
# Competition Score
# ======================

def calculate_score(ssim, psnr):
    psnr_norm = (psnr - 10) / (40 - 10)
    psnr_norm = max(0, min(1, psnr_norm))
    score = 0.7 * ssim + 0.3 * psnr_norm
    return score
