# -*- coding: utf-8 -*-
"""
E01: 旧模型真实指标体检（只读，不改任何代码/checkpoint）

在验证集上评估现有 best_score_model.pth，用**真实 SSIM**（skimage 参照实现）
算真实成绩，并对比当前代码报告的 fake SSIM（1-MSE）。

用法：
    python eval.py                 # 默认评估 checkpoints/best_score_model.pth
    python eval.py --ckpt <路径>    # 指定其他 checkpoint
    python eval.py --split train   # 评估训练集（可选）

输出同时打印到终端并写入 experiments/E01_eval/eval_result.txt
"""
import argparse
import os
import sys

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_ref

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    DEVICE, TRAIN_DAPI_DIR, TRAIN_TARGET_DIR,
    BATCH_SIZE, VAL_RATIO, SEED, NUM_WORKERS, IMAGE_SIZE, MODEL_NAME,
    BEST_MODEL_PATH,
)
from src.data.loaders import build_train_val_loaders
from src.models import build_model
from src.inference.predictor import load_model
from src.metrics.ssim_psnr import calculate_score, calculate_ssim


def real_ssim(pred_np, target_np):
    """真实 SSIM（skimage, gaussian 窗, data_range=1.0）。pred/target 均为 2D 灰度 [0,1]。"""
    return ssim_ref(target_np, pred_np, data_range=1.0)


def psnr_from_mse(mse):
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def evaluate(model, loader, device):
    model.eval()
    n = 0
    real_ssim_sum = 0.0
    fake_ssim_sum = 0.0      # 旧代码口径：1 - MSE（逐图像，便于对比）
    fake_ssim_batch_sum = 0.0  # 旧代码口径：按 batch 的 1-MSE（复现旧 trainer 日志）
    torch_ssim_batch_sum = 0.0  # 新指标：trainer 实际使用的 batch 级真实 SSIM
    psnr_sum = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            # 旧代码的 batch 级 fake SSIM（1-MSE）
            mse_b = torch.mean((pred - y) ** 2).item()
            fake_ssim_batch_sum += (1 - mse_b)

            # 新指标：trainer 实际调用的 batch 级真实 SSIM
            torch_ssim_batch_sum += calculate_ssim(pred, y)

            for i in range(x.size(0)):
                p = pred[i, 0].cpu().numpy()
                t = y[i, 0].cpu().numpy()
                real_ssim_sum += real_ssim(p, t)
                mse = float(np.mean((p - t) ** 2))
                fake_ssim_sum += (1 - mse)
                psnr_sum += psnr_from_mse(mse)
                n += 1

    n_batch = len(loader)
    avg_real_ssim = real_ssim_sum / n
    avg_fake_ssim = fake_ssim_sum / n
    avg_fake_ssim_batch = fake_ssim_batch_sum / n_batch
    avg_torch_ssim_batch = torch_ssim_batch_sum / n_batch
    avg_psnr = psnr_sum / n

    # Score：真实 SSIM 口径 vs 旧口径
    score_real = calculate_score(avg_real_ssim, avg_psnr)
    score_fake = calculate_score(avg_fake_ssim, avg_psnr)

    return {
        "n_images": n,
        "real_ssim": avg_real_ssim,
        "torch_ssim_batch(trainer新指标)": avg_torch_ssim_batch,
        "fake_ssim_per_image": avg_fake_ssim,
        "fake_ssim_batch(旧口径)": avg_fake_ssim_batch,
        "psnr": avg_psnr,
        "score_real(真实SSIM)": score_real,
        "score_fake(旧口径)": score_fake,
    }


def main():
    parser = argparse.ArgumentParser(description="E01 真实指标评估")
    parser.add_argument("--ckpt", default=BEST_MODEL_PATH, help="checkpoint 路径")
    parser.add_argument("--split", default="val", choices=["val", "train"],
                        help="评估集：val(默认) 或 train")
    args = parser.parse_args()

    train_loader, val_loader, n_train, n_val, n_total = build_train_val_loaders(
        dapi_dir=TRAIN_DAPI_DIR, target_dir=TRAIN_TARGET_DIR,
        batch_size=BATCH_SIZE, val_ratio=VAL_RATIO, seed=SEED,
        num_workers=NUM_WORKERS, image_size=IMAGE_SIZE,
    )
    loader = val_loader if args.split == "val" else train_loader
    n_loader = n_val if args.split == "val" else n_train

    model = build_model(MODEL_NAME).to(DEVICE)
    model = load_model(model, args.ckpt, DEVICE)

    print("=" * 60)
    print(f"E01 评估: {args.ckpt}  |  {args.split} 集 ({n_loader} 张)")
    print("=" * 60)

    metrics = evaluate(model, loader, DEVICE)

    lines = []
    lines.append("=" * 60)
    lines.append(f"E01 评估结果: {args.ckpt}")
    lines.append(f"数据集: {args.split} ({metrics['n_images']} 张, 与训练同一划分 seed=42)")
    lines.append("=" * 60)
    lines.append(f"真实 SSIM (skimage 逐图):      {metrics['real_ssim']:.4f}")
    lines.append(f"trainer新指标 SSIM (torch batch): {metrics['torch_ssim_batch(trainer新指标)']:.4f}")
    lines.append(f"旧口径 fake SSIM (1-MSE):       {metrics['fake_ssim_per_image']:.4f}")
    lines.append(f"旧口径 fake SSIM (batch):       {metrics['fake_ssim_batch(旧口径)']:.4f}")
    lines.append(f"PSNR:                           {metrics['psnr']:.4f}")
    lines.append("-" * 60)
    lines.append(f"Score(真实SSIM口径):            {metrics['score_real(真实SSIM)']:.4f}")
    lines.append(f"Score(旧口径):                  {metrics['score_fake(旧口径)']:.4f}")
    lines.append("-" * 60)
    lines.append(f"口径差距 (real - fake):         {metrics['real_ssim'] - metrics['fake_ssim_per_image']:+.4f}")
    lines.append(f"新指标 vs skimage 差:           {metrics['real_ssim'] - metrics['torch_ssim_batch(trainer新指标)']:+.4f}")
    lines.append("=" * 60)

    for line in lines:
        print(line)

    out_dir = os.path.join("experiments", "E01_eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_result.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
