# -*- coding: utf-8 -*-
"""
朝向鲁棒性评估：对同一批图像施加多种 D4 变换，对比不同模型在每种变换下的分数。

用途：
- 判断"增强/模型"是否学到了朝向无关性（旋转/翻转后分数不掉）。
- 可复用：更换 --model / --ckpts 即可评估任意两个训练后的模型。

用法示例：
    # 对比 基准(无增强) vs D4(增强)，各取一个 checkpoint，测全部 8 种朝向
    python eval_orientation.py --model unet_skip --ckpts checkpoints/best_e007_s0.6564_20260817_141507.pth
  checkpoints/best_e007_s0.6583_20260815_143235.pth --names D4 baseline --transforms all --n 100

    # 只测旋转（key 0-3）
    python eval_orientation.py --model unet_skip --ckpts a.pth b.pth \
        --transforms 0,1,2,3 --n 100

输出：每模型 × 每变换的 SSIM / PSNR / Score，以及相对 identity 的下降量（越小越抗朝向）。
结果同时打印并保存到 experiments/eval_orientation_result_<时间戳>.txt
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (DEVICE, TRAIN_DAPI_DIR, TRAIN_TARGET_DIR,
                        BATCH_SIZE, VAL_RATIO, SEED, NUM_WORKERS, IMAGE_SIZE,
                        MODEL_NAME)
from src.data.loaders import build_train_val_loaders
from src.data.transforms import apply_dihedral
from src.models import build_model
from src.inference.predictor import load_model
from src.metrics.ssim_psnr import calculate_ssim, calculate_psnr, calculate_score

TRANSFORM_NAMES = {
    0: "identity", 1: "rot90", 2: "rot180", 3: "rot270",
    4: "hflip", 5: "vflip", 6: "mainDiag", 7: "antiDiag",
}


def parse_keys(s):
    if s == "all":
        return list(range(8))
    return [int(x) for x in s.split(",")]


def tensor_to_np(t):
    return t.squeeze(0).numpy()  # [1,H,W] -> [H,W]


def eval_model(model, samples, keys, device):
    """对 samples（(dapi[1,H,W], target[1,H,W]) 列表）逐变换评估。返回 {key: (ssim, psnr, score)}。"""
    model.eval()
    sums = {k: [0.0, 0.0, 0.0] for k in keys}
    counts = {k: 0 for k in keys}
    with torch.no_grad():
        for dapi, target in samples:
            dn = tensor_to_np(dapi)
            tn = tensor_to_np(target)
            for k in keys:
                # 先 ascontiguousarray 强制连续拷贝，避免 fliplr/rot90 负步长视图无法转 torch
                dk = torch.tensor(np.ascontiguousarray(apply_dihedral(dn, k)), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                tk = torch.tensor(np.ascontiguousarray(apply_dihedral(tn, k)), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                pred = model(dk)
                s = calculate_ssim(pred, tk)
                p = calculate_psnr(pred, tk)
                sc = calculate_score(s, p)
                sums[k][0] += s
                sums[k][1] += p
                sums[k][2] += sc
                counts[k] += 1
    return {k: [sums[k][0] / counts[k], sums[k][1] / counts[k], sums[k][2] / counts[k]] for k in keys}


def main():
    parser = argparse.ArgumentParser(description="朝向鲁棒性评估")
    parser.add_argument("--model", default=MODEL_NAME, help="模型名")
    parser.add_argument("--ckpts", nargs="+", required=True, help="多个 checkpoint 路径")
    parser.add_argument("--names", nargs="+", default=None, help="各 checkpoint 的标签")
    parser.add_argument("--transforms", default="all", help="要测的 D4 key，如 'all' 或 '0,1,2,3'")
    parser.add_argument("--n", type=int, default=100, help="取多少张 val 图（默认100）")
    args = parser.parse_args()

    keys = parse_keys(args.transforms)
    names = args.names or [os.path.basename(c) for c in args.ckpts]

    # 取验证集样本（augment=False，与训练同一划分）
    _, val_loader, _, _, _ = build_train_val_loaders(
        dapi_dir=TRAIN_DAPI_DIR, target_dir=TRAIN_TARGET_DIR,
        batch_size=BATCH_SIZE, val_ratio=VAL_RATIO, seed=SEED,
        num_workers=NUM_WORKERS, image_size=IMAGE_SIZE, augment=False)
    samples = []
    for x, y in val_loader:
        for i in range(x.size(0)):
            samples.append((x[i].cpu(), y[i].cpu()))
            if len(samples) >= args.n:
                break
        if len(samples) >= args.n:
            break
    print(f"评估样本数: {len(samples)}")

    results = {}
    for ckpt, name in zip(args.ckpts, names):
        model = build_model(args.model).to(DEVICE)
        model = load_model(model, ckpt, DEVICE)
        results[name] = eval_model(model, samples, keys, DEVICE)

    # 输出表格
    lines = []
    lines.append("=" * 78)
    lines.append(f"朝向鲁棒性评估 | model={args.model} | 样本={len(samples)} | transforms={keys}")
    lines.append("=" * 78)
    header = f"{'模型':<12} | " + " | ".join(f"{TRANSFORM_NAMES[k]:>9}" for k in keys)
    lines.append(header)
    for name, res in results.items():
        row_ssim = " | ".join(f"{res[k][0]:.4f}" for k in keys)
        lines.append(f"{name+' SSIM':<12} | " + row_ssim)
        row_score = " | ".join(f"{res[k][2]:.4f}" for k in keys)
        lines.append(f"{name+' Score':<12} | " + row_score)
    lines.append("-" * 78)

    # 相对 identity 的下降量（朝向鲁棒性指标，越小越好）
    lines.append("相对 identity 的 Score 下降量（越小越抗朝向）：")
    for name, res in results.items():
        base = res[0][2]
        drops = " ".join(f"{TRANSFORM_NAMES[k]}={res[k][2]-base:+.4f}" for k in keys if k != 0)
        lines.append(f"  {name:<12}: {drops}")
    lines.append("=" * 78)

    for line in lines:
        print(line)

    out_dir = os.path.join("experiments", "eval_orientation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eval_orientation_result_{datetime.now():%Y%m%d_%H%M%S}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
