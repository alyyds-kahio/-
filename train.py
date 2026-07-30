import os
import argparse
import time

import torch

from torch.utils.data import DataLoader, random_split

from src.dataset import VirtualStainingDataset
from src.model import UNet
from src.loss import ReconstructionLoss
from src.utils import (
    save_checkpoint,
    save_best_checkpoint,
    load_checkpoint,
    calculate_ssim,
    calculate_psnr,
    calculate_score,
)

# ======================
# 设备
# ======================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# 路径常量
# ======================

CHECKPOINT_DIR = "./checkpoints"
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pth")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_score_model.pth")


# ======================
# 验证函数
# ======================

def validate(model, loader):
    model.eval()
    total_ssim = 0.0
    total_psnr = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)

            ssim = calculate_ssim(pred, y)
            psnr = calculate_psnr(pred, y)

            total_ssim += ssim
            total_psnr += psnr

    avg_ssim = total_ssim / len(loader)
    avg_psnr = total_psnr / len(loader)
    score = calculate_score(avg_ssim, avg_psnr)

    return avg_ssim, avg_psnr, score


# ======================
# 主函数
# ======================

def train():
    parser = argparse.ArgumentParser(description="Virtual Staining Training")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last_model.pth",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Virtual Staining Training")
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resume: {'yes' if args.resume else 'no'}")
    print("=" * 60)

    # ======================
    # Dataset
    # ======================

    dataset = VirtualStainingDataset(
        dapi_dir="./data/train/DAPI",
        target_dir="./data/train/CD68",
    )
    print(f"Dataset size: {len(dataset)}")

    # ======================
    # Train / Val split
    # ======================

    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    # ======================
    # Model
    # ======================

    model = UNet().to(DEVICE)
    print("Model loaded")

    # ======================
    # Loss & Optimizer
    # ======================

    criterion = ReconstructionLoss(ssim_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 20
    best_score = -1.0
    start_epoch = 0

    # ======================
    # Resume
    # ======================

    if args.resume:
        if not os.path.exists(LAST_MODEL_PATH):
            print(
                f"\n[Error] --resume specified but checkpoint not found:\n"
                f"  {LAST_MODEL_PATH}\n"
                f"  Train without --resume first, or check the path."
            )
            return

        start_epoch, best_score = load_checkpoint(
            LAST_MODEL_PATH, model, optimizer, DEVICE
        )

    # ======================
    # Training loop
    # ======================

    for epoch in range(start_epoch, epochs):
        start = time.time()

        model.train()
        total_loss = 0.0
        total_l1 = 0.0
        total_ssim = 0.0

        print(f"\nEpoch [{epoch + 1}/{epochs}]")

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            loss, l1_loss, ssim = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_l1 += l1_loss.item()
            total_ssim += ssim.item()

            if batch_idx % 50 == 0:
                print()
                print(f"  Batch [{batch_idx}/{len(train_loader)}]")
                print(f"  Total Loss: {loss.item():.6f}")
                print(f"  L1 Loss: {l1_loss.item():.6f}")
                print(f"  SSIM: {ssim.item():.6f}")

        avg_loss = total_loss / len(train_loader)
        avg_l1 = total_l1 / len(train_loader)
        avg_ssim = total_ssim / len(train_loader)

        print(f"\n{'-' * 50}")
        print(f"Train | Total Loss: {avg_loss:.6f} | L1: {avg_l1:.6f} | SSIM: {avg_ssim:.6f}")

        # ======================
        # Validation
        # ======================

        val_ssim, val_psnr, val_score = validate(model, val_loader)
        val_metrics = (val_ssim, val_psnr, val_score)

        print(f"Val   | SSIM: {val_ssim:.6f} | PSNR: {val_psnr:.6f} | Score: {val_score:.6f}")

        if val_score > best_score:
            best_score = val_score

            saved_path = save_best_checkpoint(
                model, optimizer, epoch, best_score, CHECKPOINT_DIR, val_metrics
            )
            print(f"  >>> New best! Saved: {os.path.basename(saved_path)}")

            save_checkpoint(
                model, optimizer, epoch, best_score, BEST_MODEL_PATH, val_metrics
            )
            print(f"  >>> Predict copy updated (best_score_model.pth)")

        # ======================
        # Save last checkpoint
        # ======================

        save_checkpoint(
            model, optimizer, epoch, best_score, LAST_MODEL_PATH, val_metrics
        )
        print(f"  Last checkpoint saved")

        print(f"  Epoch time: {time.time() - start:.2f}s")
        print(f"{'-' * 50}")

    print(f"\nTraining Finished!")
    print(f"Best Score: {best_score:.6f}")


if __name__ == "__main__":
    train()
