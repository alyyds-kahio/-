import os
import time

import torch

from ..checkpoints.saver import save_checkpoint, save_best_checkpoint, load_checkpoint
from ..metrics.ssim_psnr import calculate_ssim, calculate_psnr, calculate_score
from ..utils.logger import get_logger


class Trainer:
    """训练器：封装训练循环、验证、--resume 恢复、best/last checkpoint 保存。"""

    def __init__(
            self,
            model,
            criterion,
            device,
            lr=1e-4,
            epochs=20,
            checkpoint_dir="./checkpoints",
            last_model_path="./checkpoints/last_model.pth",
            best_model_path="./checkpoints/best_score_model.pth",
            logger=None):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.logger = logger or get_logger()

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.last_model_path = last_model_path
        self.best_model_path = best_model_path

        self.best_score = -1.0
        self.best_epoch = None
        self.start_epoch = 0

    # ======================
    # 验证
    # ======================

    def _validate(self, loader):
        self.model.eval()
        total_ssim = 0.0
        total_psnr = 0.0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)

                ssim = calculate_ssim(pred, y)
                psnr = calculate_psnr(pred, y)

                total_ssim += ssim
                total_psnr += psnr

        avg_ssim = total_ssim / len(loader)
        avg_psnr = total_psnr / len(loader)
        score = calculate_score(avg_ssim, avg_psnr)

        return avg_ssim, avg_psnr, score

    # ======================
    # Resume
    # ======================

    def _resume(self):
        if not os.path.exists(self.last_model_path):
            self.logger.error(
                f"[--resume] checkpoint not found: {self.last_model_path}\n"
                f"  Train without --resume first, or check the path."
            )
            return False

        self.start_epoch, self.best_score = load_checkpoint(
            self.last_model_path, self.model, self.optimizer, self.device
        )
        self.logger.info(f"Checkpoint loaded: {self.last_model_path}")
        self.logger.info(f"  Resume epoch: {self.start_epoch}")
        self.logger.info(f"  Previous best score: {self.best_score:.6f}")
        return True

    # ======================
    # 训练
    # ======================

    def fit(self, train_loader, val_loader, resume=False):
        if resume:
            ok = self._resume()
            if not ok:
                return None

        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()

            self.model.train()
            total_loss = 0.0
            total_l1 = 0.0
            total_ssim = 0.0

            self.logger.info(f"Epoch [{epoch + 1}/{self.epochs}]")

            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                loss, l1_loss, ssim = self.criterion(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_l1 += l1_loss.item()
                total_ssim += ssim.item()

                if batch_idx % 50 == 0:
                    self.logger.info(f"  Batch [{batch_idx}/{len(train_loader)}]")
                    self.logger.info(f"  Total Loss: {loss.item():.6f}")
                    self.logger.info(f"  L1 Loss: {l1_loss.item():.6f}")
                    self.logger.info(f"  SSIM: {ssim.item():.6f}")

            avg_loss = total_loss / len(train_loader)
            avg_l1 = total_l1 / len(train_loader)
            avg_ssim = total_ssim / len(train_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"Train | Total Loss: {avg_loss:.6f} | L1: {avg_l1:.6f} "
                f"| SSIM: {avg_ssim:.6f} | LR: {current_lr:.2e}"
            )

            # ======================
            # Validation
            # ======================

            val_ssim, val_psnr, val_score = self._validate(val_loader)
            is_new_best = val_score > self.best_score
            val_metrics = (val_ssim, val_psnr, val_score)

            self.logger.info(
                f"Val   | SSIM: {val_ssim:.6f} | PSNR: {val_psnr:.6f} "
                f"| Score: {val_score:.6f} | New best: {'yes' if is_new_best else 'no'}"
            )

            if is_new_best:
                self.best_score = val_score
                self.best_epoch = epoch + 1

                saved_path = save_best_checkpoint(
                    self.model, self.optimizer, epoch, self.best_score,
                    self.checkpoint_dir, val_metrics,
                )
                self.logger.info(f"  >>> New best! Saved: {os.path.basename(saved_path)}")

                save_checkpoint(
                    self.model, self.optimizer, epoch, self.best_score,
                    self.best_model_path, val_metrics,
                )
                self.logger.info(f"  >>> Predict copy updated (best_score_model.pth)")

            # ======================
            # Save last checkpoint
            # ======================

            save_checkpoint(
                self.model, self.optimizer, epoch, self.best_score,
                self.last_model_path, val_metrics,
            )
            self.logger.info(f"  Last checkpoint saved")

            self.logger.info(f"  Epoch time: {time.time() - start:.2f}s")

        self.logger.info(f"Training Finished!")
        self.logger.info(f"Best Score: {self.best_score:.6f}")
        self.logger.info(f"Best Epoch: {self.best_epoch}")

        return self.best_score
