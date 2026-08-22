import os
import time
from datetime import datetime

import torch

from ..metrics.ssim_psnr import calculate_ssim, calculate_psnr, calculate_score
from ..utils.logger import get_logger


class Pix2PixTrainer:
    """完整 Pix2Pix 训练器：Generator + Conditional PatchGAN。

    - 每 batch 先更新 Discriminator，再更新 Generator
    - D：real→1，fake.detach()→0
    - G：D(fake)→1，同时计算 L1 + SSIM；G Loss = L1 + SSIM + GAN_WEIGHT × GAN
    - Checkpoint 保存：G(model_state_dict) + D(discriminator_state_dict) + opt_G + opt_D + epoch + best
    - 独立类，不修改原 Trainer
    """

    def __init__(self, generator, discriminator, criterion, gan_loss, device,
                 lr=1e-4, epochs=100, gan_weight=0.01,
                 checkpoint_dir="./checkpoints",
                 last_model_path="./checkpoints/last_model.pth",
                 best_model_path="./checkpoints/best_score_model.pth",
                 logger=None):
        self.generator = generator
        self.discriminator = discriminator
        self.criterion = criterion
        self.gan_loss = gan_loss
        self.device = device
        self.logger = logger or get_logger()

        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        self.epochs = epochs
        self.gan_weight = gan_weight

        self.checkpoint_dir = checkpoint_dir
        self.last_model_path = last_model_path
        self.best_model_path = best_model_path

        self.best_score = -1.0
        self.best_epoch = None
        self.start_epoch = 0

    # ======================
    # 验证（只评估 Generator）
    # ======================

    def _validate(self, loader):
        self.generator.eval()
        total_ssim = 0.0
        total_psnr = 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.generator(x)
                total_ssim += calculate_ssim(pred, y)
                total_psnr += calculate_psnr(pred, y)
        avg_ssim = total_ssim / len(loader)
        avg_psnr = total_psnr / len(loader)
        return avg_ssim, avg_psnr, calculate_score(avg_ssim, avg_psnr)

    # ======================
    # Checkpoint（GAN 格式）
    # ======================

    def _make_ckpt(self, epoch, best_score, val_metrics=None):
        ckpt = {
            "model_state_dict": self.generator.state_dict(),              # 兼容 predictor
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
        }
        if val_metrics is not None:
            ckpt["val_ssim"], ckpt["val_psnr"], ckpt["val_score"] = val_metrics
        return ckpt

    def _save(self, path, epoch, best_score, val_metrics=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._make_ckpt(epoch, best_score, val_metrics), path)

    def _save_best_ts(self, epoch, best_score, val_metrics):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"best_e{epoch + 1:03d}_s{best_score:.4f}_{ts}.pth"
        self._save(os.path.join(self.checkpoint_dir, name), epoch, best_score, val_metrics)
        return name

    def _resume(self):
        if not os.path.exists(self.last_model_path):
            self.logger.error(
                f"[--resume] checkpoint not found: {self.last_model_path}\n"
                f"  Train without --resume first, or check the path."
            )
            return False
        ckpt = torch.load(self.last_model_path, map_location=self.device)
        self.generator.load_state_dict(ckpt["model_state_dict"])
        self.discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        self.optimizer_G.load_state_dict(ckpt["optimizer_G_state_dict"])
        self.optimizer_D.load_state_dict(ckpt["optimizer_D_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_score = ckpt.get("best_score", -1.0)
        self.logger.info(f"Checkpoint loaded: {self.last_model_path}")
        self.logger.info(f"  Resume epoch: {self.start_epoch}")
        self.logger.info(f"  Previous best score: {self.best_score:.6f}")
        return True

    # ======================
    # 训练
    # ======================

    def fit(self, train_loader, val_loader, resume=False):
        if resume and not self._resume():
            return None

        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            self.generator.train()
            self.discriminator.train()

            total_G = 0.0
            total_gan = 0.0
            total_l1 = 0.0
            total_ssim = 0.0
            n_batches = 0

            self.logger.info(f"Epoch [{epoch + 1}/{self.epochs}]")

            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                # ---- 更新 Discriminator ----
                self.optimizer_D.zero_grad()
                fake = self.generator(x).detach()
                d_real = self.gan_loss(self.discriminator(torch.cat([x, y], dim=1)), True)
                d_fake = self.gan_loss(self.discriminator(torch.cat([x, fake], dim=1)), False)
                d_loss = (d_real + d_fake) / 2
                d_loss.backward()
                self.optimizer_D.step()

                # ---- 更新 Generator ----
                self.optimizer_G.zero_grad()
                fake = self.generator(x)
                g_gan = self.gan_loss(self.discriminator(torch.cat([x, fake], dim=1)), True)
                recon, l1_loss, ssim, edge = self.criterion(fake, y)
                g_loss = recon + self.gan_weight * g_gan
                g_loss.backward()
                self.optimizer_G.step()

                total_G += g_loss.item()
                total_gan += g_gan.item()
                total_l1 += l1_loss.item()
                total_ssim += ssim.item()
                n_batches += 1

                if batch_idx % 50 == 0:
                    self.logger.info(f"  Batch [{batch_idx}/{len(train_loader)}]")
                    self.logger.info(f"  G Loss: {g_loss.item():.6f} | GAN: {g_gan.item():.6f} | L1: {l1_loss.item():.6f} | SSIM: {ssim.item():.6f}")

            val_ssim, val_psnr, val_score = self._validate(val_loader)
            is_new_best = val_score > self.best_score

            self.logger.info(
                f"Val   | SSIM: {val_ssim:.6f} | PSNR: {val_psnr:.6f} | Score: {val_score:.6f} "
                f"| New best: {'yes' if is_new_best else 'no'}"
            )

            if is_new_best:
                self.best_score = val_score
                self.best_epoch = epoch + 1
                ts_name = self._save_best_ts(epoch, self.best_score, (val_ssim, val_psnr, val_score))
                self.logger.info(f"  >>> New best! Saved: {ts_name}")
                self._save(self.best_model_path, epoch, self.best_score, (val_ssim, val_psnr, val_score))
                self.logger.info(f"  >>> Predict copy updated (best_score_model.pth)")

            self._save(self.last_model_path, epoch, self.best_score, (val_ssim, val_psnr, val_score))
            self.logger.info(f"  Last checkpoint saved")
            self.logger.info(f"  Epoch time: {time.time() - start:.2f}s")

        self.logger.info(f"Training Finished!")
        self.logger.info(f"Best Score: {self.best_score:.6f}")
        self.logger.info(f"Best Epoch: {self.best_epoch}")
        return self.best_score
