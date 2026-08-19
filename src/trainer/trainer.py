import os
import time

import torch

from ..checkpoints.saver import save_checkpoint, save_best_checkpoint, load_checkpoint
from ..metrics.ssim_psnr import calculate_ssim, calculate_psnr, calculate_score
from ..utils.logger import get_logger
from .ema import EMA


class Trainer:
    """训练器：封装训练循环、验证、--resume 恢复、best/last checkpoint 保存。

    E06_BOOST 新增（均可关闭，关闭时与原行为一致）：
    - scheduler_name="cosine"：CosineAnnealingLR
    - use_ema=True：权重指数滑动平均（验证/选 best 同时评估普通与 EMA）
    - weight_decay：Adam weight decay
    - grad_clip_max_norm：梯度裁剪
    - grad_accum_steps：梯度累积（等效放大 batch）
    """

    def __init__(self, model, criterion, device, lr=1e-4, epochs=20,
                 checkpoint_dir="./checkpoints", last_model_path="./checkpoints/last_model.pth",
                 best_model_path="./checkpoints/best_score_model.pth", logger=None,
                 scheduler_name="none", use_ema=False, ema_decay=0.999,
                 weight_decay=0.0, grad_clip_max_norm=None, grad_accum_steps=1,
                 use_warm_restarts=False, warm_restarts_t0=20, warm_restarts_t_mult=2):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.logger = logger or get_logger()

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        self.scheduler_name = scheduler_name
        self.scheduler = None
        if use_warm_restarts:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=warm_restarts_t0, T_mult=warm_restarts_t_mult)
        elif scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs)

        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema = None
        if use_ema:
            self.ema = EMA(model, decay=ema_decay)

        self.grad_clip_max_norm = grad_clip_max_norm
        self.grad_accum_steps = max(1, int(grad_accum_steps or 1))

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
        # scheduler 续训：从对应 last_epoch 继续余弦曲线
        if self.scheduler is not None:
            self.scheduler.last_epoch = self.start_epoch - 1
        # EMA 从加载后的权重初始化 shadow
        if self.ema is not None:
            self.ema.load_state_dict(self.model.state_dict())
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
            total_edge = 0.0
            n_batches = 0

            self.logger.info(f"Epoch [{epoch + 1}/{self.epochs}]")

            self.optimizer.zero_grad()
            accum = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                loss, l1_loss, ssim, edge = self.criterion(pred, y)

                (loss / self.grad_accum_steps).backward()
                accum += 1
                n_batches += 1
                total_loss += loss.item()
                total_l1 += l1_loss.item()
                total_ssim += ssim.item()
                total_edge += edge.item() if torch.is_tensor(edge) else float(edge)

                if accum >= self.grad_accum_steps:
                    if self.grad_clip_max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip_max_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.ema is not None:
                        self.ema.update(self.model)
                    accum = 0

                if batch_idx % 50 == 0:
                    self.logger.info(f"  Batch [{batch_idx}/{len(train_loader)}]")
                    self.logger.info(f"  Total Loss: {loss.item():.6f}")
                    self.logger.info(f"  L1 Loss: {l1_loss.item():.6f}")
                    self.logger.info(f"  SSIM: {ssim.item():.6f}")
                    self.logger.info(f"  Edge: {edge.item() if torch.is_tensor(edge) else edge:.6f}")

            # 末尾不足一个累积周期的残差
            if accum > 0:
                if self.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.ema is not None:
                    self.ema.update(self.model)

            avg_loss = total_loss / max(n_batches, 1)
            avg_l1 = total_l1 / max(n_batches, 1)
            avg_ssim = total_ssim / max(n_batches, 1)
            avg_edge = total_edge / max(n_batches, 1)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"Train | Total Loss: {avg_loss:.6f} | L1: {avg_l1:.6f} "
                f"| SSIM: {avg_ssim:.6f} | Edge: {avg_edge:.6f} | LR: {current_lr:.2e}"
            )

            # ======================
            # Validation：普通 + EMA
            # ======================

            val_ssim, val_psnr, val_score = self._validate(val_loader)
            normal_sd = None
            ema_ssim = ema_psnr = ema_score = None
            if self.ema is not None:
                normal_sd = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                self.ema.apply(self.model)
                ema_ssim, ema_psnr, ema_score = self._validate(val_loader)
                self.model.load_state_dict(normal_sd)

            # best 选择：普通 / EMA 取较高
            best_by = "normal"
            if ema_score is not None and ema_score > val_score:
                best_by = "ema"
                best_ssim, best_psnr, best_score = ema_ssim, ema_psnr, ema_score
            else:
                best_ssim, best_psnr, best_score = val_ssim, val_psnr, val_score

            is_new_best = best_score > self.best_score
            val_metrics = (best_ssim, best_psnr, best_score)

            self.logger.info(
                f"Val   | SSIM: {val_ssim:.6f} | PSNR: {val_psnr:.6f} | Score: {val_score:.6f} "
                f"| EMA_SSIM: {ema_ssim if ema_ssim is not None else -1:.6f} "
                f"| EMA_Score: {ema_score if ema_score is not None else -1:.6f} "
                f"| best_by: {best_by} | New best: {'yes' if is_new_best else 'no'}"
            )

            if is_new_best:
                self.best_score = best_score
                self.best_epoch = epoch + 1

                if best_by == "ema" and self.ema is not None:
                    self.ema.apply(self.model)

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

                if best_by == "ema" and normal_sd is not None:
                    self.model.load_state_dict(normal_sd)

            # ======================
            # Save last checkpoint（普通权重，续训用）
            # ======================

            save_checkpoint(
                self.model, self.optimizer, epoch, self.best_score,
                self.last_model_path, (val_ssim, val_psnr, val_score),
            )
            self.logger.info(f"  Last checkpoint saved")

            # ======================
            # Scheduler step
            # ======================

            if self.scheduler is not None:
                self.scheduler.step()

            self.logger.info(f"  Epoch time: {time.time() - start:.2f}s")

        self.logger.info(f"Training Finished!")
        self.logger.info(f"Best Score: {self.best_score:.6f}")
        self.logger.info(f"Best Epoch: {self.best_epoch}")

        return self.best_score
