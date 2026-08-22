import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """Conditional GAN 对抗损失（BCEWithLogitsLoss）。

    - 判别器：real→1，fake→0
    - 生成器：fake→1
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, real):
        """pred 为 logits；real=True 目标 1，real=False 目标 0。"""
        target = torch.ones_like(pred) if real else torch.zeros_like(pred)
        return self.bce(pred, target)
