import torch
from torch import nn


class PatchGAN(nn.Module):
    """Conditional PatchGAN 判别器（Pix2Pix 风格）。

    输入：cat(DAPI, CD68) = [B, 2, 256, 256]
    输出：[B, 1, H, W] logits（**不做 Sigmoid**，配合 BCEWithLogitsLoss）。
    """

    def __init__(self, in_channels=2, base=64):
        super().__init__()

        def block(in_c, out_c, stride=2, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, base, stride=2, bn=False),   # 128×128
            block(base, base * 2, stride=2),                # 64×64
            block(base * 2, base * 4, stride=2),            # 32×32
            block(base * 4, base * 8, stride=1),            # 32×32
            nn.Conv2d(base * 8, 1, 4, stride=1, padding=1), # [B,1,H,W] logits
        )

    def forward(self, x):
        return self.model(x)
