import torch
from torch import nn


class ResBlock(nn.Module):
    """Residual block：Conv3×3+BN+ReLU → Conv3×3+BN + 残差（shortcut 通道不匹配时用 1×1）。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResUNet(nn.Module):
    """ResU-Net：Residual block + U-Net Encoder/Decoder + Skip Connection。

    与 unet_skip4s 的区别（关键差异）：
    - 用 ResBlock（Conv+BN+ReLU ×2 + 残差），而非 DoubleConv（Conv+ReLU ×2，无 BN）
    - 有 BatchNorm、残差连接
    其余拓扑（4 层、32→64→128→256、MaxPool 下采样、skip concat）与 unet_skip4s 相同。

    输入 [B,1,256,256] → 输出 [B,1,256,256]，sigmoid [0,1]。
    """

    def __init__(self, features=(32, 64, 128, 256)):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.encs = nn.ModuleList()
        prev = 1
        for f in features:
            self.encs.append(ResBlock(prev, f))
            prev = f

        rev = list(reversed(features))
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for i in range(len(rev) - 1):
            self.ups.append(nn.ConvTranspose2d(rev[i], rev[i + 1], 2, stride=2))
            self.decs.append(ResBlock(rev[i + 1] + rev[i], rev[i + 1]))

        self.last_up = nn.ConvTranspose2d(rev[-1], rev[-1], 2, stride=2)
        self.last_dec = ResBlock(rev[-1], rev[-1])
        self.output = nn.Conv2d(rev[-1], 1, 1)

    def forward(self, x):
        skips = []
        for enc in self.encs:
            x = enc(x)
            skips.append(x)          # e1@256, e2@128, e3@64, e4@32
            x = self.pool(x)
        # bottleneck = features[-1] @ H/16

        for i in range(len(self.ups)):
            x = self.ups[i](x)
            skip = skips[len(skips) - 1 - i]   # e4, e3, e2
            x = torch.cat([x, skip], dim=1)
            x = self.decs[i](x)

        x = self.last_up(x)
        x = self.last_dec(x)
        out = self.output(x)
        return torch.sigmoid(out)
