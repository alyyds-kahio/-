import torch
from torch import nn


class DoubleConv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.ReLU()

        )

    def forward(self, x):

        return self.block(x)


class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder1 = DoubleConv(
            1,
            64
        )

        self.pool = nn.MaxPool2d(2)

        self.encoder2 = DoubleConv(
            64,
            128
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                128,
                64,
                2,
                stride=2
            ),

            DoubleConv(
                64,
                64
            )

        )

        self.output = nn.Conv2d(
            64,
            1,
            1
        )

    def forward(self, x):

        x1 = self.encoder1(x)

        x2 = self.pool(x1)

        x2 = self.encoder2(x2)

        x3 = self.decoder(x2)

        out = self.output(x3)

        return torch.sigmoid(out)


class UNetSkip(nn.Module):
    """带 skip connection 的 U-Net（2 层）。

    与 UNet 的区别：decoder 上采样后与对应 encoder 特征拼接，
    保留细粒度结构信息，通常可显著提升 SSIM。
    """

    def __init__(self):

        super().__init__()

        self.encoder1 = DoubleConv(
            1,
            64
        )

        self.pool = nn.MaxPool2d(2)

        self.encoder2 = DoubleConv(
            64,
            128
        )

        self.up = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        # 输入 = up(64) concat skip(64) = 128
        self.decoder = DoubleConv(
            128,
            64
        )

        self.output = nn.Conv2d(
            64,
            1,
            1
        )

    def forward(self, x):

        x1 = self.encoder1(x)          # [64, H, W]  -> skip
        x2 = self.pool(x1)
        x2 = self.encoder2(x2)         # [128, H/2, W/2]

        d = self.up(x2)                # [64, H, W]
        d = torch.cat([d, x1], dim=1)  # [128, H, W]

        d = self.decoder(d)            # [64, H, W]
        out = self.output(d)

        return torch.sigmoid(out)


class UNetSkip4(nn.Module):
    """4 层带 skip connection 的 U-Net。

    encoder: 1→64→128→256→512（bottleneck 32×32）
    decoder: 512→256→128→64，每层上采样后与对应 encoder 特征 concat。
    相比 2 层版容量/感受野更大，SSIM 上限更高。
    """

    def __init__(self):

        super().__init__()

        self.enc1 = DoubleConv(1, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)   # cat(up4 256, skip e3 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)   # cat(up3 128, skip e2 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)    # cat(up2 64, skip e1 64)

        self.output = nn.Conv2d(64, 1, 1)

    def forward(self, x):

        e1 = self.enc1(x)            # [64, H, W]
        e2 = self.enc2(self.pool(e1))  # [128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [256, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # [512, H/8, W/8]

        d = self.up4(e4)             # [256, H/4, W/4]
        d = torch.cat([d, e3], dim=1)
        d = self.dec3(d)             # [256, H/4, W/4]

        d = self.up3(d)              # [128, H/2, W/2]
        d = torch.cat([d, e2], dim=1)
        d = self.dec2(d)             # [128, H/2, W/2]

        d = self.up2(d)              # [64, H, W]
        d = torch.cat([d, e1], dim=1)
        d = self.dec1(d)             # [64, H, W]

        out = self.output(d)         # [1, H, W]
        return torch.sigmoid(out)
