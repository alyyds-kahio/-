import torch
from torch import nn

from .attention import AttentionGate


class Pix2PixUNet(nn.Module):
    """Pix2Pix 风格 U-Net Generator（适配当前 1 通道灰度 paired 任务）。

    结构：
    - Encoder: 32→64→128→256，每个 Conv3×3(stride2)+BN+LeakyReLU(0.2)
    - Bottleneck: 256 @ 16×16（Conv3×3+BN+ReLU）
    - Decoder: 256→128→64→32，每个 ConvTranspose3×3(stride2)+BN+ReLU，concat skip
    - 输出: Conv1×1 → sigmoid → [0,1]

    与标准 Pix2Pix UNet 的最小适配：
    - 输入/输出 1 通道（原 RGB 3 通道）
    - 输出 sigmoid [0,1]（原 tanh [-1,1]）
    - 通道缩放为 32→64→128→256（原 64→128→256→512→512→512），适配 CPU 训练
    """

    def __init__(self, in_channels=1, out_channels=1, features=(32, 64, 128, 256),
                 use_attention=False):
        super().__init__()
        self.features = features
        self.use_attention = use_attention

        # Encoder（下采样，stride=2）
        self.encoders = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.encoders.append(nn.Sequential(
                nn.Conv2d(prev, f, 3, stride=2, padding=1),
                nn.BatchNorm2d(f),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            prev = f

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1], 3, padding=1),
            nn.BatchNorm2d(features[-1]),
            nn.ReLU(inplace=True),
        )

        # Decoder（镜像；上采样 + concat skip + 卷积）
        rev = list(reversed(features))
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(rev) - 1):
            # rev[i] -> rev[i+1]，输出尺寸翻倍
            self.upconvs.append(nn.ConvTranspose2d(
                rev[i], rev[i + 1], 3, stride=2, padding=1, output_padding=1))
            # 输入 = up(rev[i+1]) concat skip(rev[i+1]) = 2*rev[i+1]
            self.decoders.append(nn.Sequential(
                nn.Conv2d(rev[i + 1] * 2, rev[i + 1], 3, padding=1),
                nn.BatchNorm2d(rev[i + 1]),
                nn.ReLU(inplace=True),
            ))

        # 轻量 Attention Gate（可选，作用于 skip concat 前）
        self.attentions = nn.ModuleList()
        if use_attention:
            for i in range(len(rev) - 1):
                self.attentions.append(AttentionGate(rev[i + 1], rev[i + 1]))

        # 最后一层上采样（无 skip 可拼）
        self.last_up = nn.ConvTranspose2d(
            rev[-1], rev[-1], 3, stride=2, padding=1, output_padding=1)
        self.last_conv = nn.Sequential(
            nn.Conv2d(rev[-1], rev[-1], 3, padding=1),
            nn.BatchNorm2d(rev[-1]),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(rev[-1], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)  # e1..e4

        x = self.bottleneck(x)

        for i in range(len(self.upconvs)):
            up = self.upconvs[i](x)
            skip = skips[len(skips) - 2 - i]   # 对应 encoder 层
            if self.use_attention:
                skip = self.attentions[i](skip, up)   # gate = 上采样后的 up
            up = torch.cat([up, skip], dim=1)
            x = self.decoders[i](up)

        x = self.last_up(x)
        x = self.last_conv(x)
        x = self.output(x)

        return torch.sigmoid(x)
