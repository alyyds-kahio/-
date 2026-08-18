import torch
from torch import nn


class ResnetBlock(nn.Module):
    """ResNet block（CycleGAN 风格）：ReflectionPad + Conv3×3 + InstanceNorm + ReLU ×2 + 残差。"""

    def __init__(self, dim, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0),
            norm_layer(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0),
            norm_layer(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class Pix2PixResNet(nn.Module):
    """ResNet Generator（Pix2Pix/CycleGAN 风格），适配当前 1 通道灰度 paired 任务。

    结构：
    - 输入 1 通道 → Conv7×7+IN+ReLU（32@256）
    - 下采样 ×2（32→64@128，64→128@64）
    - ResNet blocks ×n（128@64）
    - 上采样 ×2（128→64@128，64→32@256）
    - Conv7×7 → sigmoid（→ [0,1]）

    与标准实现的差异（最小适配）：
    - 输入/输出 1 通道（原 RGB 3 通道）
    - 输出 sigmoid [0,1]（原 tanh [-1,1]）
    - 通道 base=32、n_blocks=6（适配 CPU 训练规模）

    输入 [B,1,256,256] → 输出 [B,1,256,256]，sigmoid [0,1]。
    """

    def __init__(self, in_channels=1, out_channels=1, base=32, n_blocks=6):
        super().__init__()
        norm = nn.InstanceNorm2d

        def conv7(in_c, out_c):
            return nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_c, out_c, 7),
                norm(out_c),
                nn.ReLU(inplace=True),
            )

        def down(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
                norm(out_c),
                nn.ReLU(inplace=True),
            )

        def up(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 3, stride=2, padding=1, output_padding=1),
                norm(out_c),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv7(in_channels, base),
            down(base, base * 2),
            down(base * 2, base * 4),
            *[ResnetBlock(base * 4, norm) for _ in range(n_blocks)],
            up(base * 4, base * 2),
            up(base * 2, base),
            nn.ReflectionPad2d(3),
            nn.Conv2d(base, out_channels, 7),
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))
