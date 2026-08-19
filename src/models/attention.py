import torch
from torch import nn


class AttentionGate(nn.Module):
    """轻量 Attention Gate（Attention U-Net 风格，简化版）。

    对 skip 特征重加权：α = sigmoid(ψ(ReLU(W_g·gate + W_x·skip)))，输出 skip·α。
    - skip / gate 尺寸相同，通道均为 in_channels（用于 Pix2Pix-UNet 的 skip concat 前）。
    """

    def __init__(self, in_channels, gate_channels):
        super().__init__()
        self.w_g = nn.Conv2d(gate_channels, in_channels, 1)
        self.w_x = nn.Conv2d(in_channels, in_channels, 1)
        self.psi = nn.Conv2d(in_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip, gate):
        g = self.w_g(gate)
        x = self.w_x(skip)
        alpha = self.sigmoid(self.psi(self.relu(g + x)))
        return skip * alpha
