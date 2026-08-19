import torch
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)


def sobel_edges(x):
    """输入 [B,C,H,W]（0-1），返回边缘幅值图 [B,C,H,W]（可微）。"""
    pad = 1
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    gx = F.conv2d(xp, _SOBEL_X.to(x.device), padding=0)
    gy = F.conv2d(xp, _SOBEL_Y.to(x.device), padding=0)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def edge_loss(pred, target):
    """pred 与 target 的 Sobel 边缘幅值图 L1。返回标量 tensor。"""
    return F.l1_loss(sobel_edges(pred), sobel_edges(target))
