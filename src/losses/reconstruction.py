import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """
    虚拟染色训练Loss

    组合:

    L1 Loss:
        保证像素接近

    SSIM Loss:
        保证结构相似（真实局部窗口 SSIM，与验证指标 src/metrics/ssim_psnr.calculate_ssim 口径一致）


    Loss:

    L = L1 + lambda*(1-SSIM)
    """

    def __init__(
            self,
            ssim_weight=0.5
    ):

        super().__init__()

        self.l1 = nn.L1Loss()

        self.ssim_weight = ssim_weight

    @staticmethod
    def _ssim(
            x,
            y,
            data_range=1.0,
            win_size=7
    ):
        """
        可微的局部窗口 SSIM（uniform 窗、样本协方差）。

        与 src/metrics/ssim_psnr.calculate_ssim 口径一致：
        - uniform 7×7 窗
        - 样本协方差归一化
        - 裁掉边界 strip 后取均值

        返回均值 SSIM 张量（可反向传播）。
        """
        K1, K2 = 0.01, 0.03
        C1 = (K1 * data_range) ** 2
        C2 = (K2 * data_range) ** 2

        pad = win_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        y = F.pad(y, (pad, pad, pad, pad), mode="reflect")

        channels = x.size(1)
        kernel = torch.ones(
            1, 1, win_size, win_size, device=x.device, dtype=x.dtype
        ) / (win_size ** 2)

        def conv(t):
            return F.conv2d(t, kernel, stride=1, padding=0, groups=channels)

        mu_x = conv(x)
        mu_y = conv(y)
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = conv(x * x) - mu_x_sq
        sigma_y_sq = conv(y * y) - mu_y_sq
        sigma_xy = conv(x * y) - mu_xy

        n_pix = win_size ** 2
        cov_norm = n_pix / (n_pix - 1)
        sigma_x_sq = (sigma_x_sq * cov_norm).clamp(min=0)
        sigma_y_sq = (sigma_y_sq * cov_norm).clamp(min=0)
        sigma_xy = sigma_xy * cov_norm

        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        # C1/C2 > 0 ⇒ denominator > 0，无需处理除零
        ssim_map = numerator / denominator

        if pad > 0:
            ssim_map = ssim_map[:, :, pad:-pad, pad:-pad]
        return ssim_map.mean()

    def forward(
            self,
            prediction,
            target
    ):
        l1_loss = self.l1(
            prediction,
            target
        )

        ssim = self._ssim(
            prediction,
            target
        )

        total_loss = (

                l1_loss

                +

                self.ssim_weight * (1 - ssim)

        )

        return (
            total_loss,
            l1_loss,
            ssim
        )
