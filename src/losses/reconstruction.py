import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """
    虚拟染色训练Loss

    组合:

    L1 Loss:
        保证像素接近

    SSIM Loss:
        保证结构相似


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

    def simple_ssim(
            self,
            x,
            y
    ):
        """
        简化版SSIM

        用于训练约束

        """

        C1 = 0.01 ** 2

        C2 = 0.03 ** 2

        mu_x = torch.mean(
            x
        )

        mu_y = torch.mean(
            y
        )

        sigma_x = torch.var(
            x
        )

        sigma_y = torch.var(
            y
        )

        sigma_xy = torch.mean(
            (x - mu_x) * (y - mu_y)
        )

        ssim = (

            (2 * mu_x * mu_y + C1)
            *
            (2 * sigma_xy + C2)

        ) / (

            (mu_x ** 2 + mu_y ** 2 + C1)
            *
            (sigma_x + sigma_y + C2)

        )

        return ssim

    def forward(
            self,
            prediction,
            target
    ):
        l1_loss = self.l1(
            prediction,
            target
        )

        ssim = self.simple_ssim(
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
