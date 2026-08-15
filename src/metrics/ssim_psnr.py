import torch
import torch.nn.functional as F


# ======================
# SSIM
# ======================

def calculate_ssim(pred, target, data_range=1.0, win_size=7):
    """
    真实 SSIM（局部窗口，uniform 窗，样本协方差）。

    与 skimage.metrics.structural_similarity 默认口径一致
    （gaussian_weights=False, win_size=7, use_sample_covariance=True），
    使验证/选模型指标与 E01 参照一致。

    pred/target: [B,1,H,W] 或 [1,H,W]，像素范围 [0, data_range]。
    返回整图均值 SSIM (float)。
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    pred = pred.float()
    target = target.float()

    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    pad = win_size // 2
    pred = F.pad(pred, (pad, pad, pad, pad), mode="reflect")
    target = F.pad(target, (pad, pad, pad, pad), mode="reflect")

    channels = pred.size(1)
    kernel = torch.ones(
        1, 1, win_size, win_size, device=pred.device, dtype=pred.dtype
    ) / (win_size ** 2)

    def conv(x):
        return F.conv2d(x, kernel, stride=1, padding=0, groups=channels)

    mu1 = conv(pred)
    mu2 = conv(target)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv(pred * pred) - mu1_sq
    sigma2_sq = conv(target * target) - mu2_sq
    sigma12 = conv(pred * target) - mu1_mu2

    # 样本协方差归一化（与 skimage use_sample_covariance=True 一致）
    n_pix = win_size ** 2
    cov_norm = n_pix / (n_pix - 1)
    sigma1_sq = (sigma1_sq * cov_norm).clamp(min=0)
    sigma2_sq = (sigma2_sq * cov_norm).clamp(min=0)
    sigma12 = sigma12 * cov_norm

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    # C1/C2 > 0 ⇒ denominator > 0，无需特殊处理除零（与 skimage 一致）
    ssim_map = numerator / denominator

    # 与 skimage 一致：裁掉边界 strip（uniform_filter 边界效应）后取均值，用 float64
    if pad > 0:
        ssim_map = ssim_map[:, :, pad:-pad, pad:-pad]
    return ssim_map.double().mean().item()


# ======================
# PSNR
# ======================

def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()


# ======================
# Competition Score
# ======================

def calculate_score(ssim, psnr):
    psnr_norm = (psnr - 10) / (40 - 10)
    psnr_norm = max(0, min(1, psnr_norm))
    score = 0.7 * ssim + 0.3 * psnr_norm
    return score
