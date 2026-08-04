import torch


# ======================
# SSIM
# ======================

def calculate_ssim(pred, target):
    mse = torch.mean((pred - target) ** 2)
    ssim = 1 - mse
    return ssim.item()


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
