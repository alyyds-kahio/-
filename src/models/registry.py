from .unet import UNet, UNetSkip, UNetSkip4
from .pix2pix_unet import Pix2PixUNet


MODEL_REGISTRY = {
    "unet": UNet,
    "unet_skip": UNetSkip,
    "unet_skip4": UNetSkip4,
    "pix2pix_unet": Pix2PixUNet,
}


def build_model(model_name="unet", **kwargs):
    """按名称构建模型，便于后续替换/新增模型结构。"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)
