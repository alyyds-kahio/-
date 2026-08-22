from .unet import UNet, UNetSkip, UNetSkip4, UNetSkip4S
from .pix2pix_unet import Pix2PixUNet
from .pix2pix_resnet import Pix2PixResNet
from .resunet import ResUNet


def _pix2pix_unet_wide():
    """E06-W1：Pix2Pix-UNet 加宽（channels 48,96,192,384）。"""
    return Pix2PixUNet(features=(48, 96, 192, 384))


def _pix2pix_unet_deep():
    """E06-Deep：Pix2Pix-UNet 加深（5 stage，channels 54,80,128,192,256，MACs≈6.99G 匹配 E05.5）。"""
    return Pix2PixUNet(features=(54, 80, 128, 192, 256))


def _pix2pix_unet_big():
    """E06-B：Pix2Pix-UNet 加宽加深 6 层（64,128,256,512,512,512，bottleneck 4×4，~26M）。"""
    return Pix2PixUNet(features=(64, 128, 256, 512, 512, 512))


def _pix2pix_generator():
    """E06-C 的 Generator（同 6 层 64,128,256,512,512,512），供完整 Pix2Pix 使用。"""
    return Pix2PixUNet(features=(64, 128, 256, 512, 512, 512))


MODEL_REGISTRY = {
    "unet": UNet,
    "unet_skip": UNetSkip,
    "unet_skip4": UNetSkip4,
    "unet_skip4s": UNetSkip4S,
    "pix2pix_unet": Pix2PixUNet,
    "pix2pix_unet_wide": _pix2pix_unet_wide,
    "pix2pix_unet_deep": _pix2pix_unet_deep,
    "pix2pix_resnet": Pix2PixResNet,
    "resunet": ResUNet,
    "pix2pix_unet_big": _pix2pix_unet_big,
    "pix2pix_generator": _pix2pix_generator,
}


def build_model(model_name="unet", **kwargs):
    """按名称构建模型，便于后续替换/新增模型结构。"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)
