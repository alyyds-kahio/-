from .reconstruction import ReconstructionLoss


LOSS_REGISTRY = {
    "reconstruction": ReconstructionLoss,
}


def build_loss(loss_name="reconstruction", **kwargs):
    """按名称构建 loss，便于后续替换/新增损失函数。"""
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{loss_name}'. Available: {list(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[loss_name](**kwargs)
