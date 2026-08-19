import torch


class EMA:
    """Exponential Moving Average 权重滑动平均（独立、低耦合）。

    用法：
        ema = EMA(model, decay=0.999)
        # 每步更新：ema.update(model)
        # 验证/评估用 EMA 权重：ema.apply(model)；用完再 apply 回普通权重
    关闭（use_ema=False）时完全不实例化，训练逻辑不变。
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v.detach()
            else:
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def apply(self, model):
        """把 EMA 权重写入 model（调用后 model 的权重变为 EMA 版）。"""
        state = model.state_dict()
        for k in self.shadow:
            if k in state:
                state[k].copy_(self.shadow[k])

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}
