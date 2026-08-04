import os

import torch
from PIL import Image
from torchvision import transforms


def load_model(model, checkpoint_path, device):
    """
    加载模型权重，兼容两种格式：
    1. 标准包装格式（含 model_state_dict）
    2. 旧版裸 state_dict（checkpoint 管理前的产物，仅有权重）
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded:", checkpoint_path)
        print("Best score:", checkpoint.get("best_score", "unknown"))
        print("Epoch:", checkpoint.get("epoch", "unknown"))
    elif (
        isinstance(checkpoint, dict)
        and "model_state_dict" not in checkpoint
        and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint.items())
    ):
        model.load_state_dict(checkpoint)
        print("Model loaded:", checkpoint_path)
        print("Best score: unknown (legacy raw state_dict)")
        print("Epoch: unknown (legacy raw state_dict)")
    else:
        raise RuntimeError(f"Unrecognized checkpoint format: {checkpoint_path}")

    model.eval()
    return model


def predict_image(model, image_path, device, image_size=256):
    """单张预测，返回 PIL 图像。预处理与原 predict.py 相同。"""
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("L")

    x = transform(image)
    x = x.unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        pred = model(x)

    pred = pred.squeeze(0)
    pred = pred.cpu()
    pred = torch.clamp(pred, 0, 1)

    output = transforms.ToPILImage()(pred)

    return output


def predict_dir(model, input_dir, output_dir, device, image_size=256):
    """批量预测：输入目录内所有图片，输出到 output_dir，保持原文件名。"""
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)
    count = 0

    for filename in files:
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        result = predict_image(model, input_path, device, image_size)
        result.save(output_path)

        count += 1
        print("Saved:", output_path)

    print()
    print("Prediction finished!")
    print("Total images:", count)

    return count
