import os

import torch

from PIL import Image

from torchvision import transforms


from src.model import UNet

from src.config import DEVICE, MODEL_PATH, INPUT_DIR, OUTPUT_DIR



# ======================
# 创建输出目录
# ======================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ======================
# 图片处理
# ======================

transform = transforms.Compose(
    [
        transforms.Resize(
            (256,256)
        ),
        transforms.ToTensor()
    ]
)


# ======================
# 加载模型
# ======================

def load_model():

    model = UNet().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Model loaded:",
        MODEL_PATH
    )

    print(
        "Best score:",
        checkpoint.get(
            "best_score",
            "unknown"
        )
    )

    print(
        "Epoch:",
        checkpoint.get(
            "epoch",
            "unknown"
        )
    )

    return model


# ======================
# 单张预测
# ======================

def predict_image(
        model,
        image_path
):

    image = Image.open(
        image_path
    ).convert(
        "L"
    )

    x = transform(
        image
    )

    x = x.unsqueeze(0)

    x = x.to(DEVICE)

    with torch.no_grad():

        pred = model(x)

    pred = pred.squeeze(0)

    pred = pred.cpu()

    pred = torch.clamp(
        pred,
        0,
        1
    )

    output = transforms.ToPILImage()(
        pred
    )

    return output


# ======================
# 主程序
# ======================

def main():

    model = load_model()

    files = os.listdir(
        INPUT_DIR
    )

    count = 0

    for filename in files:

        if not filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".tif",
                ".tiff"
            )
        ):
            continue

        input_path = os.path.join(
            INPUT_DIR,
            filename
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        result = predict_image(
            model,
            input_path
        )

        result.save(
            output_path
        )

        count += 1

        print(
            "Saved:",
            output_path
        )

    print()

    print(
        "Prediction finished!"
    )

    print(
        "Total images:",
        count
    )


if __name__ == "__main__":

    main()
