import argparse

from src.models import build_model
from src.inference.predictor import load_model, predict_dir
from src.config import (
    DEVICE,
    MODEL_PATH,
    INPUT_DIR,
    OUTPUT_DIR,
    MODEL_NAME,
    IMAGE_SIZE,
)


def main():
    parser = argparse.ArgumentParser(description="Virtual Staining Inference")
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Model name override (default from config: {MODEL_NAME})",
    )
    parser.add_argument(
        "--ckpt",
        default=MODEL_PATH,
        help="Checkpoint path (default from config)",
    )
    args = parser.parse_args()

    model = build_model(args.model).to(DEVICE)
    model = load_model(model, args.ckpt, DEVICE)
    predict_dir(model, INPUT_DIR, OUTPUT_DIR, DEVICE, IMAGE_SIZE)


if __name__ == "__main__":
    main()
