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
    model = build_model(MODEL_NAME).to(DEVICE)
    model = load_model(model, MODEL_PATH, DEVICE)
    predict_dir(model, INPUT_DIR, OUTPUT_DIR, DEVICE, IMAGE_SIZE)


if __name__ == "__main__":
    main()
