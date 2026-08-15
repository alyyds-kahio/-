import argparse

import torch

from src.config import (
    DEVICE,
    TRAIN_DAPI_DIR,
    TRAIN_TARGET_DIR,
    CHECKPOINT_DIR,
    LAST_MODEL_PATH,
    BEST_MODEL_PATH,
    LOG_DIR,
    EXPERIMENT_DIR,
    PROJECT_NAME,
    BATCH_SIZE,
    VAL_RATIO,
    SEED,
    NUM_WORKERS,
    IMAGE_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SSIM_WEIGHT,
    MODEL_NAME,
    LOSS_NAME,
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
)
from src.data.loaders import build_train_val_loaders
from src.models import build_model
from src.losses import build_loss
from src.trainer.trainer import Trainer
from src.utils.logger import (
    setup_logging,
    get_startup_info,
    log_training_config,
    ExperimentRecorder,
)


def train():
    parser = argparse.ArgumentParser(description="Virtual Staining Training")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last_model.pth",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Model name override (default from config: {MODEL_NAME})",
    )
    args = parser.parse_args()

    model_name = args.model

    # ======================
    # 日志系统
    # ======================

    logger, log_path = setup_logging(LOG_DIR, run_tag="train")

    # 启动阶段
    startup = get_startup_info(PROJECT_NAME)
    logger.info("=" * 60)
    logger.info("Virtual Staining Training")
    for key, value in startup.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"  resume: {args.resume}")
    logger.info("=" * 60)

    # ======================
    # Dataset / Loaders
    # ======================

    train_loader, val_loader, n_train, n_val, n_total = build_train_val_loaders(
        dapi_dir=TRAIN_DAPI_DIR,
        target_dir=TRAIN_TARGET_DIR,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
        seed=SEED,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
    )
    logger.info(f"Dataset size: {n_total}")
    logger.info(f"Train: {n_train}, Validation: {n_val}")

    # ======================
    # Model / Loss
    # ======================

    model = build_model(model_name).to(DEVICE)
    logger.info(f"Model loaded: {model_name}")
    param_count = sum(p.numel() for p in model.parameters())

    criterion = build_loss(LOSS_NAME, ssim_weight=SSIM_WEIGHT)

    # 配置阶段
    log_training_config(logger, {
        "model": model_name,
        "param_count": param_count,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": OPTIMIZER_NAME,
        "scheduler": SCHEDULER_NAME,
        "loss": LOSS_NAME,
        "ssim_weight": SSIM_WEIGHT,
        "dapi_dir": TRAIN_DAPI_DIR,
        "target_dir": TRAIN_TARGET_DIR,
        "n_train": n_train,
        "n_val": n_val,
    })

    # ======================
    # Trainer
    # ======================

    trainer = Trainer(
        model=model,
        criterion=criterion,
        device=DEVICE,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR,
        last_model_path=LAST_MODEL_PATH,
        best_model_path=BEST_MODEL_PATH,
    )

    try:
        trainer.fit(train_loader, val_loader, resume=args.resume)
    except Exception:
        logger.exception("Training failed with exception")
        raise

    # ======================
    # 实验记录
    # ======================

    record = {
        "time": startup["time"],
        "project_name": PROJECT_NAME,
        "git_commit": startup["git_commit"],
        "model": model_name,
        "param_count": param_count,
        "config": {
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "optimizer": OPTIMIZER_NAME,
            "scheduler": SCHEDULER_NAME,
            "loss": LOSS_NAME,
            "ssim_weight": SSIM_WEIGHT,
            "val_ratio": VAL_RATIO,
            "seed": SEED,
            "dapi_dir": TRAIN_DAPI_DIR,
            "target_dir": TRAIN_TARGET_DIR,
            "n_train": n_train,
            "n_val": n_val,
        },
        "checkpoint": {
            "last_model_path": LAST_MODEL_PATH,
            "best_score_model_path": BEST_MODEL_PATH,
            "history_best_dir": CHECKPOINT_DIR,
        },
        "log_path": log_path,
        "resume": args.resume,
        "final_score": trainer.best_score,
        "best_epoch": trainer.best_epoch,
    }

    recorder = ExperimentRecorder(LOG_DIR, EXPERIMENT_DIR)
    experiment_path = recorder.save_record(record)
    logger.info(f"Experiment record saved: {experiment_path}")


if __name__ == "__main__":
    train()
