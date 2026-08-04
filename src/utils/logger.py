import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime

import torch


LOGGER_NAME = "virtual_staining"


def get_logger():
    """返回项目 logger 实例（由 setup_logging 配置 handler）。"""
    return logging.getLogger(LOGGER_NAME)


def setup_logging(log_dir, run_tag="train"):
    """
    配置日志：同时输出到 logs/<run_tag>_<时间戳>.log 与终端。

    返回: (logger, log_path)
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{run_tag}_{timestamp}.log")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 清空已有 handler，避免重复配置时叠加输出
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(message)s")
    )

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger, log_path


def get_git_commit():
    """返回当前 git commit hash；非 git 仓库时返回 None。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_startup_info(project_name="virtual_staining"):
    """采集启动阶段信息：时间 / 项目名 / git commit / 运行环境 / 硬件。"""
    cuda_available = torch.cuda.is_available()
    info = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_name": project_name,
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }
    return info


def log_training_config(logger, config):
    """以分块形式记录配置阶段信息。config 为 {配置项: 值} 的 dict。"""
    logger.info("-" * 30 + " Config " + "-" * 30)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("-" * 68)


def _atomic_json_write(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class ExperimentRecorder:
    """记录每轮训练的实验摘要。

    - 每轮:   experiments/experiment_<时间戳>.json （详情）
    - 索引:   logs/experiments.json                 （追加汇总，方便对比）
    """

    def __init__(self, log_dir, experiment_dir):
        self.log_dir = log_dir
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_record(self, record):
        """写入本轮实验 JSON，并把记录追加到 logs/experiments.json 索引。

        返回实验文件路径。
        """
        record.setdefault("time", self.timestamp)

        experiment_path = os.path.join(
            self.experiment_dir, f"experiment_{self.timestamp}.json"
        )
        _atomic_json_write(experiment_path, record)

        index_path = os.path.join(self.log_dir, "experiments.json")
        entries = []
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                try:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        entries = loaded
                except json.JSONDecodeError:
                    entries = []
        entries.append(record)
        _atomic_json_write(index_path, entries)

        return experiment_path
