import cv2
import numpy as np
import torch


def read_grayscale(path):
    """读取灰度图（与原 dataset.py 相同：cv2.imread IMREAD_GRAYSCALE）。"""
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def resize_to_square(image, image_size):
    """resize 到 (image_size, image_size)（与原 dataset.py 相同）。"""
    return cv2.resize(image, (image_size, image_size))


def normalize_to_01(image):
    """像素归一化到 0-1（与原 dataset.py 相同：/ 255.0）。"""
    return image / 255.0


def to_single_channel_tensor(image):
    """转 float32 tensor 并加单通道维度（与原 dataset.py 相同）。"""
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)


# ======================
# D4 离散几何数据增强
# ======================

def random_dihedral_key():
    """均匀随机返回一个 D4 变换 key（0~7，各 1/8）。"""
    return int(np.random.randint(0, 8))


def apply_dihedral(image, key):
    """对 2D 灰度图施加指定的 D4 离散几何变换。

    - 无插值、尺寸不变、像素值重排。
    - key 0~7 对应 8 个 D4 元素。
    """
    if key == 0:      # identity（不变）
        return image
    if key == 1:      # rotate 90°
        return np.rot90(image, 1)
    if key == 2:      # rotate 180°
        return np.rot90(image, 2)
    if key == 3:      # rotate 270°
        return np.rot90(image, 3)
    if key == 4:      # horizontal flip（水平翻转）
        return np.fliplr(image)
    if key == 5:      # vertical flip（垂直翻转）
        return np.flipud(image)
    if key == 6:      # 主对角线镜像（转置）
        return np.swapaxes(image, 0, 1)
    if key == 7:      # 副对角线镜像
        return np.flipud(np.fliplr(np.swapaxes(image, 0, 1)))
    raise ValueError(f"Unknown dihedral key: {key}")
