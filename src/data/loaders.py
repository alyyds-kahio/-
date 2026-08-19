import torch
from torch.utils.data import DataLoader, Subset

from .dataset import VirtualStainingDataset


def _roi_split_indices(images, val_ratio, seed, n_total):
    """按 ROI 前缀分组划分 train/val 索引。

    - ROI 前缀：文件名 split("_")[0]，如 "ROI025_00_00.jpg" -> "ROI025"
    - 用 seed 打乱 ROI 顺序（可复现）
    - 贪心把 ROI 加入 val，直到 val patch 数 >= int(n_total * val_ratio)
    - 被选中 ROI 整组进 val，其余进 train（val 只含训练未见 ROI）

    返回 (train_idx, val_idx)。
    """
    rois = {}
    for i, name in enumerate(images):
        roi = name.split("_")[0]
        rois.setdefault(roi, []).append(i)

    roi_list = list(rois.keys())
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(roi_list), generator=g).tolist()

    target_val = int(n_total * val_ratio)
    val_idx = []
    n_val_patches = 0
    for k in order:
        roi = roi_list[k]
        val_idx.extend(rois[roi])
        n_val_patches += len(rois[roi])
        if n_val_patches >= target_val:
            break

    val_set = set(val_idx)
    train_idx = [i for i in range(n_total) if i not in val_set]
    return train_idx, val_idx


def build_train_val_loaders(
        dapi_dir,
        target_dir,
        batch_size=4,
        val_ratio=0.1,
        seed=42,
        num_workers=0,
        image_size=256,
        augment=True,
        roi_split=False):
    """
    构建 train/val DataLoader。

    - train Dataset：augment=True（若 augment 参数为 True）
    - val Dataset：  augment=False（始终不增强）
    - 划分方式：
      - roi_split=False（默认）：patch 级 random_split（原行为）
      - roi_split=True：按 ROI 前缀分组，val 只含训练未见过 ROI
    - train/val 使用同一组划分索引（独立 Dataset 实例，仅 augment 不同）

    返回: (train_loader, val_loader, n_train, n_val, n_total)
    """
    train_ds = VirtualStainingDataset(
        dapi_dir=dapi_dir,
        target_dir=target_dir,
        image_size=image_size,
        augment=augment,
    )
    val_ds = VirtualStainingDataset(
        dapi_dir=dapi_dir,
        target_dir=target_dir,
        image_size=image_size,
        augment=False,
    )

    # 耦合保护：train/val 必须共享同一文件列表，否则同一索引会指向不同文件
    assert train_ds.images == val_ds.images, "train/val dataset 文件列表不一致"

    n_total = len(train_ds)

    if roi_split:
        train_idx, val_idx = _roi_split_indices(train_ds.images, val_ratio, seed, n_total)
    else:
        train_size = int(n_total * (1 - val_ratio))
        # 与原有 random_split 相同的索引生成方式，保持历史划分不变
        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).tolist()
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]

    train_dataset = Subset(train_ds, train_idx)
    val_dataset = Subset(val_ds, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, len(train_idx), len(val_idx), n_total
