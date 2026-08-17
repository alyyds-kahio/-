import torch
from torch.utils.data import DataLoader, Subset

from .dataset import VirtualStainingDataset


def build_train_val_loaders(
        dapi_dir,
        target_dir,
        batch_size=4,
        val_ratio=0.1,
        seed=42,
        num_workers=0,
        image_size=256,
        augment=True):
    """
    构建 train/val DataLoader。

    - train Dataset：augment=True（若 augment 参数为 True）
    - val Dataset：  augment=False（始终不增强）
    - train/val 使用同一组划分索引（与历史 random_split 的 randperm 一致，seed 固定）
    - train/val 是独立 Dataset 实例，仅 augment 标志不同

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
