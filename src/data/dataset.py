from pathlib import Path

from torch.utils.data import Dataset

from .transforms import (
    read_grayscale,
    resize_to_square,
    normalize_to_01,
    to_single_channel_tensor,
    random_dihedral_key,
    apply_dihedral,
)


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


class VirtualStainingDataset(Dataset):
    """
    DAPI -> CD68 数据集

    输入:
        DAPI图像

    输出:
        CD68染色图像

    要求:
    data/train/DAPI/xxx.png
    data/train/CD68/xxx.png

    文件名必须对应
    """

    def __init__(self, dapi_dir, target_dir, image_size=256, augment=False):
        self.dapi_dir = Path(dapi_dir)
        self.target_dir = Path(target_dir)
        self.image_size = image_size
        self.augment = augment

        self.images = [
            x.name
            for x in self.dapi_dir.iterdir()
            if x.suffix.lower() in IMAGE_SUFFIXES
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]

        dapi_path = self.dapi_dir / name
        target_path = self.target_dir / name

        dapi = read_grayscale(dapi_path)
        target = read_grayscale(target_path)

        dapi = resize_to_square(dapi, self.image_size)
        target = resize_to_square(target, self.image_size)

        # D4 数据增强：仅训练集；同一 key 施加到 DAPI 与 CD68，保证空间对应不变
        if self.augment:
            key = random_dihedral_key()
            dapi = apply_dihedral(dapi, key)
            target = apply_dihedral(target, key)

        dapi = normalize_to_01(dapi)
        target = normalize_to_01(target)

        dapi = to_single_channel_tensor(dapi)
        target = to_single_channel_tensor(target)

        return dapi, target
