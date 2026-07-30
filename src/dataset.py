from pathlib import Path

import cv2
import torch

from torch.utils.data import Dataset



class VirtualStainingDataset(Dataset):
    """
    DAPI -> CD68 数据集


    输入:
        DAPI图像


    输出:
        CD68染色图像

    要求:

    data/train/

        DAPI/
            xxx.png

        CD68/
            xxx.png

    文件名必须对应
    """



    def __init__(
            self,
            dapi_dir,
            target_dir,
            image_size=256
    ):

        self.dapi_dir = Path(dapi_dir)

        self.target_dir = Path(target_dir)

        self.image_size = image_size



        self.images = [

            x.name

            for x in self.dapi_dir.iterdir()

            if x.suffix.lower()
            in [
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff"
            ]

        ]



    def __len__(self):

        return len(self.images)



    def __getitem__(self,index):

        name = self.images[index]


        dapi_path = self.dapi_dir / name

        target_path = self.target_dir / name



        dapi = cv2.imread(
            str(dapi_path),
            cv2.IMREAD_GRAYSCALE
        )


        target = cv2.imread(
            str(target_path),
            cv2.IMREAD_GRAYSCALE
        )



        dapi = cv2.resize(
            dapi,
            (
                self.image_size,
                self.image_size
            )
        )


        target = cv2.resize(
            target,
            (
                self.image_size,
                self.image_size
            )
        )



        dapi = dapi / 255.0

        target = target / 255.0



        dapi = torch.tensor(
            dapi,
            dtype=torch.float32
        ).unsqueeze(0)



        target = torch.tensor(
            target,
            dtype=torch.float32
        ).unsqueeze(0)



        return dapi,target