import os
import glob

import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandFlipd,
    RandShiftIntensityd,
    Resized,
    RandSpatialCropSamplesd,
    RandRotate90d,
)

class ImageData(Dataset):
    def __init__(self):
        self.dataset_path = "C:\\Users\\Austin Tapp\\Documents\\AMB\\NNandDL\\BIOE_658M_UNet2D\\Data\\Dataset_BUSI"
        self.raw_images_path = glob.glob(os.path.join(self.dataset_path + '\\images\\*'))
        self.segmentation_path = glob.glob(os.path.join(self.dataset_path + '\\masks\\*'))


        self.transform = Compose(

            [
                LoadImaged(keys=["image", "segmentation"]),
                EnsureChannelFirstd(keys=["image", "segmentation"]),
                Resized(keys=["image", "segmentation"], spatial_size=(256, 256)),
                RandFlipd(
                    keys=["image", "segmentation"],
                    spatial_axis=[0],
                    prob=0.50,
                ),
                RandFlipd(
                    keys=["image", "segmentation"],
                    spatial_axis=[1],
                    prob=0.50,
                ),
                RandRotate90d(
                    keys=["image", "segmentation"],
                    prob=0.50,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )

    def __len__(self):
        return len(self.dataset_path)

    def __getitem__(self, index):
        image_path = self.raw_images_path[index]
        segmentation_path = self.segmentation_path[index]
        image = {"image": image_path, "segmentation": segmentation_path}
        image_transformed = self.transform(image)
        return image_transformed

    def get_sample(self):
        return self.raw_images_path
