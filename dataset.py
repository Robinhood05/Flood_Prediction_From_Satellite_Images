import os
import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio

class FloodSegmentationDataset(Dataset):
    def __init__(self, input_dir, mask_dir, transform=None):
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.file_list = sorted(os.listdir(input_dir))
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.file_list[idx])
        mask_path = os.path.join(self.mask_dir, self.file_list[idx])

        with rasterio.open(input_path) as src:
            image = src.read().astype(np.float32) / 255.0
            image = np.transpose(image, (1, 2, 0))

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.uint8)

        image = torch.tensor(image).permute(2, 0, 1)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask
