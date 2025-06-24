import os
import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio
from matplotlib import pyplot as plt

# Albumentations imports
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Augmentation pipeline
def get_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=0.0, std=1.0, max_pixel_value=1.0),
        ToTensorV2()
    ])

class FloodSegmentationDataset(Dataset):
    def __init__(self, input_dir, mask_dir, transform=None):
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.transform = transform if transform else get_augmentation()

        self.image_filenames = sorted([
            f for f in os.listdir(input_dir)
            if f.endswith(".tif") and os.path.exists(os.path.join(mask_dir, f))
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.input_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, img_filename)

        # Load image (C, H, W) and convert to (H, W, C)
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
        image = np.transpose(image, (1, 2, 0))  # (H, W, C)

        # Normalize to [0, 1] per channel
        channel_mins = image.min(axis=(0, 1), keepdims=True)
        channel_maxs = image.max(axis=(0, 1), keepdims=True)
        image = (image - channel_mins) / (channel_maxs - channel_mins + 1e-6)

        # Load and binarize mask (H, W)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        # Apply albumentations (returns torch tensors)
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"].unsqueeze(0)  # Add channel dim: (1, H, W)

        return image, mask

# For testing & visualization
if __name__ == "__main__":
    dataset = FloodSegmentationDataset("data/masks_input", "data/masks_output")

    img, mask = dataset[0]
    print(f"Image shape: {img.shape}, min/max: {img.min().item():.4f}/{img.max().item():.4f}")
    print(f"Mask shape: {mask.shape}, unique values: {torch.unique(mask)}")

    # Visualize
    import matplotlib.pyplot as plt
    img_np = img.permute(1, 2, 0).numpy()
    mask_np = mask.squeeze().numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Augmented Image")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap="gray")
    plt.title("Augmented Mask")

    plt.tight_layout()
    plt.show()
    print(f"Total samples: {len(dataset)}")

