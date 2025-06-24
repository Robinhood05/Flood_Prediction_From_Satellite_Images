import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from dataset import FloodSegmentationDataset
from model import get_unet_model
from utils import visualize

# Paths to your dataset
input_dir = "data/masks_input"
mask_dir = "data/masks_output"

# Hyperparameters
batch_size = 4
epochs = 10
lr = 1e-4
train_val_split = 0.8

# Load dataset
dataset = FloodSegmentationDataset(input_dir, mask_dir)
train_size = int(len(dataset) * train_val_split)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_unet_model().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"[Epoch {epoch+1}/{epochs}] Training Loss: {train_loss:.4f}")

    # Validation (on 1 batch for preview)
    model.eval()
    with torch.no_grad():
        val_images, val_masks = next(iter(val_loader))
        val_images = val_images.to(device)
        preds = model(val_images)
        visualize(val_images[0], val_masks[0], preds[0])
from matplotlib import pyplot as plt

img, mask = dataset[0]
plt.subplot(1, 2, 1)
plt.imshow(img.permute(1, 2, 0))  # Convert from CHW to HWC
plt.title("Input")

plt.subplot(1, 2, 2)
plt.imshow(mask.squeeze(), cmap="gray")
plt.title("Mask")

plt.show()
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")
# Save the trained model
