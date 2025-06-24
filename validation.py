import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from dataset import FloodSegmentationDataset
from model import get_unet_model
from utils import visualize

def compute_metrics(preds, masks):
    preds = torch.sigmoid(preds) > 0.5
    preds = preds.cpu().numpy().astype(int).flatten()
    masks = masks.cpu().numpy().astype(int).flatten()
    f1 = f1_score(masks, preds)
    acc = accuracy_score(masks, preds)
    return f1, acc

def evaluate(model, dataloader, device):
    model.eval()
    all_f1, all_acc = [], []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device).float()
            outputs = model(images)
            f1, acc = compute_metrics(outputs, masks)
            all_f1.append(f1)
            all_acc.append(acc)

    print(f"✅ F1 Score: {np.mean(all_f1):.4f} | Accuracy: {np.mean(all_acc):.4f}")

def main():
    # Paths
    input_dir = "data/masks_input"
    mask_dir = "data/masks_output"

    # Dataset
    dataset = FloodSegmentationDataset(input_dir, mask_dir)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_unet_model()
    model.load_state_dict(torch.load("model.pth", map_location=device))  # <- Your saved model
    model.to(device)

    # Run evaluation
    evaluate(model, val_loader, device)

    # Optional visualization
    for images, masks in val_loader:
        images = images.to(device)
        with torch.no_grad():
            preds = model(images)
        visualize(images[0], masks[0], preds[0])
        break  # remove break to visualize more samples

if __name__ == "__main__":
    main()
