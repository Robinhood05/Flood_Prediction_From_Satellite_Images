import torch
import numpy as np
import rasterio
from model import get_unet_model
from utils import visualize

def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32) / 255.0
        image = np.transpose(image, (1, 2, 0))  # HWC
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # 1CHW
    return image

def predict_flood(image_path, model_path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_unet_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = load_image(image_path).to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output)
        binary_mask = (prob > 0.5).float()

    # Decision: if >1% of pixels are flood, we call it "Flood"
    flood_ratio = binary_mask.sum().item() / binary_mask.numel()
    is_flood = flood_ratio > 0.01

    result = "🌊 Flood detected" if is_flood else "✅ No flood detected"
    print(f"[RESULT]: {result} — Flood pixels: {flood_ratio*100:.2f}%")

    # Optional visualization
    visualize(image[0].cpu(), binary_mask[0].cpu(), output[0].cpu())

    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.tif")
    else:
        image_path = sys.argv[1]
        predict_flood(image_path)
