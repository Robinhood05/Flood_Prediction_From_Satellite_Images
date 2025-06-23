import matplotlib.pyplot as plt
import torch

def visualize(image, mask, prediction=None):
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze().cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Ground Truth Mask")

    if prediction is not None:
        pred = torch.sigmoid(prediction).squeeze().cpu().detach().numpy()
        plt.subplot(1, 3, 3)
        plt.imshow(pred > 0.5, cmap='gray')
        plt.title("Predicted Mask")

    plt.tight_layout()
    plt.show()
