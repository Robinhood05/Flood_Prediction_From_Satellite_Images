import segmentation_models_pytorch as smp

def get_unet_model():
    model = smp.Unet(
        encoder_name="resnet18",        # Pretrained ResNet-18
        encoder_weights="imagenet",     # Use ImageNet weights
        in_channels=3,                  # 3 for RGB
        classes=1,                      # Binary mask (1 channel)
        activation=None                 # We'll use BCEWithLogitsLoss
    )
    return model
