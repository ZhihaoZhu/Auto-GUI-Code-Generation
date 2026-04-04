"""Utility functions for image preprocessing."""

from PIL import Image
from torchvision import transforms

from model.classes.model.Config import IMAGE_SIZE


def get_image_transform():
    """Return the standard image transform pipeline for model input.

    Resizes to IMAGE_SIZE x IMAGE_SIZE, converts to tensor, and normalizes
    with ImageNet mean and standard deviation.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_preprocessed_img(img_path):
    """Load an image and apply the standard transform pipeline.

    Args:
        img_path: Path to the image file.

    Returns:
        Tensor of shape (3, IMAGE_SIZE, IMAGE_SIZE), normalized.
    """
    img = Image.open(img_path).convert("RGB")
    return get_image_transform()(img)


def show(image_tensor):
    """Display a transformed image tensor using matplotlib.

    Args:
        image_tensor: Tensor of shape (3, H, W), ImageNet-normalized.
    """
    import matplotlib.pyplot as plt

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = image_tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1, 2, 0).clamp(0, 1).numpy()
    plt.imshow(img)
    plt.axis("off")
    plt.show()
