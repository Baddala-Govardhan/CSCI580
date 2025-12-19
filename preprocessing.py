import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


def get_preprocessing_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor and scales [0,255] -> [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # Scales [0,1] -> [-1,1]
    ])
    return transform


def preprocess_images(images: np.ndarray) -> torch.Tensor:
    transform = get_preprocessing_transform()
    processed_images = []
    
    for img in images:
        # Convert numpy array to PIL Image
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img.astype(np.uint8), mode='L')
        else:
            img_pil = img
        
        # Apply transforms
        img_tensor = transform(img_pil)
        processed_images.append(img_tensor)
    
    # Stack all images into a single tensor
    return torch.stack(processed_images)


if __name__ == "__main__":
    # Test preprocessing
    # Create dummy images
    dummy_images = np.random.randint(0, 256, size=(10, 28, 28), dtype=np.uint8)
    print(f"Input shape: {dummy_images.shape}")
    print(f"Input range: [{dummy_images.min()}, {dummy_images.max()}]")
    
    processed = preprocess_images(dummy_images)
    print(f"Output shape: {processed.shape}")
    print(f"Output range: [{processed.min():.3f}, {processed.max():.3f}]")

