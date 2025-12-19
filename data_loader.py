import os
import re
import numpy as np
from PIL import Image
from typing import Tuple


def ProjectDataLoader(repo_url: str = "https://github.com/boshen-csuchico/Handwritten-Digits-Fall-2025.git",
                     local_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    
    if local_path and os.path.exists(local_path):
        # Read from local directory
        image_files = [f for f in os.listdir(local_path) if f.endswith('.png')]
        image_files.sort()
        
        for filename in image_files:
            # Extract label from filename: <digit>-<groupID>-<memberID>.png
            match = re.match(r'^(\d+)-(\d+)-(\d+)\.png$', filename)
            if match:
                digit = int(match.group(1))
                filepath = os.path.join(local_path, filename)
                
                try:
                    img = Image.open(filepath)
                    # Convert to grayscale, handling RGBA/RGB/P modes
                    if img.mode != 'L':
                        # Convert RGBA/RGB to grayscale properly
                        if img.mode in ('RGBA', 'LA'):
                            # Create white background for transparency
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'RGBA':
                                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                            else:
                                background.paste(img)
                            img = background.convert('L')
                        else:
                            img = img.convert('L')
                    
                    # Ensure image is 28x28 (resize if needed)
                    if img.size != (28, 28):
                        img = img.resize((28, 28), Image.Resampling.LANCZOS)
                    
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(digit)
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")
    else:
        # Try to read from local 'collected_images' directory if it exists
        if os.path.exists('collected_images'):
            return ProjectDataLoader(local_path='collected_images')
        
        print("Note: Local images not found. Please download images from the GitHub repo")
        print(f"Repo: {repo_url}")
        print("Place images in a 'collected_images' directory with format: <digit>-<groupID>-<memberID>.png")
    
    if len(images) == 0:
        print("Warning: No images found. Returning empty arrays.")
        return np.array([]), np.array([])
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Loaded {len(images)} images with labels shape: {labels.shape}")
    print(f"Image shape: {images[0].shape if len(images) > 0 else 'N/A'}")
    print(f"Label range: {labels.min()} to {labels.max()}")
    
    return images, labels


if __name__ == "__main__":
    # Test the data loader
    images, labels = ProjectDataLoader()
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    if len(labels) > 0:
        print(f"Unique labels: {np.unique(labels)}")

