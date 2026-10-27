"""Generate standard test images for experiments."""

import numpy as np
from PIL import Image
import os

def generate_test_images(output_dir: str, size: int = 256):
    """Generate a set of standard test images for the experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Cameraman-like (synthetic gradient + structure)
    y, x = np.mgrid[0:size, 0:size]
    
    # 2. Baboon-like (high frequency texture using noise pattern)
    rng = np.random.RandomState(123)
    baboon = rng.randint(0, 256, (size, size), dtype=np.uint8)
    # Add some structure
    for i in range(0, size, 4):
        baboon[i:i+2, :] = (baboon[i:i+2, :].astype(int) + 30) % 256
    Image.fromarray(baboon.astype(np.uint8), mode='L').save(
        os.path.join(output_dir, 'baboon_synthetic.png'))
    
    # 3. Simple geometric pattern (like a binary/structured image)
    geometric = np.zeros((size, size), dtype=np.uint8)
    # Draw concentric squares
    for i in range(0, size//2, 16):
        geometric[i:size-i, i:size-i] = 255 - geometric[i, i]
    Image.fromarray(geometric, mode='L').save(
        os.path.join(output_dir, 'geometric.png'))
    
    # 4. Gradient image (smooth)
    gradient = np.tile(np.linspace(0, 255, size).astype(np.uint8), (size, 1))
    Image.fromarray(gradient, mode='L').save(
        os.path.join(output_dir, 'gradient.png'))
    
    # 5. Checkerboard (binary pattern)
    checker = np.zeros((size, size), dtype=np.uint8)
    block = size // 8
    for i in range(size):
        for j in range(size):
            if ((i // block) + (j // block)) % 2 == 0:
                checker[i, j] = 255
    Image.fromarray(checker, mode='L').save(
        os.path.join(output_dir, 'checkerboard.png'))
    
    # 6. Natural-looking pattern (Gaussian blobs)
    natural = np.zeros((size, size), dtype=np.float64)
    for _ in range(20):
        cy, cx = rng.randint(0, size, 2)
        sigma = rng.uniform(10, 40)
        amplitude = rng.uniform(50, 200)
        yy, xx = np.mgrid[0:size, 0:size]
        gaussian = amplitude * np.exp(-((yy-cy)**2 + (xx-cx)**2) / (2*sigma**2))
        natural += gaussian
    natural = np.clip(natural, 0, 255).astype(np.uint8)
    Image.fromarray(natural, mode='L').save(
        os.path.join(output_dir, 'natural_blobs.png'))
    
    print(f"Generated test images in {output_dir}")
    return [
        'baboon_synthetic.png',
        'geometric.png', 
        'gradient.png',
        'checkerboard.png',
        'natural_blobs.png',
    ]


if __name__ == '__main__':
    generate_test_images('test_images')
