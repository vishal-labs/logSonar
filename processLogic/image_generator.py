"""
image_generator.py — Converts a pixel grid (numpy array) into a PNG image.

Saves generated images to the `generated_images/` directory with
timestamp-based filenames for easy chronological ordering.
"""

import os
from datetime import datetime

import numpy as np
from PIL import Image


# Default output directory (relative to project root)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "generated_images",
)


class ImageGenerator:
    def __init__(self, output_dir=None):
        """
        Args:
            output_dir: Directory to save generated images.
                        Defaults to `<project_root>/generated_images/`.
        """
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, pixel_grid, filename=None, scale_factor=10):
        """
        Create and save a PNG image from a pixel grid.

        Args:
            pixel_grid: numpy array of shape (H, W, 3), dtype uint8.
            filename:   Optional custom filename. If None, uses timestamp.
            scale_factor: Upscale factor so the tiny 50×50 image is viewable.
                          A factor of 10 produces a 500×500 image.

        Returns:
            Absolute path to the saved image file.
        """
        if not isinstance(pixel_grid, np.ndarray):
            pixel_grid = np.array(pixel_grid, dtype=np.uint8)

        # Create the image from the raw pixel data
        img = Image.fromarray(pixel_grid, mode="RGB")

        # Upscale with nearest-neighbor to keep crisp pixels
        if scale_factor > 1:
            new_size = (img.width * scale_factor, img.height * scale_factor)
            img = img.resize(new_size, Image.NEAREST)

        # Build filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"log_image_{timestamp}.png"

        filepath = os.path.join(self.output_dir, filename)
        img.save(filepath)
        print(f"[ImageGenerator] Saved image → {filepath}")
        return filepath

    def save_raw(self, pixel_grid, filename=None):
        """
        Save the image at its native resolution (no upscaling).
        Useful for analysis pipelines that need the exact pixel data.

        Returns:
            Absolute path to the saved image file.
        """
        return self.save(pixel_grid, filename=filename, scale_factor=1)
