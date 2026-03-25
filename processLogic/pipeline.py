"""
pipeline.py — Orchestrates the full log-pixel → image → analysis flow.

This is the glue module that wires:
  PixelBuffer  →  ImageGenerator  →  ImageAnalyzer

Usage from painter.py:
    from processLogic.pipeline import LogSonarPipeline
    pipeline = LogSonarPipeline()
    pipeline.process_pixel(color=(255, 0, 0))
"""

import os
import threading

from processLogic.pixel_buffer import PixelBuffer
from processLogic.image_generator import ImageGenerator
from processLogic.image_analyzer import ImageAnalyzer


class LogSonarPipeline:
    def __init__(self, width=50, height=50, generated_dir=None, analysis_dir=None):
        """
        Args:
            width:          Pixel grid width (default 50).
            height:         Pixel grid height (default 50).
            generated_dir:  Output directory for raw images.
            analysis_dir:   Output directory for analysis results.
        """
        self.width = width
        self.height = height
        self._image_count = 0
        self._lock = threading.Lock()

        # --- Initialize components ---
        self.generator = ImageGenerator(output_dir=generated_dir)
        self.analyzer = ImageAnalyzer(output_dir=analysis_dir)
        self.buffer = PixelBuffer(
            width=width,
            height=height,
            on_grid_full=self._on_grid_complete,
        )

        print(f"[Pipeline] Initialized — grid {width}×{height}, "
              f"images → {self.generator.output_dir}, "
              f"analysis → {self.analyzer.output_dir}")

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def process_pixel(self, color):
        """
        Feed a single pixel (R, G, B) into the pipeline.
        When enough pixels accumulate to fill the grid, an image
        is automatically generated and analyzed.
        """
        self.buffer.add_pixel(color)

    def process_pixel_at(self, row, col, color):
        """
        Feed a pixel at a specific grid position.
        """
        self.buffer.add_pixel_at(row, col, color)

    def get_status(self):
        """Return current pipeline status."""
        fill = self.buffer.get_fill_percentage()
        return {
            "grid_fill_percentage": round(fill * 100, 2),
            "images_generated": self._image_count,
            "grid_size": f"{self.width}×{self.height}",
        }

    # ------------------------------------------------------------------ #
    #  Callback: invoked when the pixel buffer fills up                    #
    # ------------------------------------------------------------------ #

    def _on_grid_complete(self, grid):
        """
        Called automatically when the PixelBuffer has accumulated
        width × height pixels.

        Steps:
          1. Save the grid as a viewable PNG (upscaled)
          2. Save a raw PNG (native resolution) for analysis
          3. Run the analyzer on the raw image
        """
        with self._lock:
            self._image_count += 1
            count = self._image_count

        print(f"\n[Pipeline] Grid #{count} complete — generating image & running analysis...")

        # 1. Save upscaled version for human viewing
        self.generator.save(grid)

        # 2. Save raw version for analysis
        raw_path = self.generator.save_raw(
            grid, filename=f"raw_grid_{count:04d}.png"
        )

        # 3. Analyze
        report = self.analyzer.analyze(raw_path)

        print(f"[Pipeline] Grid #{count} done — "
              f"dominant color: {report['dominant_colors'][0]['hex'] if report['dominant_colors'] else 'N/A'}, "
              f"anomalies: {report['anomalies']['count']}")

        return report
