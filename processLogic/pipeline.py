"""
pipeline.py — Service-aware orchestrator for the log→image→analysis flow.

Wires:
  ServiceAwareBuffer  →  ImageGenerator  →  ImageAnalyzer

Usage from painter.py:
    from processLogic.pipeline import LogSonarPipeline
    pipeline = LogSonarPipeline(
        width=50, rows_per_service=16,
        services=["svc-1", "svc-2", "svc-3"]
    )
    pipeline.process_pixel("svc-1", color=(255, 0, 0))
"""

import os
import threading

from processLogic.pixel_buffer import ServiceAwareBuffer
from processLogic.image_generator import ImageGenerator
from processLogic.image_analyzer import ImageAnalyzer


class LogSonarPipeline:
    def __init__(self, width=50, rows_per_service=16, services=None,
                 generated_dir=None, analysis_dir=None):
        """
        Args:
            width:            Pixel grid width.
            rows_per_service: Rows allocated per service band.
            services:         List of expected service names.
            generated_dir:    Output directory for raw images.
            analysis_dir:     Output directory for analysis results.
        """
        self.width = width
        self.rows_per_service = rows_per_service
        self.services = services or []
        self._image_count = 0
        self._lock = threading.Lock()

        # --- Initialize components ---
        self.generator = ImageGenerator(output_dir=generated_dir)
        self.analyzer = ImageAnalyzer(output_dir=analysis_dir)
        self.buffer = ServiceAwareBuffer(
            width=width,
            rows_per_service=rows_per_service,
            services=self.services,
            on_grid_full=self._on_grid_complete,
        )

        print(f"[Pipeline] Initialized — {len(self.services)} services, "
              f"{rows_per_service} rows each, {width} cols")
        print(f"[Pipeline] Total image: {self.buffer.total_height}×{width}")
        print(f"[Pipeline] Images → {self.generator.output_dir}")
        print(f"[Pipeline] Analysis → {self.analyzer.output_dir}")

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def process_pixel(self, service, color):
        """
        Feed a single pixel from a specific service into the pipeline.

        Args:
            service: Service name string.
            color:   Tuple (R, G, B) with values 0-255.
        """
        self.buffer.add_pixel(service, color)

    def get_status(self):
        """Return per-service fill status and image count."""
        fill = self.buffer.get_fill_status()
        return {
            "services": fill,
            "images_generated": self._image_count,
            "grid_dimensions": f"{self.buffer.total_height}×{self.width}",
        }

    # ------------------------------------------------------------------ #
    #  Callback: invoked when all service bands are full                   #
    # ------------------------------------------------------------------ #

    def _on_grid_complete(self, grid, service_layout):
        """
        Called when all services have filled their bands.
        Spawns a background thread for image generation + analysis
        so the caller (Flask) isn't blocked.
        """
        with self._lock:
            self._image_count += 1
            count = self._image_count

        # Run the heavy work in background so Flask can keep accepting requests
        thread = threading.Thread(
            target=self._generate_and_analyze,
            args=(grid, service_layout, count),
            daemon=True,
        )
        thread.start()

    def _generate_and_analyze(self, grid, service_layout, count):
        """Background worker: save images and run analysis."""
        try:
            print(f"\n[Pipeline] Grid #{count} complete — generating & analyzing (background)...")

            # 1. Save upscaled version for human viewing
            self.generator.save(grid)

            # 2. Save raw version for analysis
            raw_path = self.generator.save_raw(
                grid, filename=f"raw_grid_{count:04d}.png"
            )

            # 3. Analyze with service layout info
            report = self.analyzer.analyze(raw_path, service_layout=service_layout)

            # Print summary
            print(f"[Pipeline] Grid #{count} analysis done:")
            if "per_service" in report:
                for svc, svc_report in report["per_service"].items():
                    colors = svc_report.get("dominant_colors", [])
                    top_color = colors[0]["hex"] if colors else "N/A"
                    anomalies = svc_report.get("anomalies", {}).get("count", 0)
                    print(f"  {svc}: top_color={top_color}, anomalies={anomalies}")
        except Exception as e:
            print(f"[Pipeline] ERROR in grid #{count} analysis: {e}")

