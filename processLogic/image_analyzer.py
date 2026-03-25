"""
image_analyzer.py — Image processing & pattern analysis on generated log images.

Techniques applied:
  1. Dominant Color Extraction  (K-means clustering)
  2. Repeating Pattern Detection (2D autocorrelation via FFT)
  3. Anomaly Detection           (statistical outlier highlighting)

Results are saved as:
  - A JSON report  (dominant colors, periodicity, anomaly stats)
  - An annotated composite image (original + heatmap overlay)
"""

import json
import os
from datetime import datetime

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from sklearn.cluster import KMeans
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Default output directory
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "analysis_output",
)


class ImageAnalyzer:
    def __init__(self, output_dir=None, n_clusters=5, anomaly_sigma=2.0):
        """
        Args:
            output_dir:    Where to save analysis results.
            n_clusters:    Number of dominant color clusters for K-means.
            anomaly_sigma: Standard-deviation threshold for anomaly detection.
        """
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self.n_clusters = n_clusters
        self.anomaly_sigma = anomaly_sigma
        os.makedirs(self.output_dir, exist_ok=True)

    # ================================================================== #
    #  Main entry point                                                    #
    # ================================================================== #

    def analyze(self, image_path):
        """
        Run the full analysis pipeline on a single image.

        Args:
            image_path: Path to the input PNG image.

        Returns:
            dict with keys: image_path, dominant_colors, periodicity, anomalies
        """
        img = Image.open(image_path).convert("RGB")
        pixels = np.array(img, dtype=np.uint8)

        # --- 1. Dominant colors ---
        dominant_colors = self._extract_dominant_colors(pixels)

        # --- 2. Repeating patterns (autocorrelation) ---
        periodicity = self._detect_repeating_patterns(pixels)

        # --- 3. Anomaly detection ---
        anomaly_info, anomaly_mask = self._detect_anomalies(pixels)

        # --- Build report ---
        report = {
            "image_path": os.path.basename(image_path),
            "analyzed_at": datetime.now().isoformat(),
            "image_size": {"width": pixels.shape[1], "height": pixels.shape[0]},
            "dominant_colors": dominant_colors,
            "periodicity": periodicity,
            "anomalies": anomaly_info,
        }

        # --- Save outputs ---
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        json_path = os.path.join(self.output_dir, f"{base_name}_report.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        composite_path = os.path.join(self.output_dir, f"{base_name}_analysis.png")
        self._save_composite(pixels, anomaly_mask, dominant_colors, periodicity, composite_path)

        print(f"[ImageAnalyzer] Report  → {json_path}")
        print(f"[ImageAnalyzer] Visual  → {composite_path}")

        report["json_path"] = json_path
        report["composite_path"] = composite_path
        return report

    # ================================================================== #
    #  1. Dominant Color Extraction (K-Means)                              #
    # ================================================================== #

    def _extract_dominant_colors(self, pixels):
        """
        Cluster all pixels into `n_clusters` groups and return
        the centroid colors sorted by frequency (most common first).
        """
        h, w, _ = pixels.shape
        flat = pixels.reshape(-1, 3).astype(np.float64)

        # Skip all-black pixels (unfilled grid cells)
        non_black_mask = flat.sum(axis=1) > 0
        non_black = flat[non_black_mask]

        if len(non_black) < self.n_clusters:
            return []

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        kmeans.fit(non_black)

        # Count pixels per cluster
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        total = counts.sum()

        # Sort by frequency (descending)
        order = np.argsort(-counts)
        result = []
        for idx in order:
            center = kmeans.cluster_centers_[labels[idx]].astype(int).tolist()
            pct = round(float(counts[idx]) / total * 100, 2)
            result.append({
                "rgb": center,
                "hex": "#{:02x}{:02x}{:02x}".format(*center),
                "percentage": pct,
            })
        return result

    # ================================================================== #
    #  2. Repeating Pattern Detection (Autocorrelation)                    #
    # ================================================================== #

    def _detect_repeating_patterns(self, pixels):
        """
        Compute the normalized autocorrelation for each channel and detect
        significant peaks that indicate periodic/repeating color patterns.
        """
        results = {}
        channel_names = ["red", "green", "blue"]

        for ch_idx, ch_name in enumerate(channel_names):
            channel = pixels[:, :, ch_idx].astype(np.float64)

            # Flatten to 1D for row-major autocorrelation
            signal = channel.flatten()
            signal = signal - signal.mean()

            if signal.std() < 1e-6:
                results[ch_name] = {"has_periodicity": False, "periods": []}
                continue

            # Normalized autocorrelation via FFT
            autocorr = fftconvolve(signal, signal[::-1], mode="full")
            autocorr = autocorr[len(autocorr) // 2:]  # Keep positive lags only
            autocorr = autocorr / autocorr[0]  # Normalize

            # Find peaks: local maxima above threshold
            peaks = self._find_peaks(autocorr, threshold=0.3, min_distance=5)

            results[ch_name] = {
                "has_periodicity": len(peaks) > 0,
                "periods": [
                    {"lag": int(p), "strength": round(float(autocorr[p]), 4)}
                    for p in peaks[:5]  # Top 5 peaks
                ],
            }

        return results

    @staticmethod
    def _find_peaks(signal, threshold=0.3, min_distance=5):
        """Simple peak finder: local maxima above threshold."""
        peaks = []
        for i in range(min_distance, len(signal) - 1):
            if (
                signal[i] > threshold
                and signal[i] > signal[i - 1]
                and signal[i] > signal[i + 1]
            ):
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
        return peaks

    # ================================================================== #
    #  3. Anomaly Detection                                                #
    # ================================================================== #

    def _detect_anomalies(self, pixels):
        """
        Flag pixels whose Euclidean distance from the mean color exceeds
        `anomaly_sigma` standard deviations.

        Returns:
            anomaly_info: dict with count, percentage, threshold
            anomaly_mask: boolean array (H, W) — True for anomalous pixels
        """
        flat = pixels.reshape(-1, 3).astype(np.float64)

        # Ignore black (unfilled) pixels
        non_black_mask = flat.sum(axis=1) > 0
        if non_black_mask.sum() == 0:
            return {"count": 0, "percentage": 0.0, "threshold_sigma": self.anomaly_sigma}, \
                   np.zeros(pixels.shape[:2], dtype=bool)

        mean_color = flat[non_black_mask].mean(axis=0)
        distances = np.linalg.norm(flat - mean_color, axis=1)
        distances[~non_black_mask] = 0  # Don't flag black pixels

        threshold = distances[non_black_mask].mean() + self.anomaly_sigma * distances[non_black_mask].std()
        anomaly_flat = distances > threshold
        anomaly_mask = anomaly_flat.reshape(pixels.shape[:2])

        count = int(anomaly_mask.sum())
        total = int(non_black_mask.sum())
        pct = round(count / total * 100, 2) if total > 0 else 0.0

        return {
            "count": count,
            "percentage": pct,
            "threshold_sigma": self.anomaly_sigma,
            "mean_color_rgb": mean_color.astype(int).tolist(),
        }, anomaly_mask

    # ================================================================== #
    #  Composite visualization                                             #
    # ================================================================== #

    def _save_composite(self, pixels, anomaly_mask, dominant_colors, periodicity, output_path):
        """
        Create a 2×2 composite:
          Top-left:     Original image
          Top-right:    Anomaly heatmap overlay
          Bottom-left:  Dominant color palette bar
          Bottom-right: Channel intensity profiles
        """
        fig = plt.figure(figsize=(12, 10), facecolor="#1a1a2e")
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

        # --- Top-left: Original image ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(pixels)
        ax1.set_title("Original Log Image", color="white", fontsize=12, fontweight="bold")
        ax1.axis("off")

        # --- Top-right: Anomaly heatmap ---
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(pixels, alpha=0.5)
        if anomaly_mask.any():
            heatmap = np.zeros((*anomaly_mask.shape, 4))
            heatmap[anomaly_mask] = [1, 0, 0, 0.7]  # Red overlay
            ax2.imshow(heatmap)
        ax2.set_title("Anomaly Highlight (Red = Outlier)", color="white", fontsize=12, fontweight="bold")
        ax2.axis("off")

        # --- Bottom-left: Dominant color palette ---
        ax3 = fig.add_subplot(gs[1, 0])
        if dominant_colors:
            n = len(dominant_colors)
            palette = np.zeros((1, n, 3), dtype=np.uint8)
            labels = []
            for i, dc in enumerate(dominant_colors):
                palette[0, i] = dc["rgb"]
                labels.append(f'{dc["hex"]}\n{dc["percentage"]}%')
            ax3.imshow(palette, aspect="auto", extent=[0, n, 0, 1])
            for i, label in enumerate(labels):
                ax3.text(i + 0.5, -0.15, label, ha="center", va="top",
                         color="white", fontsize=8)
        ax3.set_title("Dominant Colors (K-Means)", color="white", fontsize=12, fontweight="bold")
        ax3.set_xlim(0, max(len(dominant_colors), 1))
        ax3.axis("off")

        # --- Bottom-right: Per-channel row averages ---
        ax4 = fig.add_subplot(gs[1, 1])
        for ch_idx, (ch_name, color) in enumerate(
            [("Red", "#ff6b6b"), ("Green", "#51cf66"), ("Blue", "#339af0")]
        ):
            row_means = pixels[:, :, ch_idx].mean(axis=1)
            ax4.plot(row_means, label=ch_name, color=color, linewidth=1.5, alpha=0.85)
        ax4.set_title("Channel Intensity by Row", color="white", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Row", color="white")
        ax4.set_ylabel("Mean Intensity", color="white")
        ax4.legend(fontsize=9)
        ax4.set_facecolor("#16213e")
        ax4.tick_params(colors="white")
        ax4.spines["bottom"].set_color("white")
        ax4.spines["left"].set_color("white")
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)

        fig.suptitle("Log Sonar — Image Analysis Report",
                     color="white", fontsize=16, fontweight="bold", y=0.98)

        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
