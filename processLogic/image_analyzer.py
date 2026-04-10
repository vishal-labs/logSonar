"""
image_analyzer.py — Service-aware image analysis with per-service and cross-service reports.

Techniques applied per-service band:
  1. Dominant Color Extraction  (K-means clustering)
  2. Repeating Pattern Detection (autocorrelation via FFT)
  3. Anomaly Detection           (statistical outlier highlighting)

Additional cross-service analysis:
  4. Comparative error rates and resource pressure
  5. Pattern correlation across services

Output:
  - JSON report with per-service and cross-service sections
  - Composite visualization with per-service panels
"""

import json
import os
from datetime import datetime

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from sklearn.cluster import KMeans
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "analysis_output",
)


class ImageAnalyzer:
    def __init__(self, output_dir=None, n_clusters=5, anomaly_sigma=2.0):
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self.n_clusters = n_clusters
        self.anomaly_sigma = anomaly_sigma
        os.makedirs(self.output_dir, exist_ok=True)

    # ================================================================== #
    #  Main entry point                                                    #
    # ================================================================== #

    def analyze(self, image_path, service_layout=None):
        """
        Run analysis on a composite service image.

        Args:
            image_path:     Path to the input PNG.
            service_layout: Dict mapping service_name → {start_row, end_row, rows}.
                            If None, analyzes the whole image as one block.

        Returns:
            Full report dict with per_service and cross_service sections.
        """
        img = Image.open(image_path).convert("RGB")
        pixels = np.array(img, dtype=np.uint8)

        report = {
            "image_path": os.path.basename(image_path),
            "analyzed_at": datetime.now().isoformat(),
            "image_size": {"width": pixels.shape[1], "height": pixels.shape[0]},
        }

        if service_layout:
            # --- Per-service analysis ---
            per_service = {}
            for svc_name, layout in service_layout.items():
                start = layout["start_row"]
                end = layout["end_row"] + 1  # inclusive → exclusive
                band = pixels[start:end, :, :]
                per_service[svc_name] = self._analyze_band(band, svc_name)

            report["per_service"] = per_service

            # --- Cross-service comparison ---
            report["cross_service"] = self._cross_service_comparison(per_service)
        else:
            # Fallback: analyze whole image as single block
            report["whole_image"] = self._analyze_band(pixels, "full_image")

        # --- Save outputs ---
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        json_path = os.path.join(self.output_dir, f"{base_name}_report.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        composite_path = os.path.join(self.output_dir, f"{base_name}_analysis.png")
        self._save_composite(pixels, report, service_layout, composite_path)

        print(f"[ImageAnalyzer] Report  → {json_path}")
        print(f"[ImageAnalyzer] Visual  → {composite_path}")

        report["json_path"] = json_path
        report["composite_path"] = composite_path
        return report

    # ================================================================== #
    #  Per-band analysis                                                   #
    # ================================================================== #

    def _analyze_band(self, pixels, label=""):
        """Run all 3 analysis techniques on a single image band."""
        dominant_colors = self._extract_dominant_colors(pixels)
        periodicity, fft_spectra = self._detect_repeating_patterns(pixels)
        anomaly_info, anomaly_mask = self._detect_anomalies(pixels)
        channel_stats = self._channel_statistics(pixels)

        return {
            "label": label,
            "size": {"height": pixels.shape[0], "width": pixels.shape[1]},
            "dominant_colors": dominant_colors,
            "periodicity": periodicity,
            "anomalies": anomaly_info,
            "channel_stats": channel_stats,
            "_anomaly_mask": anomaly_mask.tolist(),
            "_fft_spectra": fft_spectra,  # for frequency domain visualization
        }

    def _channel_statistics(self, pixels):
        """Compute per-channel mean, std, min, max (semantic interpretation)."""
        stats = {}
        names = [
            ("red", "error_severity"),
            ("green", "performance_health"),
            ("blue", "resource_pressure"),
        ]
        flat = pixels.reshape(-1, 3).astype(np.float64)
        non_black = flat[flat.sum(axis=1) > 0]

        if len(non_black) == 0:
            return {n[1]: {"mean": 0, "std": 0} for n in names}

        for ch_idx, (ch_name, semantic) in enumerate(names):
            vals = non_black[:, ch_idx]
            stats[semantic] = {
                "channel": ch_name,
                "mean": round(float(vals.mean()), 1),
                "std": round(float(vals.std()), 1),
                "min": int(vals.min()),
                "max": int(vals.max()),
            }
        return stats

    # ================================================================== #
    #  Cross-service comparison                                            #
    # ================================================================== #

    def _cross_service_comparison(self, per_service):
        """Compare metrics across services to find relative health."""
        comparison = {
            "services": list(per_service.keys()),
            "error_severity_ranking": [],
            "performance_ranking": [],
            "resource_ranking": [],
            "anomaly_ranking": [],
        }

        # Build sortable lists
        svc_metrics = []
        for svc, data in per_service.items():
            stats = data.get("channel_stats", {})
            err = stats.get("error_severity", {}).get("mean", 0)
            perf = stats.get("performance_health", {}).get("mean", 0)
            res = stats.get("resource_pressure", {}).get("mean", 0)
            anomaly_pct = data.get("anomalies", {}).get("percentage", 0)
            svc_metrics.append({
                "service": svc,
                "error_severity_mean": err,
                "performance_health_mean": perf,
                "resource_pressure_mean": res,
                "anomaly_percentage": anomaly_pct,
            })

        # Rank by error severity (highest = worst)
        comparison["error_severity_ranking"] = sorted(
            svc_metrics, key=lambda x: x["error_severity_mean"], reverse=True
        )
        # Rank by performance (lowest green = worst)
        comparison["performance_ranking"] = sorted(
            svc_metrics, key=lambda x: x["performance_health_mean"]
        )
        # Rank by resource pressure (highest = most stressed)
        comparison["resource_ranking"] = sorted(
            svc_metrics, key=lambda x: x["resource_pressure_mean"], reverse=True
        )
        # Rank by anomalies (highest = most anomalous)
        comparison["anomaly_ranking"] = sorted(
            svc_metrics, key=lambda x: x["anomaly_percentage"], reverse=True
        )

        return comparison

    # ================================================================== #
    #  1. Dominant Color Extraction (K-Means)                              #
    # ================================================================== #

    def _extract_dominant_colors(self, pixels):
        flat = pixels.reshape(-1, 3).astype(np.float64)
        non_black_mask = flat.sum(axis=1) > 0
        non_black = flat[non_black_mask]

        if len(non_black) < self.n_clusters:
            return []

        n_clusters = min(self.n_clusters, len(np.unique(non_black, axis=0)))
        if n_clusters < 2:
            return []

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(non_black)

        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        total = counts.sum()

        order = np.argsort(-counts)
        result = []
        for idx in order:
            center = kmeans.cluster_centers_[labels[idx]].astype(int).tolist()
            pct = round(float(counts[idx]) / total * 100, 2)
            result.append({
                "rgb": center,
                "hex": "#{:02x}{:02x}{:02x}".format(*center),
                "percentage": pct,
                "interpretation": self._interpret_color(center),
            })
        return result

    @staticmethod
    def _interpret_color(rgb):
        """Give a semantic interpretation of an RGB pixel."""
        r, g, b = rgb
        parts = []
        if r > 150:
            parts.append("high-error")
        elif r > 50:
            parts.append("moderate-error")
        else:
            parts.append("healthy")

        if g > 180:
            parts.append("fast")
        elif g > 80:
            parts.append("moderate-speed")
        else:
            parts.append("slow")

        if b > 150:
            parts.append("heavy-resources")
        elif b > 50:
            parts.append("moderate-resources")
        else:
            parts.append("light-resources")

        return ", ".join(parts)

    # ================================================================== #
    #  2. Repeating Pattern Detection (Autocorrelation)                    #
    # ================================================================== #

    def _detect_repeating_patterns(self, pixels):
        """
        Compute autocorrelation and FFT power spectrum for each channel.

        Returns:
            results:      Dict with periodicity info per channel.
            fft_spectra:  Dict with frequency and magnitude arrays per channel
                          (for plotting the frequency domain).
        """
        results = {}
        fft_spectra = {}  # Store for visualization
        channel_names = ["red", "green", "blue"]

        for ch_idx, ch_name in enumerate(channel_names):
            channel = pixels[:, :, ch_idx].astype(np.float64)
            signal = channel.flatten()
            signal = signal - signal.mean()

            if signal.std() < 1e-6:
                results[ch_name] = {"has_periodicity": False, "periods": []}
                fft_spectra[ch_name] = {"frequencies": [], "magnitudes": []}
                continue

            # --- Autocorrelation (for periodicity detection) ---
            autocorr = fftconvolve(signal, signal[::-1], mode="full")
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / autocorr[0]

            peaks = self._find_peaks(autocorr, threshold=0.2, min_distance=5)

            results[ch_name] = {
                "has_periodicity": len(peaks) > 0,
                "periods": [
                    {"lag": int(p), "strength": round(float(autocorr[p]), 4)}
                    for p in peaks[:5]
                ],
            }

            # --- FFT Power Spectrum (frequency domain) ---
            n = len(signal)
            fft_vals = np.fft.rfft(signal)
            magnitudes = np.abs(fft_vals) / n  # Normalized magnitude
            frequencies = np.fft.rfftfreq(n)   # Normalized frequency (0 to 0.5)

            # Skip DC component (index 0) for cleaner visualization
            fft_spectra[ch_name] = {
                "frequencies": frequencies[1:].tolist(),
                "magnitudes": magnitudes[1:].tolist(),
            }

        return results, fft_spectra

    @staticmethod
    def _find_peaks(signal, threshold=0.2, min_distance=5):
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
        flat = pixels.reshape(-1, 3).astype(np.float64)
        non_black_mask = flat.sum(axis=1) > 0

        if non_black_mask.sum() == 0:
            return {"count": 0, "percentage": 0.0}, np.zeros(pixels.shape[:2], dtype=bool)

        mean_color = flat[non_black_mask].mean(axis=0)
        distances = np.linalg.norm(flat - mean_color, axis=1)
        distances[~non_black_mask] = 0

        threshold = (
            distances[non_black_mask].mean()
            + self.anomaly_sigma * distances[non_black_mask].std()
        )
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
    #  Composite Visualization                                             #
    # ================================================================== #

    def _save_composite(self, pixels, report, service_layout, output_path):
        """
        Multi-panel visualization:
          Col 0: Original service band
          Col 1: Anomaly heatmap overlay
          Col 2: Channel intensity (time domain)
          Col 3: FFT Power Spectrum (frequency domain)
          Bottom: Cross-service comparison bar chart
        """
        services = list(service_layout.keys()) if service_layout else ["full"]
        n_services = len(services)

        fig = plt.figure(figsize=(22, 4 * n_services + 2), facecolor="#1a1a2e")

        # Layout: n_services rows × 4 columns + 1 bottom row spanning all
        gs = gridspec.GridSpec(
            n_services + 1, 4,
            hspace=0.4, wspace=0.3,
            height_ratios=[1] * n_services + [1.2],
        )

        channel_colors = [
            ("Error (R)", "#ff6b6b"),
            ("Perf (G)", "#51cf66"),
            ("Resource (B)", "#339af0"),
        ]

        for i, svc in enumerate(services):
            layout = service_layout[svc]
            start = layout["start_row"]
            end = layout["end_row"] + 1
            band = pixels[start:end, :, :]

            svc_report = report.get("per_service", {}).get(svc, {})
            anomaly_mask = np.array(svc_report.get("_anomaly_mask", []), dtype=bool)
            if anomaly_mask.size == 0:
                anomaly_mask = np.zeros(band.shape[:2], dtype=bool)

            # --- Col 0: Original band ---
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.imshow(band, aspect="auto", interpolation="nearest")
            ax1.set_title(f"{svc}", color="white", fontsize=11, fontweight="bold")
            ax1.set_ylabel("Row", color="white", fontsize=8)
            ax1.tick_params(colors="white", labelsize=7)

            # --- Col 1: Anomaly overlay ---
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow(band, aspect="auto", interpolation="nearest", alpha=0.5)
            if anomaly_mask.any():
                heatmap = np.zeros((*anomaly_mask.shape, 4))
                heatmap[anomaly_mask] = [1, 0, 0, 0.8]
                ax2.imshow(heatmap, aspect="auto", interpolation="nearest")
            anomaly_count = svc_report.get("anomalies", {}).get("count", 0)
            ax2.set_title(f"Anomalies: {anomaly_count}", color="white", fontsize=10)
            ax2.tick_params(colors="white", labelsize=7)

            # --- Col 2: Channel intensity (time domain) ---
            ax3 = fig.add_subplot(gs[i, 2])
            for ch_idx, (ch_label, color) in enumerate(channel_colors):
                row_means = band[:, :, ch_idx].mean(axis=1)
                ax3.plot(row_means, label=ch_label, color=color, linewidth=1.2, alpha=0.85)
            ax3.set_title("Channel Intensity", color="white", fontsize=10)
            ax3.legend(fontsize=7, loc="upper right")
            ax3.set_facecolor("#16213e")
            ax3.tick_params(colors="white", labelsize=7)
            ax3.spines["bottom"].set_color("white")
            ax3.spines["left"].set_color("white")
            ax3.spines["top"].set_visible(False)
            ax3.spines["right"].set_visible(False)

            # --- Col 3: FFT Power Spectrum (frequency domain) ---
            ax4 = fig.add_subplot(gs[i, 3])
            fft_data = svc_report.get("_fft_spectra", {})
            channel_names = ["red", "green", "blue"]
            has_data = False
            for ch_idx, (ch_label, color) in enumerate(channel_colors):
                ch_name = channel_names[ch_idx]
                spectrum = fft_data.get(ch_name, {})
                freqs = spectrum.get("frequencies", [])
                mags = spectrum.get("magnitudes", [])
                if freqs and mags:
                    has_data = True
                    ax4.plot(freqs, mags, label=ch_label, color=color,
                             linewidth=1.0, alpha=0.85)
            ax4.set_title("FFT Power Spectrum", color="white", fontsize=10)
            ax4.set_xlabel("Frequency", color="white", fontsize=8)
            ax4.set_ylabel("Magnitude", color="white", fontsize=8)
            if has_data:
                ax4.legend(fontsize=7, loc="upper right")
            ax4.set_facecolor("#16213e")
            ax4.tick_params(colors="white", labelsize=7)
            ax4.spines["bottom"].set_color("white")
            ax4.spines["left"].set_color("white")
            ax4.spines["top"].set_visible(False)
            ax4.spines["right"].set_visible(False)

        # --- Bottom row: Cross-service comparison bar chart ---
        cross = report.get("cross_service", {})
        if cross:
            ax_bar = fig.add_subplot(gs[n_services, :])
            svc_names = [m["service"] for m in cross.get("error_severity_ranking", [])]
            if svc_names:
                x = np.arange(len(svc_names))
                bar_width = 0.25

                err_vals = [m["error_severity_mean"] for m in cross["error_severity_ranking"]]
                perf_vals = [
                    next(m["performance_health_mean"] for m in cross["performance_ranking"]
                         if m["service"] == s)
                    for s in svc_names
                ]
                res_vals = [
                    next(m["resource_pressure_mean"] for m in cross["resource_ranking"]
                         if m["service"] == s)
                    for s in svc_names
                ]

                ax_bar.bar(x - bar_width, err_vals, bar_width, label="Error Severity",
                           color="#ff6b6b", alpha=0.85)
                ax_bar.bar(x, perf_vals, bar_width, label="Performance",
                           color="#51cf66", alpha=0.85)
                ax_bar.bar(x + bar_width, res_vals, bar_width, label="Resource Pressure",
                           color="#339af0", alpha=0.85)

                ax_bar.set_xticks(x)
                ax_bar.set_xticklabels(svc_names, fontsize=9, color="white")
                ax_bar.set_title("Cross-Service Comparison (Mean Channel Values)",
                                 color="white", fontsize=12, fontweight="bold")
                ax_bar.legend(fontsize=9)
                ax_bar.set_facecolor("#16213e")
                ax_bar.tick_params(colors="white")
                ax_bar.spines["bottom"].set_color("white")
                ax_bar.spines["left"].set_color("white")
                ax_bar.spines["top"].set_visible(False)
                ax_bar.spines["right"].set_visible(False)

        fig.suptitle("Log Sonar v2 — Per-Service Analysis",
                     color="white", fontsize=16, fontweight="bold", y=0.99)

        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
