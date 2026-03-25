"""
test_pipeline.py — Standalone test that validates the full pipeline
without needing the Flask servers running.

Generates synthetic log data, feeds 2500 pixels (50×50 grid) through the
pipeline, and verifies that images + analysis reports are created correctly.

Run from the logSonar directory:
    python -m processLogic.test_pipeline
"""

import os
import sys
import json
import random
import shutil
import time

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processLogic.pixel_buffer import PixelBuffer
from processLogic.image_generator import ImageGenerator
from processLogic.image_analyzer import ImageAnalyzer
from processLogic.pipeline import LogSonarPipeline


# ------------------------------------------------------------------ #
#  Test helpers                                                       #
# ------------------------------------------------------------------ #

def generate_test_color(outcome):
    """Simulate the painter's get_color logic."""
    if outcome < 0.1:
        return (255, 0, 0)       # 500 error → red
    elif outcome < 0.3:
        latency = random.uniform(1.0, 1.7)
        factor = min(max(latency - 0.5, 0), 1.0)
        r = int(0 + 255 * factor)
        g = int(255)
        b = int(0 + 255 * factor)
        return (r, g, b)         # High latency → warm green/white
    else:
        latency = random.uniform(0.01, 0.5)
        return (0, 255, 0)       # Normal → bright green


def test_pixel_buffer():
    """Test that the pixel buffer correctly fires a callback on fill."""
    print("\n" + "=" * 60)
    print("TEST 1: PixelBuffer — grid fill callback")
    print("=" * 60)

    results = []

    def on_full(grid):
        results.append(grid)

    buf = PixelBuffer(width=10, height=10, on_grid_full=on_full)

    # Feed exactly 100 pixels (10×10)
    for i in range(100):
        buf.add_pixel((i % 256, (i * 2) % 256, (i * 3) % 256))

    assert len(results) == 1, f"Expected 1 callback, got {len(results)}"
    assert results[0].shape == (10, 10, 3), f"Unexpected shape: {results[0].shape}"
    print("✅ PixelBuffer callback fired correctly with shape (10, 10, 3)")

    # After reset, fill pct should be 0
    assert buf.get_fill_percentage() == 0.0
    print("✅ Buffer reset after callback")


def test_image_generator():
    """Test that the image generator creates a valid PNG."""
    print("\n" + "=" * 60)
    print("TEST 2: ImageGenerator — PNG creation")
    print("=" * 60)

    import numpy as np

    test_dir = "/tmp/logsonar_test_gen"
    os.makedirs(test_dir, exist_ok=True)

    gen = ImageGenerator(output_dir=test_dir)

    # Create a gradient image
    grid = np.zeros((50, 50, 3), dtype=np.uint8)
    for r in range(50):
        for c in range(50):
            grid[r, c] = (r * 5, c * 5, 128)

    path = gen.save(grid, filename="test_gradient.png")
    assert os.path.exists(path), f"Image not found at {path}"
    size = os.path.getsize(path)
    assert size > 0, "Image file is empty"
    print(f"✅ Generated image: {path} ({size} bytes)")

    # Cleanup
    shutil.rmtree(test_dir)
    print("✅ Cleaned up test directory")


def test_image_analyzer():
    """Test the analyzer on a synthetic image."""
    print("\n" + "=" * 60)
    print("TEST 3: ImageAnalyzer — analysis report")
    print("=" * 60)

    import numpy as np
    from PIL import Image

    test_img_dir = "/tmp/logsonar_test_img"
    test_analysis_dir = "/tmp/logsonar_test_analysis"
    os.makedirs(test_img_dir, exist_ok=True)

    # Create an image with deliberate patterns:
    # - Mostly green (normal)
    # - Red stripes every 10 rows (periodic errors)
    # - A few yellow pixels (anomalies)
    grid = np.zeros((50, 50, 3), dtype=np.uint8)
    for r in range(50):
        for c in range(50):
            if r % 10 == 0:
                grid[r, c] = (255, 0, 0)       # Red stripe
            elif r == 25 and c == 25:
                grid[r, c] = (255, 255, 0)      # Yellow anomaly
            else:
                grid[r, c] = (0, 255, 0)        # Normal green

    img_path = os.path.join(test_img_dir, "test_pattern.png")
    Image.fromarray(grid).save(img_path)

    analyzer = ImageAnalyzer(output_dir=test_analysis_dir)
    report = analyzer.analyze(img_path)

    # Validate report
    assert "dominant_colors" in report, "Missing dominant_colors"
    assert "periodicity" in report, "Missing periodicity"
    assert "anomalies" in report, "Missing anomalies"
    assert len(report["dominant_colors"]) > 0, "No dominant colors found"

    print(f"✅ Report generated with {len(report['dominant_colors'])} dominant colors")
    print(f"   Top color: {report['dominant_colors'][0]['hex']} "
          f"({report['dominant_colors'][0]['percentage']}%)")
    print(f"   Anomalies: {report['anomalies']['count']} "
          f"({report['anomalies']['percentage']}%)")

    # Check that red channel shows periodicity
    red_periodic = report["periodicity"].get("red", {})
    print(f"   Red periodicity: {red_periodic.get('has_periodicity', False)}")

    # Check output files
    assert os.path.exists(report["json_path"]), "JSON report not found"
    assert os.path.exists(report["composite_path"]), "Composite image not found"
    print(f"✅ JSON report: {report['json_path']}")
    print(f"✅ Composite: {report['composite_path']}")

    # Cleanup
    shutil.rmtree(test_img_dir)
    shutil.rmtree(test_analysis_dir)
    print("✅ Cleaned up test directories")


def test_full_pipeline():
    """End-to-end test: synthetic logs → pipeline → image + analysis."""
    print("\n" + "=" * 60)
    print("TEST 4: Full Pipeline — end-to-end")
    print("=" * 60)

    gen_dir = "/tmp/logsonar_test_e2e_gen"
    analysis_dir = "/tmp/logsonar_test_e2e_analysis"

    pipeline = LogSonarPipeline(
        width=50, height=50,
        generated_dir=gen_dir,
        analysis_dir=analysis_dir,
    )

    # Feed 2500 pixels (exactly fills one 50×50 grid)
    random.seed(42)
    for i in range(2500):
        outcome = random.random()
        color = generate_test_color(outcome)
        pipeline.process_pixel(color)

    # Give a moment for any async processing
    time.sleep(0.5)

    # Verify status
    status = pipeline.get_status()
    print(f"   Pipeline status: {status}")
    assert status["images_generated"] >= 1, f"Expected ≥1 image, got {status['images_generated']}"

    # Verify files exist
    gen_files = os.listdir(gen_dir)
    analysis_files = os.listdir(analysis_dir)
    print(f"   Generated images: {gen_files}")
    print(f"   Analysis outputs: {analysis_files}")

    assert len(gen_files) >= 1, "No generated images found"
    assert any(f.endswith("_report.json") for f in analysis_files), "No JSON report found"
    assert any(f.endswith("_analysis.png") for f in analysis_files), "No composite image found"

    # Read and validate a JSON report
    json_files = [f for f in analysis_files if f.endswith("_report.json")]
    with open(os.path.join(analysis_dir, json_files[0])) as f:
        report = json.load(f)
    print(f"\n   📊 Analysis Report Summary:")
    print(f"   Image: {report['image_path']}")
    print(f"   Size: {report['image_size']}")
    print(f"   Dominant colors: {len(report['dominant_colors'])}")
    for dc in report["dominant_colors"][:3]:
        print(f"     {dc['hex']} — {dc['percentage']}%")
    print(f"   Anomalies: {report['anomalies']['count']} ({report['anomalies']['percentage']}%)")
    print(f"   Red periodicity: {report['periodicity']['red']['has_periodicity']}")

    print("\n✅ Full pipeline test passed!")

    # Cleanup
    shutil.rmtree(gen_dir)
    shutil.rmtree(analysis_dir)
    print("✅ Cleaned up test directories")


# ------------------------------------------------------------------ #
#  Run all tests                                                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("🔬 Log Sonar Pipeline Tests")
    print("=" * 60)

    test_pixel_buffer()
    test_image_generator()
    test_image_analyzer()
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
