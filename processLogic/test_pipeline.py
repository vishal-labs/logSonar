"""
test_pipeline.py — Tests for the v2 service-aware multi-metric pipeline.

Run from the logSonar directory:
    python -m processLogic.test_pipeline
"""

import os
import sys
import json
import random
import shutil
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processLogic.pixel_buffer import ServiceAwareBuffer
from processLogic.image_generator import ImageGenerator
from processLogic.image_analyzer import ImageAnalyzer
from processLogic.pipeline import LogSonarPipeline


SERVICES = ["toy-service-1", "toy-service-2", "toy-service-3"]


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def make_color(status, latency, cpu, memory):
    """Replicate painter's get_color logic for testing."""
    # R: error severity
    if status >= 500:
        r = 255
    elif status >= 400:
        r = 160
    else:
        r = 0

    # G: performance (inverted latency)
    g = int(255 * (1 - min(max(latency, 0), 1.5) / 1.5))

    # B: resource pressure
    cpu_norm = min(cpu / 100.0, 1.0)
    mem_norm = min(memory / 1024.0, 1.0)
    b = int(255 * (cpu_norm * 0.6 + mem_norm * 0.4))

    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def generate_service_pixel(service):
    """Generate a realistic pixel for a specific service personality."""
    if service == "toy-service-1":  # Stable API
        status = 500 if random.random() < 0.03 else 200
        latency = max(0.001, 0.05 + random.gauss(0, 0.02))
        cpu = random.uniform(5, 20)
        memory = random.uniform(80, 150)
    elif service == "toy-service-2":  # Heavy Worker
        status = 500 if random.random() < 0.08 else 200
        latency = max(0.001, 0.3 + random.gauss(0, 0.15))
        cpu = random.uniform(50, 90)
        memory = random.uniform(300, 600)
    else:  # Flaky Gateway
        status = random.choice([502, 503, 504]) if random.random() < 0.15 else 200
        latency = max(0.001, 0.2 + random.gauss(0, 0.4))
        cpu = random.uniform(15, 45)
        memory = random.uniform(120, 250)

    return make_color(status, latency, cpu, memory)


# ------------------------------------------------------------------ #
#  Tests                                                              #
# ------------------------------------------------------------------ #

def test_service_aware_buffer():
    """Test that the buffer creates per-service bands and fires callback."""
    print("\n" + "=" * 60)
    print("TEST 1: ServiceAwareBuffer — per-service bands")
    print("=" * 60)

    results = []

    def on_full(grid, layout):
        results.append((grid, layout))

    buf = ServiceAwareBuffer(
        width=10, rows_per_service=5, services=SERVICES, on_grid_full=on_full
    )

    # Total height = 5*3 + 2 dividers = 17
    assert buf.total_height == 17, f"Expected height 17, got {buf.total_height}"
    print(f"✅ Grid dimensions: {buf.total_height}×{buf.width}")

    # Fill all 3 services (10*5 = 50 pixels each)
    for svc in SERVICES:
        for _ in range(50):
            color = generate_service_pixel(svc)
            buf.add_pixel(svc, color)

    assert len(results) == 1, f"Expected 1 callback, got {len(results)}"
    grid, layout = results[0]
    assert grid.shape == (17, 10, 3), f"Expected (17,10,3), got {grid.shape}"
    assert all(svc in layout for svc in SERVICES)

    # Verify divider rows are gray
    assert tuple(grid[5, 0]) == (80, 80, 80), "Divider row 5 should be gray"
    assert tuple(grid[11, 0]) == (80, 80, 80), "Divider row 11 should be gray"
    print("✅ Callback fired with correct grid shape and gray dividers")
    print(f"   Layout: {layout}")


def test_image_generator():
    """Test PNG generation for the taller composite image."""
    print("\n" + "=" * 60)
    print("TEST 2: ImageGenerator — composite image")
    print("=" * 60)

    import numpy as np

    test_dir = "/tmp/logsonar_v2_test_gen"
    os.makedirs(test_dir, exist_ok=True)

    gen = ImageGenerator(output_dir=test_dir)
    grid = np.zeros((50, 50, 3), dtype=np.uint8)

    # Simulate 3 service bands with different colors
    grid[0:16, :] = (0, 200, 30)    # Service 1: mostly green (healthy)
    grid[16, :] = (80, 80, 80)      # Divider
    grid[17:33, :] = (100, 150, 180) # Service 2: blueish (resource heavy)
    grid[33, :] = (80, 80, 80)      # Divider
    grid[34:50, :] = (200, 100, 50)  # Service 3: reddish (errors)

    path = gen.save(grid, filename="test_composite.png")
    assert os.path.exists(path)
    print(f"✅ Composite image saved: {path} ({os.path.getsize(path)} bytes)")

    shutil.rmtree(test_dir)
    print("✅ Cleaned up")


def test_per_service_analyzer():
    """Test per-service analysis on a synthetic image."""
    print("\n" + "=" * 60)
    print("TEST 3: ImageAnalyzer — per-service analysis")
    print("=" * 60)

    import numpy as np
    from PIL import Image

    test_img_dir = "/tmp/logsonar_v2_test_img"
    test_out_dir = "/tmp/logsonar_v2_test_analysis"
    os.makedirs(test_img_dir, exist_ok=True)

    # Build a 50-row image (16+1+16+1+16)
    grid = np.zeros((50, 50, 3), dtype=np.uint8)

    # Service 1: mostly green (healthy, fast, low resources)
    for r in range(16):
        for c in range(50):
            grid[r, c] = (0, 230, 20)
    # Inject a few red anomalies
    grid[5, 25] = (255, 50, 10)
    grid[10, 30] = (255, 50, 10)

    grid[16, :] = (80, 80, 80)  # Divider

    # Service 2: mixed blue (high resources), with periodic red spikes
    for r in range(17, 33):
        for c in range(50):
            if c % 10 == 0:
                grid[r, c] = (255, 80, 200)  # Periodic error+resource spike
            else:
                grid[r, c] = (30, 170, 180)   # Normal: moderate perf, high resource

    grid[33, :] = (80, 80, 80)  # Divider

    # Service 3: erratic red/orange (many errors, variable)
    for r in range(34, 50):
        for c in range(50):
            if random.random() < 0.15:
                grid[r, c] = (230, 60, 80)   # Error
            else:
                grid[r, c] = (50, 180, 60)    # OK

    img_path = os.path.join(test_img_dir, "test_services.png")
    Image.fromarray(grid).save(img_path)

    service_layout = {
        "toy-service-1": {"start_row": 0, "end_row": 15, "rows": 16},
        "toy-service-2": {"start_row": 17, "end_row": 32, "rows": 16},
        "toy-service-3": {"start_row": 34, "end_row": 49, "rows": 16},
    }

    analyzer = ImageAnalyzer(output_dir=test_out_dir)
    report = analyzer.analyze(img_path, service_layout=service_layout)

    # Validate structure
    assert "per_service" in report
    assert "cross_service" in report
    assert len(report["per_service"]) == 3

    for svc in SERVICES:
        svc_data = report["per_service"][svc]
        assert "dominant_colors" in svc_data
        assert "periodicity" in svc_data
        assert "anomalies" in svc_data
        assert "channel_stats" in svc_data
        print(f"  {svc}:")
        print(f"    Dominant: {svc_data['dominant_colors'][0]['hex'] if svc_data['dominant_colors'] else 'N/A'} "
              f"({svc_data['dominant_colors'][0].get('interpretation', '') if svc_data['dominant_colors'] else ''})")
        print(f"    Anomalies: {svc_data['anomalies']['count']}")
        stats = svc_data["channel_stats"]
        for metric, vals in stats.items():
            print(f"    {metric}: mean={vals['mean']}")

    cross = report["cross_service"]
    print(f"\n  Cross-service error ranking: "
          f"{[m['service'] for m in cross['error_severity_ranking']]}")

    assert os.path.exists(report["json_path"])
    assert os.path.exists(report["composite_path"])
    print(f"\n✅ Per-service analysis complete")

    shutil.rmtree(test_img_dir)
    shutil.rmtree(test_out_dir)
    print("✅ Cleaned up")


def test_full_pipeline_v2():
    """End-to-end: 3 services → service-aware pipeline → image + analysis."""
    print("\n" + "=" * 60)
    print("TEST 4: Full Pipeline v2 — end-to-end")
    print("=" * 60)

    gen_dir = "/tmp/logsonar_v2_e2e_gen"
    analysis_dir = "/tmp/logsonar_v2_e2e_analysis"

    pipeline = LogSonarPipeline(
        width=50,
        rows_per_service=16,
        services=SERVICES,
        generated_dir=gen_dir,
        analysis_dir=analysis_dir,
    )

    # Each service needs 16*50=800 pixels, total 2400
    random.seed(42)
    for _ in range(800):
        for svc in SERVICES:
            color = generate_service_pixel(svc)
            pipeline.process_pixel(svc, color)

    time.sleep(0.5)

    status = pipeline.get_status()
    print(f"   Status: {json.dumps(status, indent=2)}")

    assert status["images_generated"] >= 1

    gen_files = os.listdir(gen_dir)
    analysis_files = os.listdir(analysis_dir)
    assert len(gen_files) >= 1
    assert any(f.endswith("_report.json") for f in analysis_files)

    # Read report
    json_files = [f for f in analysis_files if f.endswith("_report.json")]
    with open(os.path.join(analysis_dir, json_files[0])) as f:
        report = json.load(f)

    print(f"\n   📊 Report Summary:")
    for svc in SERVICES:
        svc_data = report["per_service"][svc]
        top = svc_data["dominant_colors"][0] if svc_data["dominant_colors"] else {}
        print(f"   {svc}: top={top.get('hex','?')} "
              f"({top.get('interpretation','?')}), "
              f"anomalies={svc_data['anomalies']['count']}")

    cross = report["cross_service"]
    print(f"\n   Error ranking: {[m['service'] for m in cross['error_severity_ranking']]}")
    print(f"   Resource ranking: {[m['service'] for m in cross['resource_ranking']]}")

    print("\n✅ Full pipeline v2 test passed!")

    shutil.rmtree(gen_dir)
    shutil.rmtree(analysis_dir)
    print("✅ Cleaned up")


# ------------------------------------------------------------------ #
#  Run all tests                                                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("🔬 Log Sonar v2 Pipeline Tests")
    print("=" * 60)

    test_service_aware_buffer()
    test_image_generator()
    test_per_service_analyzer()
    test_full_pipeline_v2()

    print("\n" + "=" * 60)
    print("🎉 ALL v2 TESTS PASSED!")
    print("=" * 60)
