"""
painter.py — Multi-metric log ingest server.

Maps log events to RGB pixels using 3 independent metric dimensions:
  R = Error Severity    (status code → 0=healthy, 255=critical)
  G = Performance Health (latency → 255=fast, 0=slow)
  B = Resource Pressure  (CPU+memory → 0=idle, 255=maxed)

Feeds each pixel into the service-aware pipeline.
"""

from flask import Flask, request, jsonify
import sys
import os

# Ensure the parent directory is on the path so processLogic imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processLogic.pipeline import LogSonarPipeline


# ------------------------------------------------------------------ #
#  Multi-Metric Color Mapping                                         #
# ------------------------------------------------------------------ #

def get_color(status, latency, cpu_percent=0.0, memory_mb=0.0):
    """
    Map multiple metrics to an RGB tuple.

    R channel = Error Severity
        200     →  0   (healthy)
        3xx     →  50  (redirect, minor)
        4xx     → 160  (client error)
        500     → 255  (server error)
        502/503 → 230  (gateway error)

    G channel = Performance Health (inverted latency)
        0ms     → 255  (blazing fast)
        500ms   → 128  (moderate)
        1000ms+ →  0   (very slow)

    B channel = Resource Pressure (CPU + memory combined)
        idle (0% CPU, 0 MB)    →  0
        heavy (100% CPU, 1GB)  → 255
    """
    # --- R: Error Severity ---
    if status >= 500:
        r = 255
    elif status in (502, 503, 504):
        r = 230
    elif status >= 400:
        r = 160
    elif status >= 300:
        r = 50
    else:
        r = 0

    # --- G: Performance Health (inverted latency) ---
    # Clamp latency 0–1.5s → map to 255–0
    clamped_latency = min(max(latency, 0), 1.5)
    g = int(255 * (1 - clamped_latency / 1.5))

    # --- B: Resource Pressure ---
    # CPU: 0–100 → 0–1.0, Memory: 0–1024MB → 0–1.0
    cpu_norm = min(cpu_percent / 100.0, 1.0)
    mem_norm = min(memory_mb / 1024.0, 1.0)
    resource_pressure = (cpu_norm * 0.6 + mem_norm * 0.4)  # CPU weighted more
    b = int(255 * resource_pressure)

    return (
        max(0, min(255, r)),
        max(0, min(255, g)),
        max(0, min(255, b)),
    )


# ------------------------------------------------------------------ #
#  Flask app + Pipeline integration                                    #
# ------------------------------------------------------------------ #

app = Flask(__name__)

# Initialize the service-aware pipeline
pipeline = LogSonarPipeline(
    width=50,
    rows_per_service=16,
    services=["toy-service-1", "toy-service-2", "toy-service-3"],
)


@app.route('/ingest', methods=['POST'])
def ingest_log():
    data = request.json

    # 1. Extract all metrics
    service = data.get('service', 'unknown')
    status = data.get('status', 200)
    latency = data.get('latency', 0.0)
    cpu = data.get('cpu_percent', 0.0)
    memory = data.get('memory_mb', 0.0)

    # 2. Map metrics → RGB color
    color = get_color(status, latency, cpu, memory)

    # 3. Feed into service-aware pipeline
    pipeline.process_pixel(service, color)

    return jsonify({"status": "accepted"})


@app.route('/pipeline-status', methods=['GET'])
def pipeline_status():
    """Check per-service grid fill progress."""
    return jsonify(pipeline.get_status())


if __name__ == "__main__":
    print("Starting Log Painter v2 (Multi-Metric Ingest Server)...")
    print(f"  Color mapping: R=error, G=performance, B=resource_pressure")
    app.run(port=8000)
