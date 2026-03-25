"""
painter.py — Log ingest server that maps log events to RGB pixels.

Receives JSON log data via POST /ingest, determines the color based on
status/latency, and feeds each pixel into the LogSonarPipeline for
image generation and analysis.
"""

from flask import Flask, request, jsonify
import numpy as np
import time
import threading
import sys
import os

# Ensure the parent directory is on the path so processLogic imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processLogic.pipeline import LogSonarPipeline


# ------------------------------------------------------------------ #
#  Color mapping logic                                                #
# ------------------------------------------------------------------ #

def get_color(status, latency):
    """
    Map an HTTP status code + latency to an RGB tuple.

      500+  → Red        (server error)
      400+  → Yellow     (client error)
      200   → Green→White gradient based on latency
    """
    if status >= 500:
        return (255, 0, 0)      # Pure Red
    if status >= 400:
        return (255, 255, 0)    # Pure Yellow

    # 200-range: green that fades to white with higher latency
    base_r, base_g, base_b = (0, 255, 0)
    factor = min(max(latency - 0.5, 0), 1.0)

    r = int(base_r + (255 - base_r) * factor)
    g = int(base_g + (255 - base_g) * factor)
    b = int(base_b + (255 - base_b) * factor)

    return (r, g, b)


# ------------------------------------------------------------------ #
#  Service → column mapping                                           #
# ------------------------------------------------------------------ #

SERVICE_MAP = {}
NEXT_COL = 0
WIDTH = 100

def assign_column(service_name):
    """Assign a fixed column index to each unique service name."""
    global NEXT_COL
    if service_name not in SERVICE_MAP:
        SERVICE_MAP[service_name] = NEXT_COL
        NEXT_COL = (NEXT_COL + 10) % WIDTH
    return SERVICE_MAP[service_name]


# ------------------------------------------------------------------ #
#  Flask app + Pipeline integration                                    #
# ------------------------------------------------------------------ #

app = Flask(__name__)

# Initialize the image generation + analysis pipeline
pipeline = LogSonarPipeline(width=50, height=50)

ROW_BUFFER = {}


@app.route('/ingest', methods=['POST'])
def ingest_log():
    data = request.json

    # 1. Extract info
    service = data.get('service', 'unknown')
    status = data.get('status', 200)
    latency = data.get('latency', 0.0)

    # 2. Determine position (X-axis)
    col_idx = assign_column(service)

    # 3. Determine color (the pixel)
    color = get_color(status, latency)

    # 4. Paint to legacy buffer
    ROW_BUFFER[col_idx] = color

    # 5. Feed into the image pipeline
    pipeline.process_pixel(color)

    print(f"Painted [Service: {service}] at [Col: {col_idx}] with Color: {color}")

    return jsonify({"status": "accepted"})


@app.route('/pipeline-status', methods=['GET'])
def pipeline_status():
    """Check how far along the current pixel grid is."""
    return jsonify(pipeline.get_status())


if __name__ == "__main__":
    print("Starting Log Painter (Ingest Server)...")
    app.run(port=8000)
