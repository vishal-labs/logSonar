
from flask import Flask, request, jsonify
import numpy as np
import time
import threading

class RowAggregator:
    def __init__(self, width=100, flush_interval=0.1):
        self.width = width
        self.flush_interval = flush_interval
        # Initialize an empty row (Black)
        self.current_row = [(0, 0, 0)] * width
        self.lock = threading.Lock()
        
    def add_pixel(self, column_index, color):
        with self.lock:
            if 0 <= column_index < self.width:
                self.current_row[column_index] = color
                
    def start_flushing(self, callback_function):
        """
        Runs in background. Every 100ms, takes the current row,
        calls the callback (to send to frontend), and resets.
        """
        def loop():
            while True:
                time.sleep(self.flush_interval)
                with self.lock:
                    # 1. Snapshot the row
                    row_to_send = list(self.current_row)
                    # 2. Reset to Black
                    self.current_row = [(0, 0, 0)] * self.width
                
                # 3. Send it (Callback)
                callback_function(row_to_send)
                
        threading.Thread(target=loop, daemon=True).start()

# Usage Example (Add this to your main block tomorrow)
# aggregator = RowAggregator(width=100)
# aggregator.start_flushing(lambda row: print("Sending Row to Frontend!"))

app = Flask(__name__)

# --- CONFIGURATION ---
WIDTH = 100         # The 'width' of our Sonar (number of slots)
ROW_BUFFER = {}     # Stores pixels for the *current* frame: { column_index: (r, g, b) }
SERVICE_MAP = {}    # Maps "service_name" -> column_index (0 to WIDTH-1)
NEXT_COL = 0        # Tracker for assigning new services to columns

def get_color(status, latency):
    if status >= 500:
        return (255, 0, 0)  # Pure Red
    if status >= 400:
        return (255, 255, 0) # Pure Yellow (404s)
    base_r, base_g, base_b = (0, 255, 0)
    factor = min(max(latency - 0.5, 0), 1.0) 

    r = int(base_r + (255 - base_r) * factor)
    g = int(base_g + (255 - base_g) * factor)
    b = int(base_b + (255 - base_b) * factor)

    return (r, g, b)



def assign_column(service_name):
    global NEXT_COL
    if service_name not in SERVICE_MAP:
        SERVICE_MAP[service_name] = NEXT_COL
        NEXT_COL = (NEXT_COL + 10) % WIDTH # Simple spacing logic
    return SERVICE_MAP[service_name]

@app.route('/ingest', methods=['POST'])
def ingest_log():
    data = request.json

    # 1. Extract Info
    service = data.get('service', 'unknown')
    status = data.get('status', 200)
    latency = data.get('latency', 0.0)

    # 2. Determine Position (X-Axis)
    col_idx = assign_column(service)

    # 3. Determine Color (The Pixel)
    color = get_color(status, latency)

    # 4. Paint to Buffer
    ROW_BUFFER[col_idx] = color

    print(f"Painted [Service: {service}] at [Col: {col_idx}] with Color: {color}")

    return jsonify({"status": "accepted"})

if __name__ == "__main__":
    print("Starting Log Painter (Ingest Server)...")
    app.run(port=8000)
