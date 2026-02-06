
import logging
import random
import time
import requests
import json
from flask import Flask, jsonify
import threading

# Configure Logging to simply print JSON to stdout (for now, easier to pipe)
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "service": "toy-service-" + str(random.randint(1, 3)), # Simulating 3 services
            "timestamp": time.time(),
            "status": record.status_code if hasattr(record, 'status_code') else 200,
            "latency": record.latency if hasattr(record, 'latency') else 0.0,
            "message": record.getMessage()
        }
        return json.dumps(log_record)

logger = logging.getLogger("toy_logger")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = Flask(__name__)

def generate_random_traffic():
    while True:
        # Simulate Random Behavior
        outcome = random.random()
        latency = random.uniform(0.01, 1.2) # 10ms to 1.2s

        status = 200
        if outcome < 0.1: # 10% chance of error
            status = 500
        elif outcome < 0.3: # 20% chance of high latency (warning)
            latency += 0.5

        # Create a dummy record with extra attributes
        record = logging.LogRecord("toy_logger", logging.INFO, "", 0, "Request processed", None, None)
        record.status_code = status
        record.latency = latency

        logger.handle(record)

        # Send data to our Painter (Simulating OTel Exporter)
        try:
            requests.post("http://localhost:8000/ingest", json=json.loads(handler.format(record)), timeout=0.1)
        except:
            pass # Painter might be offline

        time.sleep(random.uniform(0.1, 0.5)) # Random gap between logs

# Start traffic generator in background
threading.Thread(target=generate_random_traffic, daemon=True).start()

if __name__ == "__main__":
    print("Starting Toy Flask App (Log Generator)...")
    app.run(port=5000)
