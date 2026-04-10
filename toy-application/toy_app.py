"""
toy_app.py — Smart log generator with service personality profiles.

Simulates 3 microservices, each with distinct behavioral patterns:
  - toy-service-1 (Stable API):    Low latency, rare errors, low resources
  - toy-service-2 (Heavy Worker):  High CPU/mem, periodic error bursts every ~30 reqs
  - toy-service-3 (Flaky Gateway): Erratic latency, frequent 502/503, timeout spikes
"""

import logging
import random
import time
import requests
import json
import math
from flask import Flask
import threading


# ------------------------------------------------------------------ #
#  Service Personality Profiles                                       #
# ------------------------------------------------------------------ #

class ServiceProfile:
    """Defines the behavioral characteristics of a simulated service."""

    def __init__(self, name, endpoints, base_latency, latency_jitter,
                 error_rate, error_codes, cpu_range, memory_range,
                 request_size_range, spike_pattern=None):
        self.name = name
        self.endpoints = endpoints
        self.base_latency = base_latency
        self.latency_jitter = latency_jitter
        self.error_rate = error_rate
        self.error_codes = error_codes
        self.cpu_range = cpu_range
        self.memory_range = memory_range
        self.request_size_range = request_size_range
        self.spike_pattern = spike_pattern  # (interval, burst_size) or None
        self._request_count = 0

    def generate_log(self):
        """Generate a realistic log entry based on this service's profile."""
        self._request_count += 1

        endpoint = random.choice(self.endpoints)
        latency = max(0.001, self.base_latency + random.gauss(0, self.latency_jitter))
        cpu = random.uniform(*self.cpu_range)
        memory = random.uniform(*self.memory_range)
        request_size = random.randint(*self.request_size_range)

        # Determine status code
        status = 200
        is_spike = False

        # Check for periodic spike pattern
        if self.spike_pattern:
            interval, burst_size = self.spike_pattern
            cycle_pos = self._request_count % interval
            if cycle_pos < burst_size:
                is_spike = True
                status = random.choice(self.error_codes)
                latency *= 3  # Spikes = slow
                cpu = min(100, cpu * 1.8)

        # Random errors (independent of spikes)
        if not is_spike and random.random() < self.error_rate:
            status = random.choice(self.error_codes)
            latency *= 1.5

        return {
            "service": self.name,
            "timestamp": time.time(),
            "status": status,
            "latency": round(latency, 4),
            "cpu_percent": round(cpu, 1),
            "memory_mb": round(memory, 1),
            "request_size_bytes": request_size,
            "endpoint": endpoint,
            "message": f"{endpoint} — {'ERROR' if status >= 400 else 'OK'}"
        }


# --- Define the 3 service personalities ---

SERVICES = [
    ServiceProfile(
        name="toy-service-1",
        endpoints=["/api/users", "/api/users/me", "/api/health", "/api/config"],
        base_latency=0.05,       # 50ms — fast
        latency_jitter=0.02,
        error_rate=0.03,         # 3% errors — very reliable
        error_codes=[500],
        cpu_range=(5, 20),       # Low CPU
        memory_range=(80, 150),  # Low memory
        request_size_range=(256, 2048),
        spike_pattern=None,      # No periodic issues
    ),
    ServiceProfile(
        name="toy-service-2",
        endpoints=["/api/process", "/api/batch", "/api/transform", "/api/compute"],
        base_latency=0.3,        # 300ms — moderate
        latency_jitter=0.15,
        error_rate=0.08,         # 8% random errors
        error_codes=[500, 503],
        cpu_range=(50, 90),      # High CPU — heavy worker
        memory_range=(300, 600), # High memory
        request_size_range=(4096, 65536),
        spike_pattern=(30, 4),   # Error burst of 4 every 30 requests
    ),
    ServiceProfile(
        name="toy-service-3",
        endpoints=["/gateway/proxy", "/gateway/forward", "/gateway/upstream"],
        base_latency=0.2,        # 200ms base but very erratic
        latency_jitter=0.4,      # High jitter!
        error_rate=0.15,         # 15% errors — flaky
        error_codes=[502, 503, 504],
        cpu_range=(15, 45),      # Moderate CPU
        memory_range=(120, 250), # Moderate memory
        request_size_range=(512, 8192),
        spike_pattern=(50, 8),   # Longer bursts, less frequent
    ),
]



#  JSON Logging

class JsonFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, 'log_data'):
            return json.dumps(record.log_data)
        return json.dumps({"message": record.getMessage()})


logger = logging.getLogger("toy_logger")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = Flask(__name__)


# ------------------------------------------------------------------ #
#  Traffic Generator                                                  #
# ------------------------------------------------------------------ #

def generate_random_traffic():
    """Continuously generate logs from all 3 services with weighted selection."""
    # Weights: service-2 gets more traffic (it's a heavy worker)
    weights = [0.30, 0.40, 0.30]

    while True:
        # Pick a service based on weights
        service = random.choices(SERVICES, weights=weights, k=1)[0]
        log_data = service.generate_log()

        # Log to stdout
        record = logging.LogRecord(
            "toy_logger", logging.INFO, "", 0,
            log_data["message"], None, None
        )
        record.log_data = log_data
        logger.handle(record)

        # Send to Painter
        try:
            requests.post(
                "http://localhost:8000/ingest",
                json=log_data,
                timeout=0.1
            )
        except Exception:
            pass

        time.sleep(random.uniform(0.01, 0.05))


# Start traffic generator in background
threading.Thread(target=generate_random_traffic, daemon=True).start()

if __name__ == "__main__":
    print("Starting Toy Flask App (Smart Log Generator)...")
    print(f"  Services: {[s.name for s in SERVICES]}")
    print(f"  Spike patterns: service-2 every 30 reqs, service-3 every 50 reqs")
    app.run(port=5000)
