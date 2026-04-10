"""
pixel_buffer.py — Service-aware pixel grid accumulator.

Creates a composite image with separate horizontal bands per service,
separated by gray divider rows. Each service fills its band
independently (left→right, top→bottom within its section).

Image layout for 3 services, 16 rows each, 50 cols wide:
  ┌──────────────────────────┐
  │  service-1  (rows 0-15)  │
  ├──────────────────────────┤  ← gray divider
  │  service-2  (rows 17-32) │
  ├──────────────────────────┤  ← gray divider
  │  service-3  (rows 34-49) │
  └──────────────────────────┘
"""

import threading
import numpy as np


DIVIDER_COLOR = (80, 80, 80)


class ServiceAwareBuffer:
    def __init__(self, width=50, rows_per_service=16, services=None, on_grid_full=None):
        """
        Args:
            width:            Number of columns in the pixel grid.
            rows_per_service: Number of pixel rows allocated per service.
            services:         List of service names (order = top-to-bottom).
            on_grid_full:     Callback(grid, service_layout) when all bands are full.
        """
        self.width = width
        self.rows_per_service = rows_per_service
        self.services = services or []
        self.on_grid_full = on_grid_full
        self._lock = threading.Lock()

        # Calculate total height: rows_per_service * n_services + (n_services - 1) dividers
        n = len(self.services)
        self.divider_rows = max(0, n - 1)
        self.total_height = rows_per_service * n + self.divider_rows

        # Map each service → its starting row in the composite grid
        self._service_layout = {}
        for i, svc in enumerate(self.services):
            start_row = i * (rows_per_service + 1)  # +1 for divider
            self._service_layout[svc] = {
                "start_row": start_row,
                "end_row": start_row + rows_per_service - 1,
                "rows": rows_per_service,
            }

        self._reset()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def add_pixel(self, service, color):
        """
        Append a pixel to the specified service's band.

        Args:
            service: Service name string.
            color:   Tuple (R, G, B) with values 0-255.
        """
        fire_callback = False
        completed_grid = None
        layout_snapshot = None

        with self._lock:
            if service not in self._service_counters:
                # Unknown service — dynamically register if possible
                if service not in self._service_layout:
                    return  # Ignore unknown services
                self._service_counters[service] = 0

            band_capacity = self.rows_per_service * self.width
            count = self._service_counters[service]

            if count >= band_capacity:
                return  # Band already full, wait for flush

            # Calculate position within this service's band
            local_row = count // self.width
            local_col = count % self.width
            global_row = self._service_layout[service]["start_row"] + local_row

            self._grid[global_row, local_col] = color
            self._service_counters[service] = count + 1

            # Log progress every 25%
            new_count = count + 1
            quarter = max(1, band_capacity // 4)
            if new_count % quarter == 0:
                pct = new_count / band_capacity * 100
                print(f"[Buffer] {service}: {new_count}/{band_capacity} ({pct:.0f}%)")

            # Check if ALL services are full
            if all(
                self._service_counters.get(s, 0) >= band_capacity
                for s in self.services
            ):
                completed_grid = self._grid.copy()
                layout_snapshot = dict(self._service_layout)
                fire_callback = True
                self._reset()

        if fire_callback and self.on_grid_full:
            self.on_grid_full(completed_grid, layout_snapshot)

    def get_fill_status(self):
        """Return per-service fill percentages."""
        with self._lock:
            band_capacity = self.rows_per_service * self.width
            result = {}
            for svc in self.services:
                count = self._service_counters.get(svc, 0)
                result[svc] = {
                    "pixels": count,
                    "capacity": band_capacity,
                    "percentage": round(count / band_capacity * 100, 1),
                }
            return result

    def get_grid_snapshot(self):
        """Return a copy of the current (possibly incomplete) grid."""
        with self._lock:
            return self._grid.copy(), dict(self._service_layout)

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    def _reset(self):
        """Reset the grid — all black with gray divider rows."""
        self._grid = np.zeros((self.total_height, self.width, 3), dtype=np.uint8)
        self._service_counters = {svc: 0 for svc in self.services}

        # Paint divider rows
        for i in range(len(self.services) - 1):
            divider_row = (i + 1) * self.rows_per_service + i
            self._grid[divider_row, :] = DIVIDER_COLOR
