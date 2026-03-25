"""
pixel_buffer.py — Thread-safe 50×50 pixel grid accumulator.

Collects RGB tuples from the painter into a 2D grid.
When the grid is full (width × height pixels), triggers a callback
with the completed grid and resets for the next batch.
"""

import threading
import numpy as np


class PixelBuffer:
    def __init__(self, width=50, height=50, on_grid_full=None):
        """
        Args:
            width:  Number of columns in the pixel grid.
            height: Number of rows in the pixel grid.
            on_grid_full: Callback(np.ndarray) called when the grid is full.
                          The array shape is (height, width, 3) with dtype uint8.
        """
        self.width = width
        self.height = height
        self.on_grid_full = on_grid_full
        self._lock = threading.Lock()
        self._reset()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def add_pixel(self, color):
        """
        Append a single pixel to the grid (fills left→right, top→bottom).

        Args:
            color: Tuple (R, G, B) with values 0-255.
        """
        fire_callback = False
        completed_grid = None

        with self._lock:
            row = self._pixel_count // self.width
            col = self._pixel_count % self.width
            self._grid[row, col] = color
            self._pixel_count += 1

            # Log progress every 10%
            total = self.width * self.height
            if self._pixel_count % max(1, total // 10) == 0:
                pct = self._pixel_count / total * 100
                print(f"[PixelBuffer] Grid fill: {self._pixel_count}/{total} ({pct:.0f}%)")

            if self._pixel_count >= total:
                completed_grid = self._grid.copy()
                fire_callback = True
                self._reset()

        # Fire callback *outside* the lock to avoid deadlocks
        if fire_callback and self.on_grid_full:
            self.on_grid_full(completed_grid)

    def add_pixel_at(self, row, col, color):
        """
        Place a pixel at an explicit (row, col) position.

        Args:
            row: Row index (0-based).
            col: Column index (0-based).
            color: Tuple (R, G, B) with values 0-255.
        """
        with self._lock:
            if 0 <= row < self.height and 0 <= col < self.width:
                self._grid[row, col] = color
                self._pixel_count += 1

                if self._pixel_count >= self.width * self.height:
                    completed_grid = self._grid.copy()
                    self._reset()
                    if self.on_grid_full:
                        # Release lock before callback
                        self._lock.release()
                        try:
                            self.on_grid_full(completed_grid)
                        finally:
                            self._lock.acquire()

    def get_grid_snapshot(self):
        """Return a copy of the current (possibly incomplete) grid."""
        with self._lock:
            return self._grid.copy(), self._pixel_count

    def get_fill_percentage(self):
        """Return how full the current grid is (0.0 – 1.0)."""
        with self._lock:
            return self._pixel_count / (self.width * self.height)

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    def _reset(self):
        """Reset the grid to all-black."""
        self._grid = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._pixel_count = 0
