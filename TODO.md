# Log Sonar — TODO / Scratchpad

> Use this file to track concerns, ideas, and separate them for focused work.

---

## ✅ Done

- [x] Pixel buffer: accumulates RGB into 50×50 grid (`pixel_buffer.py`)
- [x] Image generator: grid → PNG with upscaling (`image_generator.py`)
- [x] Image analyzer: K-means dominance, autocorrelation periodicity, anomaly detection (`image_analyzer.py`)
- [x] Pipeline orchestrator: wires buffer → generator → analyzer (`pipeline.py`)
- [x] Painter integration: feeds pixels into the pipeline (`painter.py`)
- [x] Test script with synthetic data (`test_pipeline.py`)

---

## 🔮 Future Ideas

- [ ] Real-time WebSocket dashboard showing the grid filling up live
- [ ] Service-specific image grids (one image per service name)
- [ ] Trend analysis across multiple sequential images
- [ ] Configurable grid sizes via CLI args or config file
- [ ] Export analysis reports as HTML with embedded images
- [ ] Add edge detection (Sobel/Canny) for sharper pattern boundaries
- [ ] Database storage of analysis results for historical comparison
- [ ] Alert system: if anomaly percentage > threshold, notify via webhook

---

## 📝 Notes

- Currently the pixel buffer fills sequentially (left→right, top→bottom).
  Each pixel corresponds to one log event, so the spatial layout represents
  chronological order of incoming logs.
- The analyzer ignores black (0,0,0) pixels — these are unfilled cells
  if the grid isn't completely full.
- Images are upscaled 10× for human viewing but analyzed at native 50×50.
