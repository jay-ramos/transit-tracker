# 🚶 Transit Tracker — Real-Time People Counter

Real-time pedestrian detection, multi-object tracking, and bidirectional people counting from video streams.

Built with **YOLO + BotSORT**, a configurable virtual crossing line, and an overlay HUD showing live IN / OUT / Net counts. Designed for production use in transit hubs, retail spaces, and smart-city infrastructure.

---

## ✨ Features

- **Person detection** — YOLO (nano → extra-large weights)
- **Multi-object tracking** — BotSORT or ByteTrack; IDs persist across frames
- **Bidirectional counting** — virtual crossing line with configurable position and orientation
- **Movement trails** — per-ID trajectory visualization with gradient opacity
- **Color-coded bounding boxes** — green (IN) / blue (OUT) / default (not yet crossed)
- **Semi-transparent HUD** — live IN, OUT, Net counters + FPS meter
- **Line flash** — visual feedback on each crossing event
- **Headless mode** — `--no-preview` flag for server/edge deployments

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF?style=flat)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

---

## 📦 Installation

```bash
git clone https://github.com/jay-ramos/transit-tracker.git
cd transit-tracker
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python people_detector.py --input input.mp4 --output output.mp4
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to input video |
| `--output` | `output.mp4` | Path to output video |
| `--model` | `yolo26x.pt` | YOLO weights (`yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26x.pt`) |
| `--tracker` | `botsort.yaml` | Tracker config (`botsort.yaml` or `bytetrack.yaml`) |
| `--conf` | `0.40` | Detection confidence threshold |
| `--iou` | `0.40` | NMS IoU threshold |
| `--imgsz` | `1280` | Input image size for inference |
| `--line` | `0.5` | Crossing line position — `0.0` to `1.0` (relative) |
| `--direction` | `horizontal` | Line orientation (`horizontal` or `vertical`) |
| `--trail` | `60` | Trail length in frames |
| `--no-preview` | off | Disable live preview window (faster on headless machines) |

### Examples

```bash
# Lightweight model, line at 40% from top, 30-frame trail
python people_detector.py --input station.mp4 --model yolo26n.pt --line 0.4 --trail 30

# Vertical line (left = IN, right = OUT), headless, high confidence
python people_detector.py --input corridor.mp4 --direction vertical --conf 0.55 --no-preview
```

---

## 📊 Output

The processed video includes:
- Bounding boxes with track IDs
- Color-coded crossing state (green = IN, blue = OUT)
- Per-ID movement trails
- HUD with live counts and FPS

Terminal summary on completion:
```
✅ Finished!
   Total IN  : 142
   Total OUT : 138
   Net       : +4
   Output    : output.mp4
```

---

## 🏗️ Real-World Context

This project reflects patterns from production computer vision systems I have built and deployed in real-world environments — including high-traffic venues with thousands of daily visitors, operating 24/7 across multiple simultaneous camera streams.

---

## 📄 License

GPL-3.0 — see [LICENSE](LICENSE) for details.
