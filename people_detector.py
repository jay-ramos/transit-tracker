"""
people_detector.py
-----------------
Real-time people detector using YOLO26 + BotSORT.

Features:
  - Person detection (YOLO26)
  - Multi-object tracking (persists IDs across frames)
  - Virtual crossing line  →  IN / OUT counters
  - Movement trails per tracked ID
  - Bounding boxes color-coded by crossing state
  - Semi-transparent HUD dashboard

Usage:
  python people_detector.py --input input.mp4 --output output.mp4

Optional flags:
  --model     yolo26n.pt | yolo26s.pt | yolo26m.pt  (default: yolo26m.pt)
  --line      0.0–1.0  relative position of the crossing line (default: 0.5)
  --direction horizontal | vertical                  (default: horizontal)
  --conf      detection confidence threshold          (default: 0.40)
  --trail     number of frames to keep trail         (default: 30)
  --no-preview  skip live window (faster on headless machines)
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────
#  Colour palette  (BGR)
# ─────────────────────────────────────────────
C_BOX_DEFAULT = (0,   220,  80)   # green  — not yet crossed
C_IN          = (50,  210,  50)   # green  — counted IN
C_OUT         = (60,   60, 220)   # blue   — counted OUT
C_TRAIL       = (0,   165, 255)   # orange trail
C_LINE        = (0,    0,  200)   # red line (idle)
C_LINE_HIT    = (0,  255, 255)    # yellow line (flash on cross)
C_HUD_BG      = (15,  15,  15)
C_TITLE       = (180, 180, 255)
C_NET         = (50,  220, 220)
C_FPS         = (160, 160, 160)


# ─────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────

def draw_hud(frame: np.ndarray, count_in: int, count_out: int, fps: float) -> None:
    """Semi-transparent top-left dashboard."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (270, 145), C_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "PEOPLE  DETECTOR",
                (18, 36), cv2.FONT_HERSHEY_DUPLEX, 0.68, C_TITLE, 1, cv2.LINE_AA)

    cv2.putText(frame, f"IN  : {count_in:>4}",
                (18, 72), cv2.FONT_HERSHEY_DUPLEX, 0.88, C_IN,  2, cv2.LINE_AA)
    cv2.putText(frame, f"OUT : {count_out:>4}",
                (18, 108), cv2.FONT_HERSHEY_DUPLEX, 0.88, C_OUT, 2, cv2.LINE_AA)

    net = count_in - count_out
    cv2.putText(frame, f"Net : {net:>+4}",
                (18, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_NET, 1, cv2.LINE_AA)

    # FPS – top-right corner
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS {fps:5.1f}",
                (w - 130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_FPS, 1, cv2.LINE_AA)


def draw_trail(frame: np.ndarray, points: list[tuple[int, int]]) -> None:
    """Gradient-opacity trail from oldest (dim) to newest (bright)."""
    n = len(points)
    for i in range(1, n):
        alpha   = i / n
        color   = tuple(int(c * alpha) for c in C_TRAIL)
        thick   = max(1, int(3 * alpha))
        cv2.line(frame, points[i - 1], points[i], color, thick, cv2.LINE_AA)


def draw_box(frame: np.ndarray,
             x1: int, y1: int, x2: int, y2: int,
             tid: int, state: str | None) -> None:
    """Bounding box + small label badge."""
    color = C_IN if state == "in" else (C_OUT if state == "out" else C_BOX_DEFAULT)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    label = f"#{tid}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label,
                (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (255, 255, 255), 1, cv2.LINE_AA)


def draw_line(frame: np.ndarray,
              start: tuple[int, int], end: tuple[int, int],
              direction: str, flash: bool) -> None:
    """Crossing line with IN / OUT labels."""
    color = C_LINE_HIT if flash else C_LINE
    cv2.line(frame, start, end, color, 2, cv2.LINE_AA)

    h, w = frame.shape[:2]
    if direction == "horizontal":
        mx, my = w // 2, start[1]
        cv2.putText(frame, "▲ IN",  (mx - 50, my - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_IN,  1, cv2.LINE_AA)
        cv2.putText(frame, "▼ OUT", (mx + 10, my + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_OUT, 1, cv2.LINE_AA)
    else:
        mx, my = start[0], h // 2
        cv2.putText(frame, "◀ IN",  (mx - 60, my - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_IN,  1, cv2.LINE_AA)
        cv2.putText(frame, "▶ OUT", (mx + 8,  my + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_OUT, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  Crossing logic
# ─────────────────────────────────────────────

def check_crossing(prev_pt: tuple[int, int],
                   curr_pt: tuple[int, int],
                   line_coord: int,
                   direction: str) -> str | None:
    """Returns 'in', 'out', or None."""
    axis = 1 if direction == "horizontal" else 0   # y-axis or x-axis
    prev = prev_pt[axis]
    curr = curr_pt[axis]
    if prev < line_coord <= curr:
        return "in"
    if prev > line_coord >= curr:
        return "out"
    return None


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    print(f"\n🚀  Loading model: {args.model}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.input}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Crossing line coordinates ──
    if args.direction == "horizontal":
        line_coord = int(H * args.line)
        line_start = (0, line_coord)
        line_end   = (W, line_coord)
    else:
        line_coord = int(W * args.line)
        line_start = (line_coord, 0)
        line_end   = (line_coord, H)

    # ── Output writer ──
    out_path = Path(args.output)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_src,
        (W, H),
    )

    # ── State ──
    track_history: dict[int, list[tuple[int, int]]] = defaultdict(list)
    crossed_ids:   dict[int, str]                   = {}
    count_in  = 0
    count_out = 0

    frame_idx  = 0
    flash_frames: set[int] = set()   # frames where a crossing happened
    t_prev = time.perf_counter()

    print(f"📹  Processing {total_frames} frames  ({W}×{H} @ {fps_src:.1f} fps)")
    print(f"    Line position : {args.direction}  at {args.line*100:.0f}%")
    print(f"    Press  Q  to quit early.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ── FPS meter ──
        t_now   = time.perf_counter()
        fps_now = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev  = t_now

        # ── YOLO track ──
        results = model.track(
            frame,
            persist   = True,
            classes   = [0],          # person only
            conf      = args.conf,
            iou       = args.iou,
            imgsz     = args.imgsz,
            verbose   = False,
            tracker   = args.tracker,
            device    = 0, 
        )

        crossed_this_frame = False

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes    = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Trail
                track_history[tid].append((cx, cy))
                if len(track_history[tid]) > args.trail:
                    track_history[tid].pop(0)

                draw_trail(frame, track_history[tid])

                # Crossing check (only once per ID)
                trail = track_history[tid]
                if len(trail) >= 2 and tid not in crossed_ids:
                    result = check_crossing(
                        trail[-2], trail[-1], line_coord, args.direction
                    )
                    if result:
                        crossed_ids[tid] = result
                        if result == "in":
                            count_in  += 1
                        else:
                            count_out += 1
                        crossed_this_frame = True

                draw_box(frame, x1, y1, x2, y2, tid, crossed_ids.get(tid))

        # Flash line for a few frames after a crossing
        flash = crossed_this_frame or (frame_idx in flash_frames)
        if crossed_this_frame:
            # keep flash for next 8 frames
            flash_frames.update(range(frame_idx + 1, frame_idx + 9))

        draw_line(frame, line_start, line_end, args.direction, flash)
        draw_hud(frame, count_in, count_out, fps_now)

        writer.write(frame)

        if not args.no_preview:
            cv2.imshow("People Detector  [Q to quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n⚠️  Interrupted by user.")
                break

        if frame_idx % 100 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"    Frame {frame_idx:>5}/{total_frames}  ({pct:5.1f}%)  "
                  f"IN={count_in}  OUT={count_out}", end="\r")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n\n✅  Finished!")
    print(f"   Total IN  : {count_in}")
    print(f"   Total OUT : {count_out}")
    print(f"   Net       : {count_in - count_out:+d}")
    print(f"   Output    : {out_path.resolve()}\n")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="People Counter — YOLO26 + BoTSORT + crossing line"
    )
    parser.add_argument("--input",      required=True,              help="Path to input video")
    parser.add_argument("--output",     default="output.mp4",       help="Path to output video")
    parser.add_argument("--model",      default="yolo26x.pt",       help="YOLO26 weights file")
    parser.add_argument("--tracker", default="botsort.yaml", 
                        choices=["botsort.yaml", "bytetrack.yaml"], help="Tracker config")
    parser.add_argument("--conf",       type=float, default=0.40,   help="Detection confidence threshold")
    parser.add_argument("--iou",        type=float, default=0.40,   help="NMS IoU threshold")
    parser.add_argument("--imgsz",      type=int, default=1280,      help="Input image size for inference")
    parser.add_argument("--line",       type=float, default=0.5,    help="Crossing line position (0.0–1.0)")
    parser.add_argument("--direction",  default="horizontal",
                        choices=["horizontal", "vertical"],         help="Line orientation")
    parser.add_argument("--trail",      type=int,   default=60,     help="Trail length in frames")
    parser.add_argument("--no-preview", action="store_true",        help="Disable live preview window")
    args = parser.parse_args()

    main(args)
