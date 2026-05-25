"""
Usage: python src/process.py <video_path> [--filter calibration/red_filter.json] [--device mps]

Colour-first detection for Enniskillen Rangers (red kit).

Per frame (sampled at 10fps):
  1. Convert to HSV.
  2. Threshold for Rangers red (wraps at 0/179) using calibration/red_filter.json.
  3. Morphological open → close to clean noise.
  4. Find contours; filter by height, area, aspect ratio (h/w).
  5. Feed surviving bboxes to ByteTrack.

Every detection is by definition a Rangers player — team is hardcoded to 0.
No YOLO inference, no classify_teams.py needed.

Outputs:
  output/tracks_generic_{clip_stem}.parquet  (same schema as YOLO pipeline)
  output/meta_generic_{clip_stem}.json
  output/annotated_generic_{clip_stem}.mp4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import supervision as sv

OUTPUT_DIR     = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
FILTER_PATH    = Path("calibration/red_filter.json")

TARGET_FPS = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video_path", type=Path)
    p.add_argument("--filter", type=Path, default=FILTER_PATH,
                   help="HSV red-filter config (default: calibration/red_filter.json)")
    return p.parse_args()


def load_filter(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Filter config not found: {path}")
    return json.loads(path.read_text())


def build_video_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def detect_red_blobs(
    frame: np.ndarray, cfg: dict
) -> list[tuple[int, int, int, int]]:
    """
    Return list of (x1, y1, x2, y2) bboxes passing the red-blob filter.
    Bboxes are the tightest rectangle around each surviving contour.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps in HSV — two hue ranges
    lo1 = np.array([cfg["hue_low_1"],  cfg["sat_min"], cfg["val_min"]], dtype=np.uint8)
    hi1 = np.array([cfg["hue_high_1"], 255,            255           ], dtype=np.uint8)
    lo2 = np.array([cfg["hue_low_2"],  cfg["sat_min"], cfg["val_min"]], dtype=np.uint8)
    hi2 = np.array([cfg["hue_high_2"], 255,            255           ], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)

    # Morphological open (remove tiny specks) then close (fill jersey gaps)
    open_k  = cfg["morph_open_k"]
    close_k = cfg["morph_close_k"]
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,  open_k))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = cfg["min_area"]
    min_h    = cfg["min_height_px"]
    max_h    = cfg["max_height_px"]
    ar_min   = cfg["aspect_ratio_min"]
    ar_max   = cfg["aspect_ratio_max"]

    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h < min_h or h > max_h:
            continue
        ar = h / w if w > 0 else 0
        if ar < ar_min or ar > ar_max:
            continue
        bboxes.append((x, y, x + w, y + h))

    return bboxes


def main():
    args    = parse_args()
    cfg     = load_filter(args.filter)

    if not args.video_path.exists():
        sys.exit(f"Video not found: {args.video_path}")

    clip_stem    = args.video_path.stem
    parquet_path = OUTPUT_DIR / f"tracks_generic_{clip_stem}.parquet"
    meta_path    = OUTPUT_DIR / f"meta_generic_{clip_stem}.json"
    video_out    = OUTPUT_DIR / f"annotated_generic_{clip_stem}.mp4"

    cap          = cv2.VideoCapture(str(args.video_path))
    source_fps   = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_step = max(1, round(source_fps / TARGET_FPS))
    print(f"Source: {args.video_path.name}  {width}x{height}  {source_fps:.2f}fps  {total_frames} frames")
    print(f"Colour-first detection — red filter: {args.filter}")
    print(f"  hue [0-{cfg['hue_high_1']}] OR [{cfg['hue_low_2']}-179], "
          f"sat>={cfg['sat_min']}, val>={cfg['val_min']}")
    print(f"  blob filter: h={cfg['min_height_px']}-{cfg['max_height_px']}px, "
          f"area>={cfg['min_area']}, ar={cfg['aspect_ratio_min']}-{cfg['aspect_ratio_max']}")
    print(f"Sampling every {frame_step} frames → ~{source_fps/frame_step:.1f}fps processed")

    t_start = time.time()

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=TARGET_FPS,
    )

    writer  = build_video_writer(video_out, width, height, TARGET_FPS)
    records = []

    cap              = cv2.VideoCapture(str(args.video_path))
    source_frame_idx = 0
    output_frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if source_frame_idx % frame_step != 0:
            source_frame_idx += 1
            continue

        timestamp_s = source_frame_idx / source_fps
        bboxes      = detect_red_blobs(frame, cfg)

        if bboxes:
            xyxy       = np.array(bboxes, dtype=np.float32)
            confidence = np.ones(len(bboxes), dtype=np.float32)
            class_id   = np.zeros(len(bboxes), dtype=int)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
            )
        else:
            detections = sv.Detections.empty()

        detections = tracker.update_with_detections(detections)

        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            records.append({
                "frame_id":    output_frame_idx,
                "timestamp_s": round(timestamp_s, 4),
                "track_id":    int(detections.tracker_id[i]),
                "x1":          float(x1),
                "y1":          float(y1),
                "x2":          float(x2),
                "y2":          float(y2),
                "confidence":  float(detections.confidence[i]),
                "pitch_x":     float("nan"),
                "pitch_y":     float("nan"),
                "team":        0,
            })

        # Annotate: draw red bboxes + track IDs
        annotated = frame.copy()
        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            tid = int(detections.tracker_id[i])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 220), 2)
            cv2.putText(annotated, f"#{tid}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1, cv2.LINE_AA)
        writer.write(annotated)

        if output_frame_idx % 50 == 0:
            print(f"  frame {output_frame_idx} ({timestamp_s:.1f}s)  detections={len(detections)}")

        source_frame_idx += 1
        output_frame_idx += 1

    cap.release()
    writer.release()

    df = pd.DataFrame(records).astype({
        "frame_id":   "int32",
        "track_id":   "int32",
        "x1":         "float32",
        "y1":         "float32",
        "x2":         "float32",
        "y2":         "float32",
        "confidence": "float32",
        "pitch_x":    "float32",
        "pitch_y":    "float32",
        "team":       "int8",
    })
    schema = pa.schema([
        ("frame_id",    pa.int32()),
        ("timestamp_s", pa.float64()),
        ("track_id",    pa.int32()),
        ("x1",          pa.float32()),
        ("y1",          pa.float32()),
        ("x2",          pa.float32()),
        ("y2",          pa.float32()),
        ("confidence",  pa.float32()),
        ("pitch_x",     pa.float32()),
        ("pitch_y",     pa.float32()),
        ("team",        pa.int8()),
    ])
    pq.write_table(
        pa.Table.from_pandas(df, schema=schema, preserve_index=False),
        parquet_path,
    )

    elapsed = time.time() - t_start
    avg     = len(df) / output_frame_idx if output_frame_idx else 0

    meta = {
        "clip":              str(args.video_path),
        "model":             "colour-first (HSV red threshold)",
        "conf_threshold":    None,
        "processing_time_s": round(elapsed, 1),
        "frames_processed":  output_frame_idx,
        "detections":        len(df),
        "avg_per_frame":     round(avg, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nDone. {output_frame_idx} frames, {len(df)} detections ({avg:.1f}/frame) in {elapsed:.0f}s.")
    print(f"Parquet → {parquet_path}")
    print(f"Meta    → {meta_path}")
    print(f"Video   → {video_out}")
    print(f"\nNext step:")
    print(f"  python src/apply_homography.py --tracks {parquet_path}")


if __name__ == "__main__":
    main()
