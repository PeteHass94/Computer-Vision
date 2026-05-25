"""
Usage: python src/process_football.py <video_path> [--model models/football_yolo.pt] [--device mps]

Football-specific detection using a model trained on player/goalkeeper/referee/ball.
Identical pipeline to process.py but:
  - Detects all 4 classes (no class filter)
  - Stores class_id and class_name per detection
  - Ball detections bypass the bbox height filter
  - Output: output/tracks_football_{clip_stem}.parquet

Model source: roboflow-jvuqo/football-players-detection-3zvbc via Roboflow Sports
  (https://github.com/roboflow/sports)
Classes: {0: ball, 1: goalkeeper, 2: player, 3: referee}
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
from ultralytics import YOLO

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH     = Path("models/football_yolo.pt")
TARGET_FPS     = 10
IMGSZ          = 1280
CONF_THRESHOLD = 0.3   # lower than generic — football model is more precise

# Class IDs from this model
CLASS_BALL       = 0
CLASS_GOALKEEPER = 1
CLASS_PLAYER     = 2
CLASS_REFEREE    = 3
CLASS_NAMES      = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

# Bbox height filter — applied to people only (not ball)
PERSON_BBOX_H_MIN = 20
PERSON_BBOX_H_MAX = 150


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video_path", type=Path)
    p.add_argument("--model",  type=Path, default=MODEL_PATH)
    p.add_argument("--device", default="mps")
    p.add_argument("--conf",   type=float, default=CONF_THRESHOLD)
    return p.parse_args()


def build_video_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def main():
    args = parse_args()
    if not args.video_path.exists():
        sys.exit(f"Video not found: {args.video_path}")
    if not args.model.exists():
        sys.exit(f"Model not found: {args.model}\nDownload with: gdown '...' -O {args.model}")

    clip_stem    = args.video_path.stem
    parquet_path = OUTPUT_DIR / f"tracks_football_{clip_stem}.parquet"
    meta_path    = OUTPUT_DIR / f"meta_football_{clip_stem}.json"
    video_out    = OUTPUT_DIR / f"annotated_football_{clip_stem}.mp4"

    cap          = cv2.VideoCapture(str(args.video_path))
    source_fps   = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_step = max(1, round(source_fps / TARGET_FPS))
    print(f"Source: {args.video_path.name}  {width}x{height}  {source_fps:.2f}fps  {total_frames} frames")
    print(f"Model: {args.model}  device: {args.device}  conf≥{args.conf}")
    print(f"Sampling every {frame_step} frames → ~{source_fps/frame_step:.1f}fps processed")
    print(f"Classes: {CLASS_NAMES}")

    t_start = time.time()

    model = YOLO(str(args.model))
    model.to(args.device)

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=TARGET_FPS,
    )

    # Annotation colours per class
    COLOURS = {
        CLASS_BALL:       sv.Color.from_hex("#FFFFFF"),
        CLASS_GOALKEEPER: sv.Color.from_hex("#FFD700"),
        CLASS_PLAYER:     sv.Color.from_hex("#00BFFF"),
        CLASS_REFEREE:    sv.Color.from_hex("#FF8C00"),
    }
    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    writer = build_video_writer(video_out, width, height, TARGET_FPS)

    records = []
    cap = cv2.VideoCapture(str(args.video_path))
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
        results     = model(frame, imgsz=IMGSZ, conf=args.conf, verbose=False)[0]
        detections  = sv.Detections.from_ultralytics(results)
        detections  = tracker.update_with_detections(detections)

        # Bbox height filter — people only; ball is typically tiny
        heights     = detections.xyxy[:, 3] - detections.xyxy[:, 1]
        class_ids   = detections.class_id
        is_ball     = class_ids == CLASS_BALL
        size_ok     = (heights >= PERSON_BBOX_H_MIN) & (heights <= PERSON_BBOX_H_MAX)
        keep        = is_ball | size_ok
        detections  = detections[keep]

        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            cid  = int(detections.class_id[i])
            records.append({
                "frame_id":    output_frame_idx,
                "timestamp_s": round(timestamp_s, 4),
                "track_id":    int(detections.tracker_id[i]),
                "class_id":    cid,
                "class_name":  CLASS_NAMES[cid],
                "x1":          float(x1),
                "y1":          float(y1),
                "x2":          float(x2),
                "y2":          float(y2),
                "confidence":  float(detections.confidence[i]),
                "pitch_x":     float("nan"),
                "pitch_y":     float("nan"),
            })

        if len(detections) > 0:
            labels    = [f"{CLASS_NAMES[int(cid)][0].upper()}#{tid}"
                         for cid, tid in zip(detections.class_id, detections.tracker_id)]
            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels)
        else:
            annotated = frame.copy()
        writer.write(annotated)

        if output_frame_idx % 50 == 0:
            class_counts = {}
            for cid in detections.class_id:
                class_counts[CLASS_NAMES[int(cid)]] = class_counts.get(CLASS_NAMES[int(cid)], 0) + 1
            print(f"  frame {output_frame_idx} ({timestamp_s:.1f}s)  {class_counts}")

        source_frame_idx += 1
        output_frame_idx += 1

    cap.release()
    writer.release()

    df = pd.DataFrame(records).astype({
        "frame_id":   "int32",
        "track_id":   "int32",
        "class_id":   "int8",
        "x1":         "float32",
        "y1":         "float32",
        "x2":         "float32",
        "y2":         "float32",
        "confidence": "float32",
        "pitch_x":    "float32",
        "pitch_y":    "float32",
    })
    schema = pa.schema([
        ("frame_id",    pa.int32()),
        ("timestamp_s", pa.float64()),
        ("track_id",    pa.int32()),
        ("class_id",    pa.int8()),
        ("class_name",  pa.string()),
        ("x1",          pa.float32()),
        ("y1",          pa.float32()),
        ("x2",          pa.float32()),
        ("y2",          pa.float32()),
        ("confidence",  pa.float32()),
        ("pitch_x",     pa.float32()),
        ("pitch_y",     pa.float32()),
    ])
    pq.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False), parquet_path)

    elapsed   = time.time() - t_start
    avg       = len(df) / output_frame_idx if output_frame_idx else 0
    by_class  = df["class_name"].value_counts().to_dict()

    meta = {
        "clip":              str(args.video_path),
        "model":             str(args.model),
        "conf_threshold":    args.conf,
        "processing_time_s": round(elapsed, 1),
        "frames_processed":  output_frame_idx,
        "detections":        len(df),
        "avg_per_frame":     round(avg, 2),
        "by_class":          by_class,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nDone. {output_frame_idx} frames, {len(df)} detections ({avg:.1f}/frame) in {elapsed:.0f}s.")
    print(f"Class breakdown: {by_class}")
    print(f"Parquet → {parquet_path}")
    print(f"Meta    → {meta_path}")
    print(f"Video   → {video_out}")
    print(f"\nNext steps:")
    print(f"  python src/apply_homography.py --tracks {parquet_path}")
    print(f"  python src/classify_teams.py --video {args.video_path} --tracks {parquet_path}")


if __name__ == "__main__":
    main()
