"""
Usage:
  python src/process_sam2.py \
      --clip clips/dev_clip3.mp4 \
      --frame 300 --x 608 --y 969 \
      --output output/tracks_sam2_dev_clip3.parquet

SAM 2 single-player tracker.

Workflow:
  1. Load the clip directly into SAM 2's video predictor (no frame extraction needed).
  2. Add one positive click on the specified SOURCE frame index.
  3. Propagate forward (then backward) to get a mask every sampled frame.
  4. For each mask, foot position = bottom-centre of the mask bounding box.
  5. Apply the per-clip homography to convert pixel → pitch coordinates.
  6. Write parquet: frame_idx, source_frame, timestamp_s, pixel_x, pixel_y, pitch_x, pitch_y.

--frame is the SOURCE video frame index (what you'd scrub to in a video player).
The script maps it to the nearest 10fps-sampled frame index used internally by SAM 2.

Model: models/sam2_hiera_tiny.pt   Config: sam2_hiera_t
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

TARGET_FPS  = 10
OBJ_ID      = 0
CHECKPOINT  = Path("models/sam2_hiera_tiny.pt")
CONFIG_NAME = "sam2_hiera_t"
CALIB_DIR   = Path("calibration")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip",   type=Path, required=True)
    p.add_argument("--frame",  type=int,  required=True,
                   help="Source video frame index to click on")
    p.add_argument("--x",      type=int,  required=True)
    p.add_argument("--y",      type=int,  required=True)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def foot_position(mask: np.ndarray) -> tuple[float, float] | None:
    """Bottom-centre of the mask bounding box. Returns None if mask is empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float((xs.min() + xs.max()) / 2), float(ys.max())


def apply_homography(pixel_x: float, pixel_y: float, H: np.ndarray,
                     pitch_length: float, pitch_width: float,
                     margin: float = 5.0) -> tuple[float, float]:
    pt  = np.array([[[pixel_x, pixel_y]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)[0][0]
    px, py = float(out[0]), float(out[1])
    if not (-margin <= px <= pitch_length + margin and
            -margin <= py <= pitch_width  + margin):
        return float("nan"), float("nan")
    return px, py


def main():
    args = parse_args()

    if not args.clip.exists():
        sys.exit(f"Clip not found: {args.clip}")
    if not CHECKPOINT.exists():
        sys.exit(f"Model not found: {CHECKPOINT}")

    clip_stem   = args.clip.stem
    output_path = args.output or Path(f"output/tracks_sam2_{clip_stem}.parquet")
    output_path.parent.mkdir(exist_ok=True)

    device = best_device()
    print(f"Device: {device}")

    # --- Video metadata ---
    cap        = cv2.VideoCapture(str(args.clip))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    n_src      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_step   = max(1, round(source_fps / TARGET_FPS))
    n_sam_frames = (n_src + frame_step - 1) // frame_step

    # Map source frame index → nearest SAM 2 frame index
    click_sam_idx = round(args.frame / frame_step)
    click_sam_idx = max(0, min(click_sam_idx, n_sam_frames - 1))
    click_src     = click_sam_idx * frame_step
    print(f"Source frame {args.frame} → SAM 2 frame {click_sam_idx} "
          f"(source {click_src}, ~{click_src/source_fps:.1f}s)")
    print(f"Click: ({args.x}, {args.y})  —  total SAM 2 frames: {n_sam_frames}")

    # --- Load homography ---
    hom_file = CALIB_DIR / f"homography_{clip_stem}.json"
    H = pitch_length = pitch_width = None
    if hom_file.exists():
        cal          = json.loads(hom_file.read_text())
        H            = np.array(cal["homography_matrix"], dtype=np.float64)
        pitch_length = cal["pitch_length_m"]
        pitch_width  = cal["pitch_width_m"]
        print(f"Homography loaded from {hom_file}")
    else:
        print(f"WARNING: {hom_file} not found — pitch_x/y will be NaN")

    # --- Extract sampled frames to JPEG dir (decord unavailable on arm64) ---
    frames_dir = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
    print(f"Extracting {n_sam_frames} frames to {frames_dir} …")
    cap = cv2.VideoCapture(str(args.clip))
    src_idx = 0
    sam_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if src_idx % frame_step == 0:
            out = frames_dir / f"{sam_idx:06d}.jpg"
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            sam_idx += 1
        src_idx += 1
    cap.release()
    print(f"  wrote {sam_idx} JPEG frames")

    # --- Build SAM 2 video predictor ---
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(CONFIG_NAME, str(CHECKPOINT), device=device)
    print("SAM 2 predictor ready")

    masks_by_sam_idx: dict[int, np.ndarray] = {}

    with torch.inference_mode():
        state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,
        )
        print(f"State initialised — {state['num_frames']} frames loaded")

        # Add positive click
        points = np.array([[args.x, args.y]], dtype=np.float32)
        labels = np.array([1],               dtype=np.int32)
        predictor.add_new_points_or_box(
            state,
            frame_idx        = click_sam_idx,
            obj_id           = OBJ_ID,
            points           = points,
            labels           = labels,
            normalize_coords = True,  # divides by original video W/H internally
        )

        # Forward propagation
        print("Propagating forward…")
        for sam_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
            if mask.any():
                masks_by_sam_idx[sam_idx] = mask
            if sam_idx % 100 == 0:
                print(f"  forward frame {sam_idx}/{state['num_frames']}  "
                      f"mask_px={mask.sum()}")

        # Backward propagation (captures frames before the click)
        if click_sam_idx > 0:
            print("Propagating backward…")
            for sam_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                state, reverse=True
            ):
                mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                if mask.any() and sam_idx not in masks_by_sam_idx:
                    masks_by_sam_idx[sam_idx] = mask
                if sam_idx % 100 == 0:
                    print(f"  backward frame {sam_idx}  mask_px={mask.sum()}")

    print(f"Masks collected for {len(masks_by_sam_idx)} frames")
    shutil.rmtree(frames_dir, ignore_errors=True)

    # --- Build records ---
    records = []
    for sam_idx in sorted(masks_by_sam_idx):
        mask    = masks_by_sam_idx[sam_idx]
        pos     = foot_position(mask)
        if pos is None:
            continue
        px_pix, py_pix = pos
        src_frame  = sam_idx * frame_step
        timestamp  = round(src_frame / source_fps, 4)

        px_pitch = py_pitch = float("nan")
        if H is not None:
            px_pitch, py_pitch = apply_homography(
                px_pix, py_pix, H, pitch_length, pitch_width
            )

        records.append({
            "frame_idx":    sam_idx,
            "source_frame": src_frame,
            "timestamp_s":  timestamp,
            "pixel_x":      px_pix,
            "pixel_y":      py_pix,
            "pitch_x":      px_pitch,
            "pitch_y":      py_pitch,
        })

    df = pd.DataFrame(records).astype({
        "frame_idx":    "int32",
        "source_frame": "int32",
        "timestamp_s":  "float64",
        "pixel_x":      "float32",
        "pixel_y":      "float32",
        "pitch_x":      "float32",
        "pitch_y":      "float32",
    })
    schema = pa.schema([
        ("frame_idx",    pa.int32()),
        ("source_frame", pa.int32()),
        ("timestamp_s",  pa.float64()),
        ("pixel_x",      pa.float32()),
        ("pixel_y",      pa.float32()),
        ("pitch_x",      pa.float32()),
        ("pitch_y",      pa.float32()),
    ])
    pq.write_table(
        pa.Table.from_pandas(df, schema=schema, preserve_index=False),
        output_path,
    )

    # --- Meta JSON ---
    meta_path = output_path.parent / output_path.name.replace("tracks_", "meta_").replace(".parquet", ".json")
    meta = {
        "clip":                str(args.clip),
        "click_source_frame":  click_src,
        "click_sam_idx":       click_sam_idx,
        "click_x":             args.x,
        "click_y":             args.y,
        "frames_tracked":      len(records),
        "processing_device":   str(device),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nDone. {len(records)} frames tracked.")
    print(f"Parquet → {output_path}")
    print(f"Meta    → {meta_path}")

    if records:
        valid = df.dropna(subset=["pitch_x"])
        if not valid.empty:
            print(f"pitch_x range: {valid['pitch_x'].min():.1f} – {valid['pitch_x'].max():.1f} m")
            print(f"pitch_y range: {valid['pitch_y'].min():.1f} – {valid['pitch_y'].max():.1f} m")


if __name__ == "__main__":
    main()
