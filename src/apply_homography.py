"""
Usage: python src/apply_homography.py --tracks output/tracks_generic_dev_clip4.parquet

Reads the pre-computed homography matrix and applies it to each detection's
foot position (bottom-centre of bbox). Overwrites pitch_x and pitch_y in the
parquet. Does not re-run YOLO or tracking.

The homography file is derived from the tracks filename automatically:
  output/tracks_generic_dev_clip4.parquet  →  calibration/homography_dev_clip4.json

Override with --homography if needed.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CALIBRATION_DIR = Path("calibration")


def clip_stem_from_tracks(tracks_path: Path) -> str:
    """Extract clip stem from tracks filename. 'tracks_generic_dev_clip4' → 'dev_clip4'."""
    stem  = tracks_path.stem                  # e.g. 'tracks_generic_dev_clip4'
    parts = stem.split("_", 2)               # ['tracks', 'generic', 'dev_clip4']
    if len(parts) < 3:
        sys.exit(
            f"Cannot derive clip name from '{stem}'. "
            "Expected pattern tracks_{{model}}_{{clipname}}. "
            "Pass --homography explicitly."
        )
    return parts[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks",     type=Path, required=True)
    p.add_argument("--homography", type=Path, default=None,
                   help="Override homography file (default: derived from tracks filename)")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.tracks.exists():
        sys.exit(f"Tracks not found: {args.tracks}")

    if args.homography is not None:
        hom_path = args.homography
    else:
        clip = clip_stem_from_tracks(args.tracks)
        hom_path = CALIBRATION_DIR / f"homography_{clip}.json"

    if not hom_path.exists():
        sys.exit(
            f"Homography not found: {hom_path}\n"
            f"Run: python src/calibrate.py --video clips/{clip_stem_from_tracks(args.tracks)}.mp4"
        )

    cal = json.loads(hom_path.read_text())
    if "homography_matrix" not in cal:
        sys.exit(
            f"{hom_path} does not contain a 'homography_matrix' key.\n"
            "Run src/calibrate.py first."
        )

    H = np.array(cal["homography_matrix"], dtype=np.float64)
    pitch_length = cal["pitch_length_m"]
    pitch_width  = cal["pitch_width_m"]

    df = pd.read_parquet(args.tracks)
    print(f"Loaded {len(df)} detections")

    # Foot position: bottom-centre of bbox
    foot_x = ((df["x1"] + df["x2"]) / 2).to_numpy(dtype=np.float32)
    foot_y = df["y2"].to_numpy(dtype=np.float32)

    # cv2.perspectiveTransform expects shape (N, 1, 2)
    pixel_pts = np.stack([foot_x, foot_y], axis=1).reshape(-1, 1, 2).astype(np.float64)
    pitch_pts = cv2.perspectiveTransform(pixel_pts, H).reshape(-1, 2)

    pitch_x = pitch_pts[:, 0].astype(np.float32)
    pitch_y = pitch_pts[:, 1].astype(np.float32)

    # Clamp to pitch bounds with a small margin (handles lens distortion near edges)
    margin = 5.0
    out_of_bounds = (
        (pitch_x < -margin) | (pitch_x > pitch_length + margin) |
        (pitch_y < -margin) | (pitch_y > pitch_width  + margin)
    )
    n_oob = int(out_of_bounds.sum())
    if n_oob:
        print(f"  {n_oob} detections projected outside pitch bounds (set to NaN)")
        pitch_x[out_of_bounds] = np.nan
        pitch_y[out_of_bounds] = np.nan

    df["pitch_x"] = pitch_x
    df["pitch_y"] = pitch_y

    # --- Off-pitch filter ---
    # Coaches, subs, photographers detected near the touchline project just
    # outside the pitch bounds. A 3m buffer keeps players legitimately near
    # the line; anything beyond that is off-pitch noise and is dropped.
    BUFFER = 2.0
    before = len(df)
    off_pitch = (
        (df["pitch_x"] < -BUFFER) | (df["pitch_x"] > pitch_length + BUFFER) |
        (df["pitch_y"] < -BUFFER) | (df["pitch_y"] > pitch_width  + BUFFER)
    )
    df = df[~off_pitch].reset_index(drop=True)
    n_filtered = before - len(df)
    if n_filtered:
        print(f"Filtered {n_filtered} off-pitch detections (>{BUFFER}m outside bounds)")

    fields = [
        ("frame_id",    pa.int32()),
        ("timestamp_s", pa.float64()),
        ("track_id",    pa.int32()),
    ]
    if "class_id" in df.columns:
        fields.append(("class_id",   pa.int8()))
    if "class_name" in df.columns:
        fields.append(("class_name", pa.string()))
    fields += [
        ("x1",          pa.float32()),
        ("y1",          pa.float32()),
        ("x2",          pa.float32()),
        ("y2",          pa.float32()),
        ("confidence",  pa.float32()),
        ("pitch_x",     pa.float32()),
        ("pitch_y",     pa.float32()),
    ]
    if "team" in df.columns:
        fields.append(("team", pa.int8()))
    schema = pa.schema(fields)
    pq.write_table(
        pa.Table.from_pandas(df, schema=schema, preserve_index=False),
        args.tracks,
    )
    print(f"Updated parquet → {args.tracks}")

    valid = (~np.isnan(pitch_x)).sum()
    print(f"\npitch_x/y populated: {valid} / {len(df)} rows")
    print(f"pitch_x range: {np.nanmin(pitch_x):.1f} – {np.nanmax(pitch_x):.1f} m")
    print(f"pitch_y range: {np.nanmin(pitch_y):.1f} – {np.nanmax(pitch_y):.1f} m")


if __name__ == "__main__":
    main()
