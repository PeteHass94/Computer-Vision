"""
Usage: python src/classify_teams.py [--video clips/dev_clip.mp4] [--tracks output/tracks.parquet]
                                     [--seeds calibration/team_colours_seed.json]

Post-processing step: assigns each detection to a named team (0=team_a, 1=team_b, 2=referee).

Strategy: temporal accumulation per track_id.
  For each unique track, pixel evidence is pooled across ALL frames where the track appears.
  This makes classification robust even when individual frames have only 1-2 jersey pixels
  (common in wide-angle / panoramic footage where players are small).

Classification uses hue-window pixel counting:
  Each seed defines a hue window (±hue_window OpenCV units) and sat_min.
  The seed whose window captures the most pixels over the full track life wins,
  provided it exceeds MIN_TRACK_PIXELS.

Outputs:
  output/tracks.parquet   — updated `team` int8 column (0/1/2, or -1 if unresolved)
  output/team_colours.json — per-cluster HSV centroid + sample RGB for Streamlit dot colours
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

OUTPUT_DIR     = Path("output")
CALIBRATION_DIR = Path("calibration")


def seeds_path_from_video(video_path: Path) -> Path:
    return CALIBRATION_DIR / f"team_colours_{video_path.stem}.json"

# Crop region: full bbox width, upper body only
CROP_X_LO, CROP_X_HI = 0.10, 0.90
CROP_Y_LO, CROP_Y_HI = 0.15, 0.55

# Global pixel quality filters applied before per-seed hue-window counting
V_MIN =  30
V_MAX = 230

# Pitch background exclusion — covers yellow-green through cyan-green (H=20-95 on 0-179 scale).
# Wider than classic grass hue because yellow-green turf infill at H=25-44 would otherwise
# pollute all crops and swamp the hue-window vote.
PITCH_HUE_LO = 20
PITCH_HUE_HI = 95
PITCH_S_MIN  = 50

# Minimum total pixels in winning bin (across all frames for this track_id) for a confident call
MIN_TRACK_PIXELS = 10

HUE_WINDOW_DEFAULT = 15


def extract_crop_arrays(
    frame: np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Crop the chest region, remove pitch background, apply V filter.
    Returns (h, s, v) flat arrays, or None if fewer than 2 pixels survive.
    """
    h_frame, w_frame = frame.shape[:2]
    bw = x2 - x1
    bh = y2 - y1

    cx1 = max(0, int(x1 + bw * CROP_X_LO))
    cx2 = min(w_frame, int(x1 + bw * CROP_X_HI))
    cy1 = max(0, int(y1 + bh * CROP_Y_LO))
    cy2 = min(h_frame, int(y1 + bh * CROP_Y_HI))

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = frame[cy1:cy2, cx1:cx2]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    is_pitch = (h_ch >= PITCH_HUE_LO) & (h_ch <= PITCH_HUE_HI) & (s_ch >= PITCH_S_MIN)
    mask = (v_ch >= V_MIN) & (v_ch <= V_MAX) & ~is_pitch
    if mask.sum() < 2:
        return None

    return h_ch[mask], s_ch[mask], v_ch[mask]


def count_hue_window_pixels(
    h: np.ndarray, s: np.ndarray,
    hue_center: float, hue_half_width: float, sat_min: float,
) -> int:
    """Count pixels with circular hue within hue_half_width of hue_center and s >= sat_min."""
    diff = np.abs(h - hue_center)
    diff = np.minimum(diff, 180.0 - diff)
    return int(((diff <= hue_half_width) & (s >= sat_min)).sum())


def read_frames_for_rows(video_path: Path, df: pd.DataFrame):
    """Single sequential video pass; yields (row_indices, frame) per needed frame."""
    needed = set(df["frame_id"].unique())
    idx_by_frame = df.groupby("frame_id").indices
    cap = cv2.VideoCapture(str(video_path))
    current = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current in needed:
                yield idx_by_frame[current], frame
            current += 1
    finally:
        cap.release()


def hsv_to_rgb(hue_0_179: float, sat_0_255: float, val_0_255: float) -> list[int]:
    pixel = np.array([[[hue_0_179, sat_0_255, val_0_255]]], dtype=np.uint8)
    return cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0, 0].tolist()


def hue_name(hue_0_179: float, sat: float) -> str:
    if sat < 60:
        return "grey/white"
    deg = hue_0_179 * 2
    if deg < 15 or deg >= 345: return "red"
    if deg < 45:  return "orange/yellow"
    if deg < 75:  return "yellow/green"
    if deg < 150: return "green"
    if deg < 195: return "cyan"
    if deg < 255: return "blue"
    if deg < 285: return "purple"
    return "pink/red"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",  type=Path, default=Path("clips/dev_clip.mp4"))
    p.add_argument("--tracks", type=Path, default=Path("output/tracks.parquet"))
    p.add_argument("--seeds",  type=Path, default=None,
                   help="Override seeds file (default: calibration/team_colours_{clip}.json)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.seeds is None:
        args.seeds = seeds_path_from_video(args.video)

    for p in (args.video, args.tracks, args.seeds):
        if not p.exists():
            sys.exit(f"File not found: {p}")

    seeds_cfg = json.loads(args.seeds.read_text())
    seed_names  = ["team_a", "team_b", "referee"]
    seed_labels = {0: "team_a", 1: "team_b", 2: "referee"}

    seed_params = []
    for name in seed_names:
        cfg = seeds_cfg[name]
        seed_params.append({
            "hue":        float(cfg["hue"]),
            "hue_window": float(cfg.get("hue_window", HUE_WINDOW_DEFAULT)),
            "sat_min":    float(cfg.get("sat_min", 0.0)),
            "sat_max":    float(cfg["sat_max"]) if "sat_max" in cfg else None,
            "catchall":   bool(cfg.get("catchall", False)),
        })

    df = pd.read_parquet(args.tracks)
    n  = len(df)
    print(f"Loaded {n} detections across {df['frame_id'].nunique()} frames, "
          f"{df['track_id'].nunique()} tracks")
    print(f"Seeds: {[(name, seed_params[i]) for i, name in enumerate(seed_names)]}")

    # --- Model-class pre-assignment (football model only) ---
    # If the parquet has a class_name column (from process_football.py), referee and ball
    # detections are already identified — skip HSV for them.
    model_assigned: dict[int, int] = {}  # track_id → team_id, pre-assigned from model class
    hsv_skip_tids: set[int] = set()
    if "class_name" in df.columns:
        print("Football model detected — pre-assigning referee/ball tracks from class labels…")
        for tid, grp in df.groupby("track_id"):
            # Majority class across all detections for this track
            majority_class = grp["class_name"].mode().iloc[0]
            if majority_class == "referee":
                model_assigned[int(tid)] = 2   # team index 2 = referee seed
                hsv_skip_tids.add(int(tid))
            elif majority_class == "ball":
                model_assigned[int(tid)] = -1  # no team
                hsv_skip_tids.add(int(tid))
        n_ref  = sum(1 for v in model_assigned.values() if v == 2)
        n_ball = sum(1 for v in model_assigned.values() if v == -1)
        print(f"  Pre-assigned: {n_ref} referee tracks, {n_ball} ball tracks → HSV skipped")

    # Filter df to only rows needing HSV (excludes pre-assigned tracks)
    df_hsv = df[~df["track_id"].isin(hsv_skip_tids)] if hsv_skip_tids else df

    # --- Accumulate per-track pixel evidence (single video pass) ---
    print("Accumulating hue-window pixel counts per track (single video pass)…")

    # Saturation threshold for "highly saturated" pixel (0-255 scale)
    HIGH_SAT_THRESH = 80

    k = len(seed_names)
    track_bin_counts    = defaultdict(lambda: np.zeros(k, dtype=np.int32))
    track_high_sat_frac = defaultdict(list)   # fraction of pixels with S > HIGH_SAT_THRESH

    for row_indices, frame in read_frames_for_rows(args.video, df_hsv):
        for ri in row_indices:
            row = df.iloc[ri]
            tid = int(row.track_id)
            arrays = extract_crop_arrays(frame, row.x1, row.y1, row.x2, row.y2)
            if arrays is None:
                continue
            h, s, v = arrays
            track_high_sat_frac[tid].append(float((s > HIGH_SAT_THRESH).mean()))
            for seed_id, params in enumerate(seed_params):
                cnt = count_hue_window_pixels(
                    h, s, params["hue"], params["hue_window"], params["sat_min"]
                )
                track_bin_counts[tid][seed_id] += cnt

    # --- Classify each track based on accumulated counts ---
    track_team: dict[int, int] = {}
    for tid in track_bin_counts:
        counts = track_bin_counts[tid]
        # Fraction of crop pixels (across all track frames) that were highly saturated.
        # Black kits → near 0.0; coloured kits → 0.15+ even in noisy wide-angle crops.
        frac = float(np.mean(track_high_sat_frac[tid])) if track_high_sat_frac[tid] else 1.0

        # sat_max (0-1 fraction) pre-filter: dark kits bypass hue competition.
        assigned = False
        for seed_id, params in enumerate(seed_params):
            if params["sat_max"] is not None and frac < params["sat_max"]:
                track_team[tid] = seed_id
                assigned = True
                break
        if assigned:
            continue

        # Exclude catchall seeds from hue competition; they win by elimination.
        non_catchall = [i for i, p in enumerate(seed_params) if not p["catchall"]]
        catchall_ids = [i for i, p in enumerate(seed_params) if p["catchall"]]

        if non_catchall:
            sub_counts = counts[non_catchall]
            best_sub = int(np.argmax(sub_counts))
            if sub_counts[best_sub] >= MIN_TRACK_PIXELS:
                track_team[tid] = non_catchall[best_sub]
            elif catchall_ids:
                track_team[tid] = catchall_ids[0]
            else:
                track_team[tid] = -1
        else:
            best_id = int(np.argmax(counts))
            track_team[tid] = best_id if counts[best_id] >= MIN_TRACK_PIXELS else -1

    # Merge model-class pre-assignments with HSV classifications
    track_team.update(model_assigned)

    # --- Back-assign all detections from their track's classification ---
    team = np.array(
        [track_team.get(int(tid), -1) for tid in df["track_id"]],
        dtype=np.int8,
    )
    df["team"] = team

    resolved = int((team >= 0).sum())
    unresolved_tracks = sum(1 for v in track_team.values() if v == -1)
    print(f"  Tracks resolved: {resolved} / {n} detections  "
          f"({unresolved_tracks} tracks with insufficient evidence)")

    # --- Save updated parquet ---
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
        ("team",        pa.int8()),
    ]
    schema = pa.schema(fields)
    pq.write_table(
        pa.Table.from_pandas(df, schema=schema, preserve_index=False),
        args.tracks,
    )
    print(f"Updated parquet → {args.tracks}")

    # --- Summary ---
    print("\n--- Assignment summary ---")
    # Collect winning-bin HSV per track for colour output
    track_avg_hsv: dict[int, tuple[float, float, float]] = {}
    for tid, team_id in track_team.items():
        if team_id < 0:
            continue
        params = seed_params[team_id]
        # Use seed hue/sat as fallback representative colour
        track_avg_hsv[tid] = (params["hue"], params["sat_min"], 120.0)

    colour_data = {}
    for cluster_id, cluster_name in seed_labels.items():
        mask = team == cluster_id
        count = int(mask.sum())
        params = seed_params[cluster_id]
        avg_h = params["hue"]
        avg_s = params["sat_min"] if params["sat_min"] > 0 else 150.0
        avg_v = 120.0
        colour_data[str(cluster_id)] = {
            "name":       cluster_name,
            "hue_0_179":  round(avg_h, 2),
            "sat_0_255":  round(avg_s, 2),
            "val_0_255":  round(avg_v, 2),
            "sample_rgb": hsv_to_rgb(avg_h, min(avg_s, 255), avg_v),
        }
        n_tracks = sum(1 for tid, v in track_team.items() if v == cluster_id)
        print(
            f"  [{cluster_id}] {cluster_name:<10}  {count:5d} detections  "
            f"{n_tracks} tracks  seed hue={params['hue']:.0f}"
        )

    unassigned = int((team == -1).sum())
    if unassigned:
        print(f"  [-1] unassigned: {unassigned} detections  "
              f"({unresolved_tracks} tracks with < {MIN_TRACK_PIXELS} qualifying pixels)")

    colours_path = OUTPUT_DIR / "team_colours.json"
    with open(colours_path, "w") as f:
        json.dump(colour_data, f, indent=2)
    print(f"\nTeam colours saved → {colours_path}")
    print("\nIf clusters look wrong, adjust hue_window/sat_min in team_colours_seed.json and re-run.")


if __name__ == "__main__":
    main()
