"""
Usage: python src/apply_referee_constraint.py --tracks output/tracks_football_dev_clip4.parquet

Max-one-referee constraint: for each frame, keep only the highest-confidence
referee detection and reclassify all others as class_name="player", class_id=2.

Writes back to the same parquet in-place.
"""

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

CLASS_PLAYER  = 2
CLASS_REFEREE = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.tracks.exists():
        sys.exit(f"Tracks not found: {args.tracks}")

    df = pd.read_parquet(args.tracks)

    if "class_name" not in df.columns:
        sys.exit("No class_name column — this script is for football parquets only.")

    original_ref_count = int((df["class_name"] == "referee").sum())

    reclassified = 0
    for frame_id, grp in df.groupby("frame_id"):
        ref_mask = grp["class_name"] == "referee"
        ref_rows = grp[ref_mask]
        if len(ref_rows) <= 1:
            continue
        # Keep the highest-confidence one; reclassify the rest
        keep_idx = ref_rows["confidence"].idxmax()
        demote_idx = ref_rows.index[ref_rows.index != keep_idx]
        df.loc[demote_idx, "class_name"] = "player"
        df.loc[demote_idx, "class_id"]   = CLASS_PLAYER
        reclassified += len(demote_idx)

    final_ref_count = int((df["class_name"] == "referee").sum())

    print(f"Tracks: {args.tracks}")
    print(f"  Original referee detections : {original_ref_count:,}")
    print(f"  Reclassified to player      : {reclassified:,}")
    print(f"  Final referee detections    : {final_ref_count:,}")

    # Rebuild schema preserving all existing columns
    col_types = {
        "frame_id":    pa.int32(),
        "timestamp_s": pa.float64(),
        "track_id":    pa.int32(),
        "class_id":    pa.int8(),
        "class_name":  pa.string(),
        "x1":          pa.float32(),
        "y1":          pa.float32(),
        "x2":          pa.float32(),
        "y2":          pa.float32(),
        "confidence":  pa.float32(),
        "pitch_x":     pa.float32(),
        "pitch_y":     pa.float32(),
    }
    if "team" in df.columns:
        col_types["team"] = pa.int8()

    fields = [(c, col_types[c]) for c in df.columns if c in col_types]
    schema = pa.schema(fields)
    pq.write_table(
        pa.Table.from_pandas(df, schema=schema, preserve_index=False),
        args.tracks,
    )
    print(f"Written → {args.tracks}")


if __name__ == "__main__":
    main()
