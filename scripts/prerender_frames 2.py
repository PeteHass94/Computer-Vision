"""
Pre-render video frames as JPEGs for the read-only deploy branch.

For each clip in CLIPS:
  - Sample at frame_step (same rate used during tracking, ~10fps)
  - Downscale to TARGET_HEIGHT px tall (maintain aspect ratio)
  - Save as assets/frames/{clip_stem}/frame_{source_idx:06d}.jpg
  - Write assets/frames/{clip_stem}/meta.json with fps/frame_step/total

Run from the project root:
  python scripts/prerender_frames.py [--clips clip3 clip4 clip5]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

TARGET_HEIGHT = 540
JPEG_QUALITY  = 45
TARGET_FPS    = 10

CLIPS = {
    "clip3": "clips/dev_clip3.mp4",
    "clip4": "clips/dev_clip4.mp4",
    "clip5": "clips/dev_clip5.mp4",
}


def render_clip(stem: str, clip_path: str):
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  SKIP — cannot open {clip_path}")
        return

    fps         = cap.get(cv2.CAP_PROP_FPS)
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_step  = max(1, round(fps / TARGET_FPS))
    out_w       = round(src_w * TARGET_HEIGHT / src_h)
    n_sampled   = (total + frame_step - 1) // frame_step

    out_dir = Path(f"assets/frames/{stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "clip":       clip_path,
        "fps":        fps,
        "frame_step": frame_step,
        "total":      total,
        "out_w":      out_w,
        "out_h":      TARGET_HEIGHT,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"  {stem}: {src_w}x{src_h} → {out_w}x{TARGET_HEIGHT}  "
          f"step={frame_step}  {n_sampled} frames to render")

    src_idx    = 0
    rendered   = 0
    skipped    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if src_idx % frame_step == 0:
            dst_path = out_dir / f"frame_{src_idx:06d}.jpg"
            if dst_path.exists():
                skipped += 1
            else:
                small = cv2.resize(frame, (out_w, TARGET_HEIGHT),
                                   interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dst_path), small,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                rendered += 1

            if (rendered + skipped) % 200 == 0:
                print(f"    … {rendered + skipped}/{n_sampled}", end="\r", flush=True)

        src_idx += 1

    cap.release()
    dir_mb = sum(p.stat().st_size for p in out_dir.glob("*.jpg")) / 1024**2
    print(f"  {stem}: rendered={rendered}  skipped(exist)={skipped}  "
          f"dir size={dir_mb:.1f} MB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", nargs="+", choices=list(CLIPS.keys()),
                   default=list(CLIPS.keys()),
                   help="Which clips to render (default: all)")
    args = p.parse_args()

    for key in args.clips:
        clip_path = CLIPS[key]
        stem      = Path(clip_path).stem
        if not Path(clip_path).exists():
            print(f"SKIP {key} — {clip_path} not found")
            continue
        print(f"Rendering {key} ({clip_path}) …")
        render_clip(stem, clip_path)

    # Summary
    total_mb = sum(
        p.stat().st_size for d in Path("assets/frames").iterdir()
        if d.is_dir() for p in d.glob("*.jpg")
    ) / 1024**2
    print(f"\nTotal assets/frames size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
