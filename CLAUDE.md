# Football CV PoC

## Project Goal
Build a Streamlit app that processes Veo football match footage (fixed wide-angle, full pitch visible) and displays:
1. Original video with YOLOv8 detections + ByteTrack IDs overlaid
2. Synchronised 2D pitch view showing each tracked player as a dot, with positions mapped from pixel space to pitch coordinates via homography
3. A frame slider to scrub through both views

This is a proof of concept. Prioritise working end-to-end over polish.

## Tech Stack
- Python 3.11
- ultralytics (YOLOv8)
- supervision (Roboflow — provides ByteTrack)
- opencv-python
- numpy, pandas, pyarrow
- streamlit, plotly
- ffmpeg (system dependency)

## Footage Context
- Source: Veo camera, fixed wide-angle, full pitch in frame
- Camera does not pan or zoom — homography is one-time calibration per match
- Players appear small in frame — use imgsz=1280 for YOLO inference, not default 640
- Working with 2-minute dev clips in clips/, not full matches

## Architecture Principles
- Heavy CV processing runs once, offline, in src/process.py
- Results saved to output/tracks.parquet (per-frame player positions) and output/annotated.mp4
- Streamlit only reads these artefacts — never invokes YOLO directly
- Run detection at 10fps not 30fps (skip frames in cv2.VideoCapture)
- Cache aggressively: if tracks.parquet exists for a clip, don't reprocess

## Parquet Schema
Columns: frame_idx, track_id, pixel_x, pixel_y, pitch_x, pitch_y, bbox_w, bbox_h
One row per player per frame.

## Out of Scope for v1
- Ball detection
- Team classification (all players one colour)
- Event detection (shots, passes, goals)
- Per-player stats or heatmaps
- Processing full 2-hour matches
- Real-time streaming

## Coding Conventions
- Use a virtual environment (.venv/), gitignored
- Pin versions in requirements.txt
- Type hints on function signatures
- One responsibility per script in src/
- No notebooks in main pipeline; notebooks/ folder for exploration only

## What Claude Code Should Do
- Ask before installing system packages (homebrew, ffmpeg)
- Always activate .venv before running Python
- After processing, show me the output paths and file sizes
- If YOLO detection looks bad, suggest diagnostics (check imgsz, check frame quality) before suggesting fine-tuning
- Don't add features listed in "Out of Scope" without asking