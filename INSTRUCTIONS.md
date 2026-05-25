# Football CV PoC — Initial Build

## Goal
Build a Streamlit app that processes a Veo football match clip (fixed wide-angle camera, full pitch in frame) and displays:
1. The original video with YOLO bounding boxes + ByteTrack IDs overlaid
2. A synchronised 2D pitch view showing each tracked player as a dot, with positions transformed from pixel space to pitch coordinates via homography
3. A frame slider to scrub through both views together

This is a PoC. Prioritise working end-to-end over polish. No ball detection, no team classification, no event detection in v1.

## Tech Stack
- Python 3.11
- ultralytics (YOLOv8)
- supervision (Roboflow — wraps ByteTrack cleanly)
- opencv-python
- numpy, pandas, pyarrow (for parquet)
- streamlit, plotly

## Project Structure
```
football-cv-poc/
├── clips/
│   └── dev_clip.mp4         # 2-min Veo segment, gitignored
├── output/
│   ├── tracks.parquet       # per-frame player positions
│   └── annotated.mp4        # video with boxes overlaid
├── calibration/
│   └── homography.json      # 4 pixel→pitch point correspondences
├── src/
│   ├── calibrate.py         # one-time interactive homography setup
│   ├── process.py           # run YOLO + ByteTrack, save parquet + annotated video
│   └── pitch.py             # plotly pitch drawing + coordinate transform helpers
├── app.py                   # streamlit entrypoint
├── requirements.txt
└── README.md
```

## Architecture Principles
- **Heavy CV work runs once, offline**: `process.py` writes tracks.parquet and annotated.mp4. Streamlit only loads these artefacts — never invokes YOLO directly. This keeps the UI snappy and iteration fast.
- **Parquet schema**: columns = `frame_idx, track_id, pixel_x, pixel_y, pitch_x, pitch_y, bbox_w, bbox_h`. One row per player per frame.
- **Run detection at 10fps not 30fps** (use `cv2.VideoCapture` and skip frames). Tracking and visualisation don't need 30fps and it cuts processing 3x.
- **Use `imgsz=1280`** for YOLO inference — Veo players are small in frame at default 640.
- **Cache processed results aggressively**. If tracks.parquet exists for a given clip hash, don't reprocess.

## Homography Approach
Veo camera is fixed for the whole match, so homography is a one-time calibration. `calibrate.py` should:
1. Open the first frame of the clip in an OpenCV window
2. Let the user click 4 points (the 4 corners of the pitch, or 4 known points like penalty box corners)
3. Map those to real pitch coordinates in metres (pitch is 105×68m standard, but ask user for actual dimensions)
4. Compute `cv2.findHomography()` and save the matrix + corresponding points to homography.json
5. `pitch.py` exposes `pixel_to_pitch(x, y)` using the saved matrix

## Streamlit App
- File picker or dropdown of clips in `clips/`
- "Process" button if no parquet exists yet — runs `process.py`, shows progress
- Once processed: two columns
  - Left: `st.video()` of annotated.mp4, OR a frame-by-frame display using `cv2.VideoCapture` driven by the slider
  - Right: Plotly figure of the pitch (green rectangle, white lines for halfway/boxes/circle) with player dots at current frame's pitch coordinates
- Slider below both, controlling current frame
- Bonus: small text showing "N players detected in frame X"

## Deliverables for v1
1. `requirements.txt` with pinned versions
2. All scripts above, working end-to-end on a 2-min dev clip
3. README explaining: how to add a clip, run calibration, run processing, launch the app
4. `.gitignore` covering clips/, output/, calibration/, venv, __pycache__

## Out of Scope for v1
- Ball detection
- Team classification (all players one colour)
- Event detection (shots, passes, etc)
- Per-player stats or heatmaps
- Full-match processing (dev clip only)
- Real-time / streaming

Build the simplest version that satisfies the goal. Don't add features I didn't ask for.