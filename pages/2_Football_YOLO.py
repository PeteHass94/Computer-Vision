import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import viewer_utils as vu

st.set_page_config(page_title="Football YOLO", layout="wide")

clip_path   = vu.clip_selector()
tracks_path = Path(f"output/tracks_football_{clip_path.stem}.parquet")

st.title("2 · Football YOLO")
st.caption(
    "Football-specific YOLOv8 model trained on player / goalkeeper / referee / ball.  \n"
    "Source: [roboflow/sports](https://github.com/roboflow/sports) — "
    "roboflow-jvuqo/football-players-detection-3zvbc"
)

pipeline_cmd = (
    f"python src/process_football.py {clip_path}\n"
    f"python src/apply_homography.py --tracks {tracks_path}\n"
    f"python src/classify_teams.py --video {clip_path} --tracks {tracks_path}"
)

vu.render_viewer(
    page_key="football",
    tracks_path=tracks_path,
    clip_path=clip_path,
    pipeline_cmd=pipeline_cmd,
)
