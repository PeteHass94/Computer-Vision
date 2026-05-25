import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import viewer_utils as vu

st.set_page_config(page_title="Generic YOLO", layout="wide")

clip_path   = vu.clip_selector(exclude=["clips/dev_clip3.mp4"])
tracks_path = Path(f"output/tracks_generic_{clip_path.stem}.parquet")

st.title("1 · Generic YOLO")
st.caption("YOLOv8m trained on COCO — detects all people, team classification via HSV colour seeds.")

pipeline_cmd = (
    f"python src/process.py {clip_path}\n"
    f"python src/apply_homography.py --tracks {tracks_path}\n"
    f"python src/classify_teams.py --video {clip_path} --tracks {tracks_path}"
)

vu.render_viewer(
    page_key="generic",
    tracks_path=tracks_path,
    clip_path=clip_path,
    pipeline_cmd=pipeline_cmd,
)
