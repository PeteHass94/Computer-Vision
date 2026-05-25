"""
conda activate streamlit_env  
streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
import viewer_utils as vu

st.set_page_config(page_title="Football CV — From Veo to Tracking Data", layout="wide")

st.info(
    "**Read-only demo.** "
    "Source code, full processing pipeline, and Veo footage are available in the "
    "[main branch on GitHub](https://github.com/phassard/football-cv).",
    icon="ℹ️",
)

vu.clip_selector()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.title("Football CV — From Veo to Tracking Data")
st.subheader("Open-source computer vision pipeline for amateur club analytics")

st.write(
    "Veo cameras are in stands across Ireland and the UK, recording every match — "
    "but the footage sits unwatched on a hard drive. "
    "This pipeline takes that raw footage and turns it into structured tracking data: "
    "every player's position on the pitch, every frame, ready for tactical analysis. "
    "The goal is to bring the kind of data analytics available to professional clubs "
    "within reach of amateur teams like Enniskillen Rangers FC."
)

st.divider()

# ---------------------------------------------------------------------------
# Detection pages
# ---------------------------------------------------------------------------
st.subheader("Detection approaches")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1 · Generic YOLO")
    st.caption("YOLOv8m trained on COCO — detects all people, team classification via HSV colour seeds.")
    st.page_link("pages/1_Generic_YOLO.py", label="Open →", use_container_width=True)

with col2:
    st.markdown("### 2 · Football YOLO")
    st.caption("Football-specific model (Roboflow Sports) — detects player, goalkeeper, referee, and ball as separate classes.")
    st.page_link("pages/2_Football_YOLO.py", label="Open →", use_container_width=True)

with col3:
    st.markdown("### 3 · Interactive SAM 2")
    st.caption("Click on players to track — Meta's SAM 2 propagates masks forward without any colour assumptions.")
    st.page_link("pages/3_Interactive_SAM2.py", label="Open →", use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.caption(
    "Built on YOLOv8, ByteTrack, OpenCV, and Streamlit. "
    "Demo clips from Enniskillen Rangers FC."
)
