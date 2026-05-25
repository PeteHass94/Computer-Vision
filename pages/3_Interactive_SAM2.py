import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import viewer_utils as vu

st.set_page_config(page_title="Interactive SAM 2", layout="wide")

# ---------------------------------------------------------------------------
# Clip selector — only clips that have SAM 2 data
# ---------------------------------------------------------------------------
SAM2_CLIPS = [p for p in vu.CLIPS if Path(f"output/tracks_sam2_{Path(p).stem}.parquet").exists()]
if not SAM2_CLIPS:
    SAM2_CLIPS = vu.CLIPS

_sam2_key = "sam2_selected_clip"
if _sam2_key not in st.session_state or st.session_state[_sam2_key] not in SAM2_CLIPS:
    st.session_state[_sam2_key] = SAM2_CLIPS[0]

selected = st.sidebar.selectbox(
    "Dev clip",
    options=SAM2_CLIPS,
    index=SAM2_CLIPS.index(st.session_state[_sam2_key]),
    format_func=lambda p: Path(p).name,
    key="_sam2_clip_select",
)
st.session_state[_sam2_key] = selected
if st.sidebar.button("Reload data"):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
CLIP_PATH    = Path(selected)
PARQUET_PATH = Path(f"output/tracks_sam2_{CLIP_PATH.stem}.parquet")
META_PATH    = Path(f"output/meta_sam2_{CLIP_PATH.stem}.json")
HOM_PATH     = vu.homography_path(CLIP_PATH)

st.title("Track a single player with SAM 2")

if not PARQUET_PATH.exists():
    st.info(
        f"No SAM 2 tracks found for **{CLIP_PATH.name}** yet. Run the tracker first:\n\n"
        "```bash\n"
        f"python src/process_sam2.py --clip {CLIP_PATH} --frame <N> --x <X> --y <Y>\n"
        "```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df   = pd.read_parquet(PARQUET_PATH)
meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
hom  = vu.load_homography(str(HOM_PATH)) if HOM_PATH.exists() else {}

pitch_length  = hom.get("pitch_length_m", 105.0)
pitch_width   = hom.get("pitch_width_m",  68.0)
click_src     = meta.get("click_source_frame", 0)
click_sam_idx = meta.get("click_sam_idx", 0)
click_x       = meta.get("click_x", 0)
click_y       = meta.get("click_y", 0)

# Clamp off-pitch coords for display; parquet unchanged
df_disp = df.copy()
df_disp["pitch_x"] = df_disp["pitch_x"].clip(0, pitch_length)
df_disp["pitch_y"] = df_disp["pitch_y"].clip(0, pitch_width)
valid = df_disp.dropna(subset=["pitch_x", "pitch_y"]).sort_values("frame_idx")

# ---------------------------------------------------------------------------
# Derived display values
# ---------------------------------------------------------------------------
n_tracked  = len(df)
first_row  = valid.iloc[0]  if not valid.empty else None
last_row   = valid.iloc[-1] if not valid.empty else None
start_px   = float(first_row["pitch_x"]) if first_row is not None else 0.0
end_px     = float(last_row["pitch_x"])  if last_row  is not None else 0.0

attack_arrow = "←" if end_px < start_px else "→"
attack_label = f"{attack_arrow} Enniskillen Rangers attacking direction"
attack_dir   = "left" if end_px < start_px else "right"

click_row_ts = df[df["frame_idx"] == click_sam_idx]
click_ts = float(click_row_ts.iloc[0]["timestamp_s"]) if not click_row_ts.empty else None
timing_str = ("from kickoff" if click_ts == 0.0
              else f"from t={click_ts:.1f}s" if click_ts is not None
              else f"from frame {click_src}")

st.markdown(
    "Meta's Segment Anything 2 model takes a single click as input and tracks "
    "that specific object across the video. "
    f"Enniskillen Rangers player tracked **{timing_str}** with one click "
    f"(frame {click_src}, pixel {click_x}, {click_y}) — "
    f"**{n_tracked} frames** of continuous tracking through the counter-attack "
    f"down the {attack_dir} wing. No bounding box, no colour threshold, no training data."
)

# ---------------------------------------------------------------------------
# Pitch drawing helpers
# ---------------------------------------------------------------------------
def _add_pitch_shapes(fig: go.Figure, pl: float, pw: float):
    fig.add_shape(type="rect", x0=0, y0=0, x1=pl, y1=pw,
                  fillcolor="#3a7d44", line=dict(color="white", width=2), layer="below")
    fig.add_shape(type="line", x0=pl/2, y0=0, x1=pl/2, y1=pw,
                  line=dict(color="white", width=2))
    r, cx, cy = 9.15, pl/2, pw/2
    fig.add_shape(type="circle", x0=cx-r, y0=cy-r, x1=cx+r, y1=cy+r,
                  line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)")
    pb_w, pb_d = 40.32, 16.5
    pb_y0 = (pw - pb_w) / 2
    for x0, x1 in [(0, pb_d), (pl-pb_d, pl)]:
        fig.add_shape(type="rect", x0=x0, y0=pb_y0, x1=x1, y1=pb_y0+pb_w,
                      line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)")


def _pitch_layout(fig: go.Figure, pl: float, pw: float, height: int = 380):
    # 1. y-axis reversed: [pw+2, -2] so pitch top is top of screen
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[-2, pl+2], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[pw+2, -2], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="#3a7d44",
        paper_bgcolor="#1e1e1e",
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
        height=height,
    )


# Trajectory colouring — split into thirds (yellow → orange → red)
thirds            = np.array_split(valid, 3)
seg_colours       = ["#ffdd00", "#ff8800", "#cc1111"]
seg_colours_faded = ["rgba(255,221,0,0.35)", "rgba(255,136,0,0.35)", "rgba(204,17,17,0.35)"]

# ---------------------------------------------------------------------------
# Slider + playback state  (defined before all renders so current_idx is ready)
# ---------------------------------------------------------------------------
min_f     = int(df["frame_idx"].min())
max_f     = int(df["frame_idx"].max())
frame_key = "sam2_frame"
play_key  = "sam2_playing"

# Reset playback when clip changes
if st.session_state.get("sam2_last_clip") != selected:
    st.session_state[frame_key]        = min(click_sam_idx, max_f)
    st.session_state[play_key]         = False
    st.session_state["sam2_last_clip"] = selected

if frame_key not in st.session_state:
    st.session_state[frame_key] = min(click_sam_idx, max_f)
if play_key not in st.session_state:
    st.session_state[play_key] = False


def _on_slider():
    st.session_state[frame_key] = st.session_state["_sam2_slider"]
    st.session_state[play_key]  = False  # dragging stops playback


current_idx = st.session_state[frame_key]
current_row = valid[valid["frame_idx"] == current_idx]
current_all = df[df["frame_idx"] == current_idx]

# ---------------------------------------------------------------------------
# Row 1: seed click frame (full width)
# ---------------------------------------------------------------------------
st.subheader(f"Frame {click_src} (click frame)")
raw = vu.fetch_frame(str(CLIP_PATH), click_src)
if raw is not None:
    ann = raw.copy()
    cv2.circle(ann, (click_x, click_y), 22, (0, 255, 0), 3)
    cv2.circle(ann, (click_x, click_y),  6, (0, 255, 0), -1)
    cv2.line(ann, (click_x-35, click_y), (click_x+35, click_y), (0, 255, 0), 2)
    cv2.line(ann, (click_x, click_y-35), (click_x, click_y+35), (0, 255, 0), 2)
    st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
st.caption(f"Green crosshair = seed click ({click_x}, {click_y}) — fixed reference frame")

# ---------------------------------------------------------------------------
# Row 2: pitch trajectory + heatmap (full width)
# ---------------------------------------------------------------------------
st.markdown(
    f"<div style='text-align:center; color:#ccc; font-size:0.9rem; margin:8px 0 4px'>"
    f"{attack_label}</div>",
    unsafe_allow_html=True,
)

fig_overview = go.Figure()
_add_pitch_shapes(fig_overview, pitch_length, pitch_width)

fig_overview.add_trace(go.Histogram2d(
    x=valid["pitch_x"], y=valid["pitch_y"],
    colorscale=[
        [0.0, "rgba(0,0,0,0)"],
        [0.1, "rgba(255,220,0,0.20)"],
        [0.5, "rgba(255,120,0,0.45)"],
        [1.0, "rgba(220,30,30,0.75)"],
    ],
    xbins=dict(start=0, end=pitch_length, size=4),
    ybins=dict(start=0, end=pitch_width,  size=4),
    showscale=False,
    name="Density",
))

# Time-graded trajectory
for seg, col in zip(thirds, seg_colours):
    if len(seg) < 2:
        continue
    fig_overview.add_trace(go.Scatter(
        x=seg["pitch_x"], y=seg["pitch_y"],
        mode="lines",
        line=dict(color=col, width=2.5),
        showlegend=False,
        hoverinfo="skip",
    ))

# Start marker (green circle, "START" text)
if first_row is not None:
    fig_overview.add_trace(go.Scatter(
        x=[first_row["pitch_x"]], y=[first_row["pitch_y"]],
        mode="markers+text",
        marker=dict(color="lime", size=14, symbol="circle",
                    line=dict(color="white", width=2)),
        text=["START"],
        textposition="top center",
        textfont=dict(color="white", size=10),
        name="Start",
    ))

# 3. End marker — dot only, no inline text label
if last_row is not None:
    fig_overview.add_trace(go.Scatter(
        x=[last_row["pitch_x"]], y=[last_row["pitch_y"]],
        mode="markers",
        marker=dict(color="#ff3333", size=14, symbol="circle",
                    line=dict(color="white", width=2)),
        name="End",
    ))

_pitch_layout(fig_overview, pitch_length, pitch_width, height=420)
st.plotly_chart(fig_overview, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 3: current playback frame (full width)
# ---------------------------------------------------------------------------
if not current_all.empty:
    cur_src = int(current_all.iloc[0]["source_frame"])
    cur_px  = int(current_all.iloc[0]["pixel_x"])
    cur_py  = int(current_all.iloc[0]["pixel_y"])
    cur_ts  = float(current_all.iloc[0]["timestamp_s"])
    cur_raw = vu.fetch_frame(str(CLIP_PATH), cur_src)
    if cur_raw is not None:
        cur_ann = cur_raw.copy()
        cv2.circle(cur_ann, (cur_px, cur_py), 28, (255, 60, 60), -1)
        cv2.circle(cur_ann, (cur_px, cur_py), 28, (255, 255, 255), 3)
        st.image(cv2.cvtColor(cur_ann, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.caption(
        f"Current frame: {current_idx} (t={cur_ts:.1f}s)  ·  "
        f"Player position: ({cur_px}, {cur_py})"
    )
else:
    st.caption(f"Current frame: {current_idx} — no tracked position")

# ---------------------------------------------------------------------------
# Row 4: slider + Prev / Play / Next buttons
# ---------------------------------------------------------------------------
st.slider(
    "Frame", min_value=min_f, max_value=max_f,
    value=current_idx,
    key="_sam2_slider", on_change=_on_slider,
)

btn_prev, btn_play, btn_next = st.columns([1, 2, 1])

with btn_prev:
    if st.button("⏮ Prev", use_container_width=True):
        st.session_state[frame_key] = max(min_f, current_idx - 1)
        st.session_state[play_key]  = False
        st.rerun()

with btn_play:
    play_label = "⏸ Pause" if st.session_state[play_key] else "▶ Play"
    if st.button(play_label, use_container_width=True):
        # Toggle; if we were at the end, restart from beginning
        if not st.session_state[play_key] and current_idx >= max_f:
            st.session_state[frame_key] = min_f
        st.session_state[play_key] = not st.session_state[play_key]
        st.rerun()

with btn_next:
    if st.button("Next ⏭", use_container_width=True):
        st.session_state[frame_key] = min(max_f, current_idx + 1)
        st.session_state[play_key]  = False
        st.rerun()

# Playback tick — advance one frame, sleep, rerun
if st.session_state[play_key]:
    if current_idx < max_f:
        time.sleep(0.1)
        st.session_state[frame_key] = current_idx + 1
        st.rerun()
    else:
        st.session_state[play_key] = False

# ---------------------------------------------------------------------------
# Row 5: metrics
# ---------------------------------------------------------------------------
dx       = valid["pitch_x"].diff()
dy       = valid["pitch_y"].diff()
distance = float(np.sqrt(dx**2 + dy**2).sum())

x_bins = pd.cut(valid["pitch_x"], bins=5, labels=False)
y_bins = pd.cut(valid["pitch_y"], bins=3, labels=False)
n_zones = len(set(zip(x_bins.dropna().astype(int), y_bins.dropna().astype(int))))

m1, m2, m3 = st.columns(3)
m1.metric("Frames tracked",     f"{n_tracked:,}")
m2.metric("Distance covered",   f"{distance:.0f} m")
m3.metric("Pitch zones visited", f"{n_zones} / 15")

st.divider()
st.caption(
    "SAM 2 — Segment Anything Model 2 (Meta AI, 2024) · "
    "Model: sam2_hiera_tiny · Device: " + meta.get("processing_device", "mps")
)
