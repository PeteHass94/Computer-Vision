"""
Shared helpers for all Streamlit viewer pages.
Import with:  import sys; sys.path.insert(0, "src"); import viewer_utils as vu
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from match_time import frame_to_match_time, load_timing as _load_timing_raw
from match_meta import load_meta as _load_meta_raw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIPS = [
    "clips/dev_clip3.mp4",
    "clips/dev_clip4.mp4",
    "clips/dev_clip5.mp4",
]
CALIBRATION_DIR = Path("calibration")
TEAMS_PATH      = Path("calibration/teams.json")
SEEDS_PATH      = Path("calibration/team_colours_seed.json")


def homography_path(clip: Path) -> Path:
    return CALIBRATION_DIR / f"homography_{clip.stem}.json"


def timing_path(clip: Path) -> Path:
    return CALIBRATION_DIR / f"match_timing_{clip.stem}.json"


def seeds_path(clip: Path) -> Path:
    return CALIBRATION_DIR / f"team_colours_{clip.stem}.json"


def teams_path(clip: Path) -> Path:
    return CALIBRATION_DIR / f"teams_{clip.stem}.json"


# ---------------------------------------------------------------------------
# Clip selector — call on every page to keep selection in sync
# ---------------------------------------------------------------------------

def clip_selector(exclude: list[str] | None = None) -> Path:
    """Sidebar dropdown that persists the selected clip via session_state.

    exclude — list of clip paths (e.g. ["clips/dev_clip3.mp4"]) to hide.
    """
    available = [c for c in CLIPS if c not in (exclude or [])]
    if not available:
        available = CLIPS

    if "selected_clip" not in st.session_state or st.session_state.selected_clip not in available:
        st.session_state.selected_clip = available[0]

    choice = st.sidebar.selectbox(
        "Dev clip",
        options=available,
        index=available.index(st.session_state.selected_clip),
        format_func=lambda p: _load_meta_raw(Path(p).name).get("match_label", Path(p).name),
    )
    st.session_state.selected_clip = choice

    if st.sidebar.button("Reload data", help="Clear cached parquet/video data"):
        st.cache_data.clear()
        st.rerun()

    return Path(choice)


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_tracks(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def load_homography(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_timing(path: str) -> dict:
    return _load_timing_raw(Path(path))


@st.cache_data
def load_teams(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_match_meta(clip_name: str) -> dict:
    return _load_meta_raw(clip_name)


@st.cache_data
def load_video_metadata(path: str) -> tuple[float, int]:
    # Deploy branch: read from pre-rendered metadata instead of opening MP4
    stem     = Path(path).stem
    meta_jpg = Path(f"assets/frames/{stem}/meta.json")
    if meta_jpg.exists():
        m = json.loads(meta_jpg.read_text())
        return m["fps"], m["total"]
    cap   = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


@st.cache_data
def colours_from_seeds(path: str) -> dict:
    """Compute display colours from calibration/team_colours_seed.json."""
    cfg        = json.loads(Path(path).read_text())
    seed_names = ["team_a", "team_b", "referee"]
    result     = {}
    for i, name in enumerate(seed_names):
        s     = cfg[name]
        pixel = np.array([[[int(s["hue"]), int(s["sat"]), int(s["val"])]]], dtype=np.uint8)
        rgb   = cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0, 0].tolist()
        result[str(i)] = {"name": name, "sample_rgb": rgb}
    return result


# ---------------------------------------------------------------------------
# Frame access
# ---------------------------------------------------------------------------

def fetch_frame(clip_path: str, source_frame_idx: int) -> np.ndarray | None:
    # Deploy branch: load pre-rendered JPEG; fall back to VideoCapture if MP4 present
    stem     = Path(clip_path).stem
    jpg_path = Path(f"assets/frames/{stem}/frame_{source_frame_idx:06d}.jpg")
    if jpg_path.exists():
        return cv2.imread(str(jpg_path))
    if not Path(clip_path).exists():
        return None
    cap = cv2.VideoCapture(clip_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def team_colour_bgr(team_id: int, colours: dict) -> tuple[int, int, int]:
    entry = colours.get(str(team_id))
    if not entry:
        return (180, 180, 180)   # grey for unassigned
    r, g, b = entry["sample_rgb"]
    return (b, g, r)


def draw_detections(frame: np.ndarray, rows: pd.DataFrame, colours: dict) -> np.ndarray:
    has_team = "team" in rows.columns
    out = frame.copy()
    for row in rows.itertuples(index=False):
        colour = team_colour_bgr(row.team if has_team else -1, colours)
        x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label = f"#{row.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def build_pitch_figure(
    rows: pd.DataFrame,
    colours: dict,
    teams: dict,
    pitch_length: float,
    pitch_width: float,
) -> go.Figure:
    fig = go.Figure()

    # Pitch surface
    fig.add_shape(type="rect", x0=0, y0=0, x1=pitch_length, y1=pitch_width,
                  fillcolor="#3a7d44", line=dict(color="white", width=2), layer="below")
    # Halfway line
    fig.add_shape(type="line",
                  x0=pitch_length / 2, y0=0, x1=pitch_length / 2, y1=pitch_width,
                  line=dict(color="white", width=2))
    # Centre circle
    r  = 9.15
    cx = pitch_length / 2
    cy = pitch_width  / 2
    fig.add_shape(type="circle",
                  x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
                  line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)")
    fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers",
                             marker=dict(color="white", size=5),
                             showlegend=False, hoverinfo="skip"))
    # Penalty boxes
    pb_w, pb_d = 40.32, 16.5
    pb_y0 = (pitch_width - pb_w) / 2
    fig.add_shape(type="rect", x0=0, y0=pb_y0, x1=pb_d, y1=pb_y0 + pb_w,
                  line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)")
    fig.add_shape(type="rect",
                  x0=pitch_length - pb_d, y0=pb_y0,
                  x1=pitch_length, y1=pb_y0 + pb_w,
                  line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)")

    # Player dots
    _id_to_key = {0: "team_a", 1: "team_b", 2: "referee", -1: None}
    plot_rows = rows.copy()
    if "team" not in plot_rows.columns:
        plot_rows["team"] = -1
    has_class = "class_name" in plot_rows.columns
    for team_id_str, grp in plot_rows.groupby(plot_rows["team"].astype(str)):
        team_id = int(team_id_str)
        rgb     = colours.get(team_id_str, {}).get("sample_rgb", [180, 180, 180])
        hex_col = "#{:02x}{:02x}{:02x}".format(*rgb)
        key     = _id_to_key.get(team_id)
        label   = teams.get(key, "Unassigned") if key else "Unassigned"
        # Ball detections: smaller dot, no track label
        is_ball  = has_class and (grp["class_name"] == "ball").all()
        dot_size = 6 if is_ball else 12
        fig.add_trace(go.Scatter(
            x=grp["pitch_x"], y=grp["pitch_y"],
            mode="markers" if is_ball else "markers+text",
            marker=dict(color=hex_col, size=dot_size, line=dict(color="white", width=1)),
            text=None if is_ball else grp["track_id"].astype(str),
            textposition="top center",
            textfont=dict(color="white", size=9),
            name="Ball" if is_ball else label,
            hovertemplate=(
                f"Track %{{text}}<br>"
                f"({grp['pitch_x'].round(1).astype(str)} m, "
                f"{grp['pitch_y'].round(1).astype(str)} m)"
                f"<extra>{'Ball' if is_ball else label}</extra>"
            ),
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[-2, pitch_length + 2], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[pitch_width + 2, -2], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="#3a7d44",
        paper_bgcolor="#1e1e1e",
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Metrics sidebar section
# ---------------------------------------------------------------------------

def _add_metrics_sidebar(tracks: pd.DataFrame, meta_path: Path) -> None:
    """Expandable Metrics section in the sidebar, aggregated across the whole clip."""
    n_frames   = tracks["frame_id"].nunique()
    total      = len(tracks)
    avg        = total / n_frames if n_frames else 0
    max_frame  = int(tracks.groupby("frame_id").size().max()) if total else 0

    if "pitch_x" in tracks.columns:
        no_pitch = int(tracks["pitch_x"].isna().sum())
    else:
        no_pitch = None

    with st.sidebar.expander("📊 Metrics", expanded=False):
        st.metric("Total detections",  f"{total:,}")
        st.metric("Avg per frame",     f"{avg:.1f}")
        st.metric("Max in any frame",  f"{max_frame}")

        if no_pitch is not None:
            pct = no_pitch / total * 100 if total else 0
            st.metric("Outside pitch boundary",
                      f"{no_pitch:,}",
                      delta=f"{pct:.1f}% of total",
                      delta_color="off")
        else:
            st.metric("Outside pitch boundary", "N/A — run apply_homography")

        if "class_name" in tracks.columns:
            by_class = tracks["class_name"].value_counts()
            st.caption("Class breakdown: " + "  ·  ".join(
                f"{cls} {cnt}" for cls, cnt in by_class.items()
            ))

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            secs = meta.get("processing_time_s")
            if secs is not None:
                mins, s = divmod(int(secs), 60)
                label = f"{mins}m {s:02d}s" if mins else f"{secs:.0f}s"
                st.metric("Processing time", label)
                if meta.get("model"):
                    st.caption(f"Model: {Path(meta['model']).name}  conf≥{meta.get('conf_threshold','?')}")
        else:
            st.metric("Processing time", "N/A — re-run process.py")


# ---------------------------------------------------------------------------
# Full viewer — shared by pages 1 and 2
# ---------------------------------------------------------------------------

def render_viewer(
    page_key: str,
    tracks_path: Path,
    clip_path: Path,
    pipeline_cmd: str = "",
) -> None:
    """
    Render the standard two-column viewer (video + pitch) with slider and stats.

    page_key    — unique string per page ("generic" / "football") used for session_state keys.
    tracks_path — parquet to load.
    clip_path   — source video for frame extraction.
    pipeline_cmd — shell command shown in the "not processed yet" error.
    """
    if not tracks_path.exists():
        st.warning(f"No tracks found at `{tracks_path}`.")
        if pipeline_cmd:
            st.code(pipeline_cmd, language="bash")
        return

    has_jpeg_frames = Path(f"assets/frames/{clip_path.stem}").exists()
    if not clip_path.exists() and not has_jpeg_frames:
        st.error(f"Clip not found: `{clip_path}`")
        return

    # --- Load data ---
    tracks   = load_tracks(str(tracks_path))

    # Derive meta JSON path: output/meta_generic_dev_clip3.json from tracks path
    meta_path = tracks_path.parent / tracks_path.name.replace("tracks_", "meta_").replace(".parquet", ".json")
    _add_metrics_sidebar(tracks, meta_path)

    hom_path = homography_path(clip_path)
    tim_path = timing_path(clip_path)

    if not hom_path.exists():
        st.error(
            f"Homography not calibrated for `{clip_path.name}`.  \n"
            f"Run: `python src/calibrate.py --video {clip_path}`"
        )
        return

    hom        = load_homography(str(hom_path))
    timing     = load_timing(str(tim_path)) if tim_path.exists() else {"kickoff_seconds": 0.0}
    teams      = load_teams(str(teams_path(clip_path)))
    colours    = colours_from_seeds(str(seeds_path(clip_path)))
    fps, _     = load_video_metadata(str(clip_path))
    meta_match = load_match_meta(clip_path.name)

    pitch_length = hom["pitch_length_m"]
    pitch_width  = hom["pitch_width_m"]

    source_fps  = fps
    output_fps  = 10.0
    frame_step  = max(1, round(source_fps / output_fps))

    min_frame = int(tracks["frame_id"].min())
    max_frame = int(tracks["frame_id"].max())

    # --- Sidebar: team filter ---
    frame_key    = f"{page_key}_frame"
    filter_key   = f"{page_key}_team_filter"
    has_team_col = "team" in tracks.columns

    away = meta_match.get("away_team", "Opposition")
    if has_team_col:
        filter_options = ["Show all", "ERFC only", f"{away} only", "Referee only"]
        team_filter = st.sidebar.selectbox(
            "Team filter", options=filter_options, index=0, key=filter_key,
        )
    else:
        team_filter = "Show all"

    debug_key = f"{page_key}_debug"
    show_debug = st.sidebar.checkbox("Show debug info", value=False, key=debug_key)

    # --- Session state: current frame ---
    if frame_key not in st.session_state:
        default_frame = min(600, max_frame)
        st.session_state[frame_key] = max(min_frame, default_frame)

    # --- Match time caption ---
    match_time_str = frame_to_match_time(st.session_state[frame_key], output_fps, timing)
    st.caption(f"{meta_match['match_label']}  ·  Match time: **{match_time_str}**")

    # --- Frame rows ---
    frame_rows = tracks[tracks["frame_id"] == st.session_state[frame_key]]
    if has_team_col and team_filter != "Show all":
        _team_map = {"ERFC only": 0, f"{away} only": 1, "Referee only": 2}
        display_rows = frame_rows[frame_rows["team"] == _team_map.get(team_filter, -99)]
    else:
        display_rows = frame_rows

    # --- Two-column layout ---
    col_video, col_pitch = st.columns(2)

    with col_video:
        source_idx = st.session_state[frame_key] * frame_step
        raw_frame  = fetch_frame(str(clip_path), source_idx)
        if raw_frame is not None:
            annotated = draw_detections(raw_frame, display_rows, colours)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.warning("Could not read frame from video.")

    with col_pitch:
        pitch_rows = display_rows.dropna(subset=["pitch_x", "pitch_y"])
        if show_debug and not pitch_rows.empty:
            st.caption(
                f"Frame {st.session_state[frame_key]}: {len(pitch_rows)} rows · "
                f"pitch_x {pitch_rows['pitch_x'].min():.1f}–{pitch_rows['pitch_x'].max():.1f} · "
                f"pitch_y {pitch_rows['pitch_y'].min():.1f}–{pitch_rows['pitch_y'].max():.1f}"
            )
        _erfc_dir = meta_match.get("erfc_attacking_direction", "right")
        _away     = meta_match.get("away_team", "Opposition")
        if _erfc_dir == "right":
            _left_ann  = "Enniskillen Rangers attacking →"
            _right_ann = f"← {_away} attacking"
        else:
            _left_ann  = f"← Enniskillen Rangers attacking"
            _right_ann = f"{_away} attacking →"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"color:#aaa;font-size:0.78rem;padding:2px 0 4px'>"
            f"<span>{_left_ann}</span><span>{_right_ann}</span></div>",
            unsafe_allow_html=True,
        )
        fig = build_pitch_figure(pitch_rows, colours, teams, pitch_length, pitch_width)
        st.plotly_chart(fig, use_container_width=True)

    # --- Aggregate heatmap ---
    if "pitch_x" in tracks.columns:
        _tm = {"ERFC only": 0, f"{away} only": 1, "Referee only": 2}
        if has_team_col and team_filter != "Show all":
            _all = tracks[tracks["team"] == _tm.get(team_filter, -99)].dropna(
                subset=["pitch_x", "pitch_y"]
            )
        else:
            _all = tracks.dropna(subset=["pitch_x", "pitch_y"])

        _hm_title = f"Density heatmap — {team_filter} ({len(_all):,} positions)"
        _fig_hm   = go.Figure()
        _fig_hm.add_shape(type="rect", x0=0, y0=0, x1=pitch_length, y1=pitch_width,
                          fillcolor="#3a7d44", line=dict(color="white", width=2), layer="below")
        _fig_hm.add_shape(type="line",
                          x0=pitch_length/2, y0=0, x1=pitch_length/2, y1=pitch_width,
                          line=dict(color="white", width=2))
        _r, _cx, _cy = 9.15, pitch_length/2, pitch_width/2
        _fig_hm.add_shape(type="circle",
                          x0=_cx-_r, y0=_cy-_r, x1=_cx+_r, y1=_cy+_r,
                          line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)")
        _pb_y0 = (pitch_width - 40.32) / 2
        for _x0, _x1 in [(0, 16.5), (pitch_length-16.5, pitch_length)]:
            _fig_hm.add_shape(type="rect", x0=_x0, y0=_pb_y0, x1=_x1, y1=_pb_y0+40.32,
                              line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)")

        if not _all.empty:
            _fig_hm.add_trace(go.Histogram2d(
                x=_all["pitch_x"], y=_all["pitch_y"],
                colorscale=[
                    [0.0,  "rgba(0,0,0,0)"],
                    [0.01, "rgba(255,255,180,0.5)"],
                    [0.3,  "rgba(255,160,0,0.7)"],
                    [1.0,  "rgba(200,0,0,0.85)"],
                ],
                xbins=dict(start=0, end=pitch_length, size=pitch_length / 30),
                ybins=dict(start=0, end=pitch_width,  size=pitch_width  / 20),
                showscale=False,
            ))
        _fig_hm.update_layout(
            title=dict(text=_hm_title, font=dict(color="white", size=13), x=0),
            margin=dict(l=0, r=0, t=36, b=0),
            xaxis=dict(range=[-2, pitch_length+2], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[pitch_width+2, -2], showgrid=False, zeroline=False, visible=False,
                       scaleanchor="x", scaleratio=1),
            plot_bgcolor="#3a7d44",
            paper_bgcolor="#1e1e1e",
            height=380,
        )
        st.plotly_chart(_fig_hm, use_container_width=True)

    # --- Slider ---
    play_key = f"{page_key}_playing"
    if play_key not in st.session_state:
        st.session_state[play_key] = False

    def _on_slider():
        st.session_state[frame_key] = st.session_state[f"_{page_key}_slider"]
        st.session_state[play_key]  = False

    st.slider("Frame", min_value=min_frame, max_value=max_frame,
              value=st.session_state[frame_key],
              key=f"_{page_key}_slider", on_change=_on_slider)

    # --- Prev / Play / Next ---
    btn_prev, btn_play, btn_next = st.columns([1, 2, 1])
    with btn_prev:
        if st.button("⏮ Prev", use_container_width=True, key=f"{page_key}_prev"):
            st.session_state[frame_key] = max(min_frame, st.session_state[frame_key] - 1)
            st.session_state[play_key]  = False
            st.rerun()
    with btn_play:
        _play_label = "⏸ Pause" if st.session_state[play_key] else "▶ Play"
        if st.button(_play_label, use_container_width=True, key=f"{page_key}_play"):
            if not st.session_state[play_key] and st.session_state[frame_key] >= max_frame:
                st.session_state[frame_key] = min_frame
            st.session_state[play_key] = not st.session_state[play_key]
            st.rerun()
    with btn_next:
        if st.button("Next ⏭", use_container_width=True, key=f"{page_key}_next"):
            st.session_state[frame_key] = min(max_frame, st.session_state[frame_key] + 1)
            st.session_state[play_key]  = False
            st.rerun()

    # Playback tick
    if st.session_state[play_key]:
        if st.session_state[frame_key] < max_frame:
            import time as _time
            _time.sleep(0.1)
            st.session_state[frame_key] += 1
            st.rerun()
        else:
            st.session_state[play_key] = False

    # --- Stats metrics row ---
    if has_team_col:
        tc = frame_rows["team"].value_counts()
        team_a_name = teams.get("team_a", "Team A")
        team_b_name = teams.get("team_b", "Team B")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Detected", len(frame_rows))
        m2.metric(team_a_name, int(tc.get(0, 0)))
        m3.metric(team_b_name, int(tc.get(1, 0)))
        m4.metric("Referee",   int(tc.get(2, 0)))
        m5.metric("Unassigned", int((frame_rows["team"] == -1).sum()))
    else:
        avg_per_frame = len(tracks) / tracks["frame_id"].nunique()
        m1, m2 = st.columns(2)
        m1.metric("Detected this frame", len(frame_rows))
        m2.metric("Avg per frame (clip)", f"{avg_per_frame:.1f}")
