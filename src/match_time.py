"""
Match timing helpers.

frame_to_match_time is intentionally pure (no I/O, no state) so it can be
called per-frame in the Streamlit slider callback without overhead.
"""

import json
from pathlib import Path


def load_timing(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def frame_to_match_time(frame_idx: int, fps: float, timing: dict) -> str:
    """
    Convert a video frame index to a display string for match time.

    Returns one of:
      "Pre-match"        — before kickoff_seconds
      "MM:SS"            — normal play (first or second half)
      "45+SS" style      — stoppage time in first half (if it runs long)
      "HT"               — between first_half_end and second_half_start
      "Post-match"       — after full_time_seconds

    Rules when optional fields are null:
      - first_half_end / second_half_start null → treat the whole clip as
        one continuous half from kickoff onward
      - full_time_seconds null → never show "Post-match"
    """
    video_s = frame_idx / fps

    kickoff   = timing.get("kickoff_seconds")
    ht_end    = timing.get("first_half_end_seconds")
    sh_start  = timing.get("second_half_start_seconds")
    full_time = timing.get("full_time_seconds")

    # Pre-match
    if kickoff is not None and video_s < kickoff:
        return "Pre-match"

    # Post-match
    if full_time is not None and video_s > full_time:
        return "Post-match"

    # Half-time interval
    if ht_end is not None and sh_start is not None:
        if ht_end <= video_s <= sh_start:
            return "HT"

    # Second half: match clock continues from end-of-first-half
    if sh_start is not None and ht_end is not None and video_s > sh_start:
        first_half_duration = ht_end - (kickoff or 0.0)
        elapsed = first_half_duration + (video_s - sh_start)
        return _format_mm_ss(elapsed)

    # First half (or single-half clip)
    elapsed = video_s - (kickoff or 0.0)
    return _format_mm_ss(elapsed)


def _format_mm_ss(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"
