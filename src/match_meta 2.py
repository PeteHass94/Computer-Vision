import json
from pathlib import Path

_CALIBRATION_DIR = Path("calibration")

_DEFAULTS: dict = {
    "home_team": "Enniskillen Rangers",
    "away_team": "Opposition",
    "starting_phase": "",
    "erfc_attacking_direction": "right",
    "opposition_attacking_direction": "left",
    "kit_colours": {"erfc": "red", "opposition": "blue", "referee": "yellow"},
}


def load_meta(clip_name: str) -> dict:
    """Return match metadata for clip_name (e.g. 'dev_clip3.mp4').

    Falls back to safe defaults when the JSON file does not exist.
    """
    stem = Path(clip_name).stem
    path = _CALIBRATION_DIR / f"match_meta_{stem}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {**_DEFAULTS, "clip_name": clip_name, "match_label": stem}
