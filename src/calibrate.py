"""
Usage: python src/calibrate.py --video clips/dev_clip4.mp4 [--frame 600]

Interactive homography calibration using 6 spread points.
Opens a video frame, collects 6 mouse clicks at known pitch landmarks,
then computes and saves the homography matrix.

Output path is derived from the clip name:
  calibration/homography_{clip_stem}.json   e.g. homography_dev_clip4.json

Six points spread across the pitch distribute reprojection error evenly.
With >4 points OpenCV solves a least-squares problem — more robust than
a 4-point exact fit that's only accurate near the calibration corners.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

CALIBRATION_DIR = Path("calibration")

PENALTY_BOX_DEPTH = 16.5    # metres from goal line to the 18-yard line
PENALTY_BOX_WIDTH = 40.32   # metres, full width of the penalty area

N_POINTS = 6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, default=Path("clips/dev_clip2.mp4"))
    p.add_argument("--frame", type=int, default=0,
                   help="Source video frame index to use for calibration")
    return p.parse_args()


def load_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def build_default_pitch_points(pitch_length: float, pitch_width: float) -> list[tuple[float, float]]:
    """
    Returns the 6 default real-world pitch coordinates (metres) in click order.

    Pitch coordinate system:
      origin (0, 0) = top-left corner of pitch
      x increases along the LENGTH (left goal → right goal)
      y increases across the WIDTH (far touchline → near touchline)

    Far touchline = y=0 (top of pitch in aerial view)
    Near touchline = y=pitch_width (bottom of pitch, closest to camera)
    """
    y0 = (pitch_width - PENALTY_BOX_WIDTH) / 2      # far edge of both penalty boxes
    y1 = (pitch_width + PENALTY_BOX_WIDTH) / 2      # near edge of both penalty boxes
    half = pitch_length / 2

    return [
        (PENALTY_BOX_DEPTH,          y0),    # 1: Left box — far side, 16.5m line
        (PENALTY_BOX_DEPTH,          y1),    # 2: Left box — near side, 16.5m line
        (half,                       0.0),   # 3: Halfway line × far touchline
        (half,               pitch_width),   # 4: Halfway line × near touchline
        (pitch_length - PENALTY_BOX_DEPTH, y0),  # 5: Right box — far side, 16.5m line
        (pitch_length - PENALTY_BOX_DEPTH, y1),  # 6: Right box — near side, 16.5m line
    ]


# Labels printed in the terminal alongside each click prompt
POINT_LABELS = [
    "Left  penalty box — FAR touchline top corner  (16.5m line, far side)",
    "Left  penalty box — NEAR touchline top corner (16.5m line, near side)",
    "Halfway line — FAR  touchline intersection",
    "Halfway line — NEAR touchline intersection",
    "Right penalty box — FAR touchline top corner  (16.5m line, far side)",
    "Right penalty box — NEAR touchline top corner (16.5m line, near side)",
]


class ClickCollector:
    def __init__(self, frame: np.ndarray, n: int = N_POINTS):
        self.display = frame.copy()
        self.points: list[tuple[int, int]] = []
        self.n = n

    def callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.points) >= self.n:
            return
        idx = len(self.points)
        self.points.append((x, y))

        # Numbered dot
        cv2.circle(self.display, (x, y), 9, (0, 255, 255), -1)
        cv2.circle(self.display, (x, y), 10, (0, 0, 0), 2)
        cv2.putText(self.display, str(idx + 1), (x + 12, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(self.display, str(idx + 1), (x + 12, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw grouped lines: left box (1-2), halfway (3-4), right box (5-6)
        group_pairs = [(0, 1), (2, 3), (4, 5)]
        for a, b in group_pairs:
            if idx == b and a < len(self.points):
                cv2.line(self.display, self.points[a], (x, y), (0, 255, 255), 2)

        cv2.imshow("Calibration — 6 points", self.display)
        print(f"  Point {idx + 1}: pixel ({x}, {y})  —  {POINT_LABELS[idx]}")

        if len(self.points) == self.n:
            print(f"\nAll {self.n} points collected. Press any key to continue.")


def collect_clicks(frame: np.ndarray) -> list[tuple[int, int]]:
    collector = ClickCollector(frame)
    win = "Calibration — 6 points"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 900)
    cv2.imshow(win, collector.display)
    cv2.setMouseCallback(win, collector.callback)

    print("\n" + "=" * 60)
    print("CLICK 6 PITCH LANDMARKS IN ORDER:")
    print("=" * 60)
    for i, label in enumerate(POINT_LABELS, 1):
        print(f"  {i}. {label}")
    print()
    print("'Far touchline'  = the touchline FURTHEST from camera (top of image)")
    print("'Near touchline' = the touchline CLOSEST  to camera (bottom of image)")
    print("'Top corner'     = the corner on the 16.5m line (not the goal line)")
    print()
    print("If a landmark is obscured, click the nearest visible alternative")
    print("(centre spot, corner flag, etc.) — you can correct the pitch")
    print("coordinate after all 6 clicks.")
    print()
    print("Press Q or Esc to abort.\n")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            cv2.destroyAllWindows()
            sys.exit("Calibration aborted.")
        if key != 255 and len(collector.points) == N_POINTS:
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            sys.exit("Window closed — calibration aborted.")

    cv2.destroyAllWindows()
    return collector.points


def main():
    args = parse_args()

    if not args.video.exists():
        sys.exit(f"Video not found: {args.video}")

    CALIBRATION_DIR.mkdir(exist_ok=True)

    clip_stem   = args.video.stem
    output_path = CALIBRATION_DIR / f"homography_{clip_stem}.json"

    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if "homography_matrix" in existing:
            ans = input(f"{output_path.name} already has a matrix. Overwrite? [y/N] ").strip().lower()
            if ans != "y":
                sys.exit("Aborted.")

    frame = load_frame(args.video, args.frame)
    h_px, w_px = frame.shape[:2]
    print(f"Loaded frame {args.frame} from {args.video.name}  ({w_px}×{h_px})")

    # --- Pitch dimensions ---
    print("\nPitch dimensions (Enter to accept defaults):")
    try:
        pl = input("  Pitch length in metres [105.0]: ").strip()
        pw = input("  Pitch width  in metres [68.0]:  ").strip()
    except EOFError:
        pl, pw = "", ""
    pitch_length = float(pl) if pl else 105.0
    pitch_width  = float(pw) if pw else 68.0

    # --- Collect clicks ---
    pixel_points = collect_clicks(frame)

    # --- Default pitch coordinates ---
    default_pitch_pts = build_default_pitch_points(pitch_length, pitch_width)

    # --- Let user correct any pitch coordinate (for substituted landmarks) ---
    print("\nVerify/correct pitch coordinates for each clicked point.")
    print("Press Enter to accept the default. Enter new values if you clicked")
    print("a substitute landmark (e.g. centre spot instead of penalty box corner).\n")
    pitch_points = []
    for i, (def_x, def_y) in enumerate(default_pitch_pts):
        px, py = pixel_points[i]
        print(f"  Point {i+1}: pixel ({px}, {py})")
        print(f"    Default pitch coord: ({def_x:.2f}, {def_y:.2f}) m")
        try:
            rx = input(f"    pitch_x [{def_x:.2f}]: ").strip()
            ry = input(f"    pitch_y [{def_y:.2f}]: ").strip()
        except EOFError:
            rx, ry = "", ""
        pitch_points.append((
            float(rx) if rx else def_x,
            float(ry) if ry else def_y,
        ))
        print()

    pixel_pts_arr = np.array(pixel_points,  dtype=np.float32)
    pitch_pts_arr = np.array(pitch_points, dtype=np.float32)

    # Least-squares homography (method=0) with >4 points distributes error evenly
    H, _ = cv2.findHomography(pixel_pts_arr, pitch_pts_arr, method=0)
    if H is None:
        sys.exit("findHomography failed — points may be collinear.")

    # --- Reprojection check ---
    print("Reprojection check (pixel → pitch metres):")
    residuals = []
    for i, (px, py) in enumerate(pixel_points):
        src = np.array([[[float(px), float(py)]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H)[0][0]
        exp_x, exp_y = pitch_points[i]
        err = float(np.hypot(dst[0] - exp_x, dst[1] - exp_y))
        residuals.append(err)
        print(f"  Point {i+1}: ({px},{py}) → ({dst[0]:.2f}, {dst[1]:.2f})m"
              f"  [expected ({exp_x:.2f}, {exp_y:.2f})m]  err={err:.2f}m")

    rms = float(np.sqrt(np.mean(np.array(residuals) ** 2)))
    print(f"\nRMS reprojection error: {rms:.3f}m")
    if rms > 3.0:
        print("WARNING: RMS > 3m — consider re-clicking with more care.")
    elif rms > 1.5:
        print("NOTE: RMS between 1.5-3m — acceptable but re-clicking may improve accuracy.")
    else:
        print("Good calibration.")

    out = {
        "video_path":        str(args.video),
        "calibration_frame": args.frame,
        "pitch_length_m":    pitch_length,
        "pitch_width_m":     pitch_width,
        "n_calibration_points": N_POINTS,
        "pixel_points":      [list(map(int, p)) for p in pixel_points],
        "pitch_points_m":    [list(p) for p in pitch_points],
        "point_labels":      POINT_LABELS,
        "homography_matrix": H.tolist(),
        "rms_reprojection_error_m": round(rms, 4),
        "notes": (
            "Homography maps pixel (x,y) → pitch (x_m, y_m). "
            "Apply to foot position: bottom-centre of bbox. "
            "6-point least-squares fit for even error distribution."
        ),
    }
    output_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {output_path}")
    print(f"\nNext step:")
    print(f"  python src/apply_homography.py --tracks output/tracks_generic_{clip_stem}.parquet")


if __name__ == "__main__":
    main()
