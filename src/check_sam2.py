"""
SAM 2 smoke test — image predictor on a single frame.

Usage: python src/check_sam2.py
Hardcoded inputs: /tmp/sam2_test_frame.jpg, click at a Rangers player.
Output: /tmp/sam2_mask_overlay.png
"""

import numpy as np
import torch
from PIL import Image
import cv2

# --- Config ---
IMAGE_PATH  = "/tmp/sam2_test_frame.jpg"
OUTPUT_PATH = "/tmp/sam2_mask_overlay.png"
CLICK_XY    = (1253, 363)  # Largest Rangers red blob, frame 750, dev_clip3

# --- Device ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# SAM 2 image predictor (stateless — no video state needed)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint  = "models/sam2_hiera_tiny.pt"
config_name = "sam2_hiera_t"   # matches the tiny variant

model     = build_sam2(config_name, checkpoint, device=device)
predictor = SAM2ImagePredictor(model)

# Load image
image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
print(f"Image shape: {image.shape}")

predictor.set_image(image)

point_coords = np.array([CLICK_XY], dtype=np.float32)
point_labels = np.array([1], dtype=np.int32)   # 1 = foreground

masks, scores, _ = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=True,
)

best = int(np.argmax(scores))
print(f"Masks returned: {len(masks)}  best score: {scores[best]:.3f}")

# Overlay best mask in red on the original image
overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
mask    = masks[best].astype(np.uint8)
overlay[mask == 1] = (
    overlay[mask == 1] * 0.5 + np.array([0, 0, 200]) * 0.5
).astype(np.uint8)
cv2.circle(overlay, CLICK_XY, 8, (0, 255, 0), -1)
cv2.imwrite(OUTPUT_PATH, overlay)
print(f"Saved overlay → {OUTPUT_PATH}")
print(f"Mask pixels: {mask.sum()} / {mask.size}  ({100*mask.mean():.2f}%)")
