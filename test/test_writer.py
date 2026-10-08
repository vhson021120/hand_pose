import os
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / 'media' / 'videos' / 'test'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / 'writer_test.mp4'
w, h = 640, 480

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, 30, (w, h))

print("VideoWriter opened:", out.isOpened())

for i in range(60):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        f"Frame {i}",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    out.write(frame)

out.release()

print("Finished")
print("Exists:", OUTPUT_PATH.exists())

if OUTPUT_PATH.exists():
    print("Size:", os.path.getsize(OUTPUT_PATH))