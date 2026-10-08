import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = PROJECT_ROOT / 'media' / 'videos' / 'test' / 'test.mp4'
FRAME_PATH = PROJECT_ROOT / 'media' / 'images' / 'test' / 'debug_first_frame.png'

path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else VIDEO_PATH
print('Checking video:', path)
cap = cv2.VideoCapture(str(path))
print('cap.isOpened():', cap.isOpened())
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'FPS={fps}, frames={frames}, size={w}x{h}')
ret, frame = cap.read()
print('read ret:', ret)
if ret:
    print('frame dtype:', frame.dtype, 'shape:', frame.shape)
    cv2.imwrite(str(FRAME_PATH), frame)
else:
    print('Failed to read frame')
cap.release()