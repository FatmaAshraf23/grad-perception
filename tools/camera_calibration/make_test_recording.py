#!/usr/bin/env python3
"""make_test_recording.py -- a FAKE Pi recording from real calibration pictures, to test camcal_pick.py.
For each picture: the board held still for --hold s (sensor noise + JPEG), then --move s of motion to the next pose
(the picture shifted step by step with motion blur, so the corners move and the frames are not still).
Writes <out>/f_00000.jpg ... + pts.txt in rpicam's --save-pts format (with timing jitter and one dropped frame).
Usage: python3 make_test_recording.py <folder with img_*.png> <out> [--scale 0.5 --fps 15 --hold 0.8 --move 0.4]"""
import argparse
import glob
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('out')
ap.add_argument('--scale', type=float, default=0.5); ap.add_argument('--fps', type=float, default=15.0)
ap.add_argument('--hold', type=float, default=0.8); ap.add_argument('--move', type=float, default=0.4)
ap.add_argument('--seed', type=int, default=1)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
os.makedirs(a.out, exist_ok=True)
files = sorted(glob.glob(os.path.join(a.src, 'img_*.png')))
t, k, pts = 0.0, 0, []
def emit(img):
    global t, k
    noisy = np.clip(img.astype(np.float32) + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
    if k == 37:                                    # one dropped frame: a gap in the timestamps
        t += 1.0 / a.fps
    cv2.imwrite(os.path.join(a.out, f'f_{k:05d}.jpg'), noisy, [cv2.IMWRITE_JPEG_QUALITY, 95])
    pts.append(1000.0 * t + rng.normal(0, 1.5))
    t += 1.0 / a.fps; k += 1
for i, f in enumerate(files):
    img = cv2.imread(f)
    img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
    for _ in range(int(round(a.hold * a.fps))):
        emit(img)
    n = int(round(a.move * a.fps))
    for j in range(1, n + 1):                      # moving away: shift + motion blur
        d = 12.0 * j
        M = np.float32([[1, 0, d], [0, 1, 0.5 * d]])
        sh = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        kern = np.zeros((1, 15), np.float32); kern[0, :] = 1.0 / 15
        emit(cv2.filter2D(sh, -1, kern))
with open(os.path.join(a.out, 'pts.txt'), 'w') as f:
    f.write('# timecode format v2\n' + ''.join(f'{p:.3f}\n' for p in pts))
print(f'{k} frames from {len(files)} poses, {t:.1f} s, {a.fps} fps -> {a.out}')
