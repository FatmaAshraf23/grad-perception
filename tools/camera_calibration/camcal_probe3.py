#!/usr/bin/env python3
"""camcal_probe3.py -- why did job 099 get only 1 picture/s from the iPhone (DirectShow camera 1, 2560 x 1440)?
For each setting: how many pictures/s the CAMERA delivers (read loop only, no processing), how long our processing
takes per picture (grey + board search + window drawing, without a window), and whether the picture is sane
(brightness, green tint). Saves the last picture of each try to probe/. No window, read only."""
import os
import time

import cv2
import numpy as np

os.makedirs('probe', exist_ok=True)
IDX = int(os.environ.get('CAM', '1'))
tries = [('dshow_default', cv2.CAP_DSHOW, None, None), ('dshow_MJPG', cv2.CAP_DSHOW, None, 'MJPG'),
         ('dshow_1920', cv2.CAP_DSHOW, (1920, 1080), None), ('dshow_1920_MJPG', cv2.CAP_DSHOW, (1920, 1080), 'MJPG'),
         ('dshow_1280', cv2.CAP_DSHOW, (1280, 720), None), ('dshow_640', cv2.CAP_DSHOW, (640, 480), None),
         ('msmf_default', cv2.CAP_MSMF, None, None)]
for name, api, size, fcc in tries:
    t_open = time.time()
    cap = cv2.VideoCapture(IDX, api)
    if not cap.isOpened():
        print(f'{name}: not opened', flush=True)
        continue
    if fcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fcc))
    if size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    t_open = time.time() - t_open
    n, last, t0 = 0, None, time.time()
    while time.time() - t0 < 4.0:
        ok, f = cap.read()
        if ok and f is not None:
            n += 1
            last = f
    fps = n / (time.time() - t0)
    if last is None:
        print(f'{name}: opened in {t_open:.1f} s, NO picture in 4 s', flush=True)
        cap.release()
        continue
    h, w = last.shape[:2]
    four = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = ''.join(chr((four >> 8 * k) & 0xFF) for k in range(4)) if four > 0 else '?'
    b, g, r = [float(last[:, :, k].mean()) for k in range(3)]
    t1 = time.time()
    for _ in range(5):          # our per-picture work (as camcal_capture.py, without imshow)
        gray = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
        s = min(1.0, 960.0 / w)
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
        cv2.findChessboardCorners(small, (9, 6), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE |
                                  cv2.CALIB_CB_FAST_CHECK)
        over = last.copy()
        cv2.addWeighted(over, 0.18, last, 0.82, 0)
    proc_ms = (time.time() - t1) / 5 * 1000
    cv2.imwrite(f'probe/p3_cam{IDX}_{name}.jpg', cv2.resize(last, (640, int(640 * h / w))))
    print(f'{name}: {w} x {h} {fourcc}, CAMERA {fps:.1f} pictures/s, our processing {proc_ms:.0f} ms/picture, '
          f'mean B/G/R {b:.0f}/{g:.0f}/{r:.0f}{"  <-- GREEN?" if g > 1.6 * max(b, r, 1) else ""}', flush=True)
    cap.release()
    time.sleep(0.5)
