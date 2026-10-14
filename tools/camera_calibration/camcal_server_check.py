#!/usr/bin/env python3
"""Open the camera EXACTLY as camera_server.py does (index 0, default backend, set 640x480 @30) and report the frame
size the DRIVER delivers before the resize; save one raw frame + the 640x480 resized one. Read only."""
import time
import cv2
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)
print('opened', cam.isOpened(), 'backend', cam.getBackendName() if cam.isOpened() else '-', flush=True)
f, n, t0 = None, 0, time.time()
while time.time() - t0 < 4:
    ok, fr = cam.read()
    if ok:
        f, n = fr, n + 1
cam.release()
if f is None:
    print('NO FRAME (is camera_server.py running and holding the camera?)')
else:
    h, w = f.shape[:2]
    b, g, r = [f[:, :, k].mean() for k in range(3)]
    print(f'DRIVER DELIVERS {w} x {h}, {n / 4:.1f} frames/s, mean B/G/R {b:.0f}/{g:.0f}/{r:.0f}', flush=True)
    cv2.imwrite('probe/server_raw.jpg', f if w <= 1280 else cv2.resize(f, (w // 2, h // 2)))
    cv2.imwrite('probe/server_640x480.jpg', cv2.resize(f, (640, 480), interpolation=cv2.INTER_AREA))
