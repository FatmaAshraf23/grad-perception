# Camera calibration tools

Intrinsic calibration (focal length, optical centre, lens distortion) of a camera, and the result for the
iPhone development camera (Iriun). Run on **Windows** Python (OpenCV), where the camera is.

## Steps
1. **Board:** `python make_checkerboard.py` -> A4 PDF, 9 x 6 inner corners, 25 mm squares. Print at 100 %.
   **Measure it:** 10 squares across and 7 down with a ruler (printers stretch: ours was 250 x 176 mm, not 250 x 175).
   Glue it flat on cardboard if you can.
2. **Find the camera:** `python camcal_probe3.py` (set `CAM` env var for the index) -- pictures/s and a snapshot per
   backend/format. Iriun via DirectShow needs **MJPG** (otherwise black frames at 1/s).
3. **Capture:** `python camcal_capture.py --source cam:1:dshow:mjpg --out captures_X`
   Live window; saves automatically when the whole board is visible, held still, and the view is new.
   Aim for 25-40 pictures: all image areas (grid turns green), near + far, tilted 20-40 deg. `q` finish, `u` undo.
4. **Calibrate:** `python camcal_calibrate.py captures_X --fix-k3 --square 0.025 --square-y <down/7 in m> --tag _k3fixed_sq`
   -> calibration_<WxH>.yaml (ROS camera_info format), calibration_report.txt (RMS, hold-out test, std devs),
   per_image.csv, undistort_check.jpg.
   Good: RMS and hold-out TEST error close together, < ~1 px; fx/fy ~ 1.000 for square pixels.
5. **The size must match the stream.** A calibration converts exactly to a resized picture (scale fx, cx by the x
   factor; fy, cy by the y factor; D unchanged) and to a crop (subtract the crop offset from cx, cy). `camcal_server_check.py`
   shows what the Flask camera_server really delivers before its resize.

## Result: iPhone 14 Pro via Iriun (2026-10-13, 41 pictures)
`results/` -- use the `_k3fixed_sq` files. 2560 x 1440: fx 2993.9, fy 3002.4, cx 1342.2, cy 706.8, RMS 0.72 px,
hold-out 0.75 px. The ROS stream (camera_server.py squashes 2560x1440 -> 640x480) uses
`camera_publisher/config/iphone_iriun_640x480.yaml` (fx 748.5, fy 1000.8, cx 335.2, cy 235.3).
Known limits: board never reached the image edges (65 % coverage); error grows with tilt (paper not flat?).
**The car's real camera needs its own calibration with these tools.**
