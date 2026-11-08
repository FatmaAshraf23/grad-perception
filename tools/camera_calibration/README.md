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

## The car's camera (Raspberry Pi Camera V2, IMX219) -- record on the Pi, pick on the laptop (2026-11-08)
No live window over SSH, so: RECORD while moving the board, then `camcal_pick.py` picks the pictures offline with
camcal_capture.py's own rules (whole board, held still >= 0.3 s, new view) + the SHARPEST frame of each still moment.
1. **Sensor mode first.** A calibration is only valid for the picture it was made from (or a pure shrink of it).
   Use the full-view binned mode **1640 x 1232** (4:3, whole sensor) -- the same mode the car will stream.
   Do NOT calibrate at 1920 x 1080 or 640 x 480: those modes CROP the sensor (a different field of view).
2. **Fixed exposure** so the board is sharp: `--shutter 8000 --gain 2` (8 ms; brighter room if it is dark). The V2 has
   fixed focus (sharp from ~1 m): keep the board 0.6-2 m away.
3. **Record** (~2 minutes; hold the board still ~1 s at each place, then move to the next):
   `mkdir -p ~/calib/frames && rpicam-vid -t 120000 --mode 1640:1232 --width 1640 --height 1232 --framerate 15 --codec mjpeg -q 95 --segment 1 -o ~/calib/frames/f_%05d.jpg --save-pts ~/calib/frames/pts.txt --shutter 8000 --gain 2 -n`
   (`--segment 1` = one JPEG per frame, no video compression between frames; `--save-pts` = each frame's sensor time.
   NOT yet run on our Pi: check the first frames + pts.txt before the full session.)
   Places: all over the picture INCLUDING the corners and edges, near (board fills ~half the width) and far, tilted
   20-40 deg left/right AND up/down. Aim for 25-40 picked pictures.
4. **Copy** to the laptop: `scp -r apex@<pi ip>:~/calib/frames .`
5. **Pick:** `python camcal_pick.py frames --out captures_pi` -> img_*.png + captures.csv + summary.txt (with ADVICE:
   which areas the corners never reached, too little tilt, too few pictures -- record again if it says so).
6. **Calibrate:** `python camcal_calibrate.py captures_pi --fix-k3 --square <across/10 m> --square-y <down/7 m> --tag _pi`
   -> calibration_1640x1232_pi.yaml + 820x616 / 410x308 copies (exact shrinks of the same view; 2026-11-08 change:
   halves / quarters for any camera, never a 16:9 size of a 4:3 picture).
Test of the picker (2026-11-08): a fake recording made from the 42 iPhone pictures above (half size, sensor noise, JPEG
q95, motion between poses, timing jitter, a dropped frame; `make_test_recording.py`) -> 41 picked (the 42nd, img_031,
is a near-duplicate of img_030: view distance 0.008), calibration fx 1500.0 +- 2.3 / fy 1503.5 +- 2.3 vs 1496.9 / 1501.2
expected (half of the result above), hold-out 0.36 px. A video file works too; a never-still recording picks nothing
and says why.
