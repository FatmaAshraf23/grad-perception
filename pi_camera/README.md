# pi_camera — the car's camera + cone detector on the Pi host

The Raspberry Pi Camera V2 (IMX219, CSI) works only through Raspberry Pi's libcamera / picamera2, which run on
Raspberry Pi OS — not in our Ubuntu 22.04 ROS 2 container. So (decision 2026-11-08):

```
Pi host (Raspberry Pi OS)                                  ROS container (apex-humble, --net=host)
  cam_detect.py: picamera2 -> YOLO (ONNX, CPU) --UDP 5600-->  camera_publisher detection_bridge
  stamp = the moment the picture was TAKEN                    -> /camera/detections (CameraDetectionArray)
                                                              -> /camera/camera_info (the calibration)
                                                              -> sensor_fusion fusion_node -> /obstacles/fused
```

Only the boxes cross over (a 640 × 480 picture is ~0.9 MB; boxes are ~200 bytes). If YOLO is slower than the camera,
old frames are skipped, never queued.

## Files
| File | What |
|---|---|
| `cam_detect.py` | the loop: source → detector → UDP; stats every 5 s |
| `sources.py` | `picamera2` (the car), `folder`, `video` (laptop tests); each frame stamped in CLOCK_REALTIME ns |
| `detectors.py` | `onnx:<file>:<imgsz>` (CPU), `ultralytics:<file>` (laptop reference), `none` (timing tests) |
| `../camera_publisher/camera_publisher/detection_packet.py` | the UDP packet (shared by both sides) |
| `test_pi_camera.py` | 21 checks without a Pi or ROS |
| `compare_onnx.py` | ONNX vs ultralytics vs the labels on real pictures |

## On the car (steps; not yet run on our Pi)
1. **Model, on the laptop:** `yolo export model=best.pt format=onnx imgsz=320 opset=12` → `best.onnx` (rename
   `best_320.onnx`). Copy it to the Pi: `scp best_320.onnx apex@<pi ip>:~/ros2_ws/src/pi_camera/`.
2. **Python on the Pi host** (picamera2 comes with Raspberry Pi OS):
   `python3 -m venv --system-site-packages ~/camenv && ~/camenv/bin/pip install onnxruntime opencv-python-headless`
   (`--system-site-packages` so the venv still sees picamera2.) Check: `~/camenv/bin/python -c "import picamera2, onnxruntime"`.
3. **Calibrate the camera first** (camera_calibration/README, Pi section) → `calibration_1640x1232_pi.yaml`, and copy
   it into the workspace on the Pi.
4. **Start the bridge in the container:**
   `ros2 run camera_publisher detection_bridge --ros-args -p camera_info_file:=<path>/calibration_1640x1232_pi.yaml`
5. **Start the camera on the host:**
   `~/camenv/bin/python ~/ros2_ws/src/pi_camera/cam_detect.py --source picamera2:820x616:30 --detector onnx:$HOME/ros2_ws/src/pi_camera/best_320.onnx:320 --exposure-us 4000 --gain 4`
   It streams the full sensor view (mode 1640 × 1232, the calibrated framing) shrunk to 820 × 616.
6. **Check:** the bridge's 5-s line (pictures/s, picture → published latency, dropped / lost); `ros2 topic hz /camera/detections`.
7. **Measure the delay:** `sensor_fusion latency_probe` with the car's camera → `--stamp-offset-ms` (or the fusion's
   `camera_delay_s`).

Exposure: fixed (no auto exposure); 4 ms limits motion blur at speed; raise the gain (or the light) if pictures are dark.
Clocks: the host and the container share the kernel, so their CLOCK_REALTIME is the same; the bridge drops packets
older than 0.5 s and logs stamps in the future.
