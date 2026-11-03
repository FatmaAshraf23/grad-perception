# grad-perception — Perception & State Estimation for a 1:10 autonomous racing car

This repository is the **perception and state-estimation** part of our Cairo University graduation project (team name "APEX"):
a 1:10 scale autonomous RC racing car, similar to the F1TENTH platform. It runs on **ROS 2 Humble**.

The perception stack answers two questions for the rest of the team:

1. **Where is the car, and how is it moving?** It publishes `apex_msgs/StateEstimate` on `/state_estimate`: pose, velocities,
   yaw rate, track (Frenet) coordinates, a 1-sigma uncertainty for each value, and a health mode with flags.
2. **What is on the track?** It publishes `apex_msgs/ObstacleArray` on `/obstacles`: confirmed obstacles and cones in the
   map frame and in track coordinates.

Planning and MPC (other team members) read these two messages. The message formats are agreed with them in a written
**Perception → Control state contract**. The current version is v0.3, with v0.4 in progress.

```
 RPLIDAR A1 ──> scan_deskew ──> AMCL (map) ───────┐
 IMU ─────────────────────────────────────────────┤
 wheel speed + steering ──────────────────────────┼──> EKF (local + global) ──> state_estimate ──> /state_estimate ──> MPC / planning
                                                  │                                   │
 LiDAR scan + map ──> obstacle_node (map difference + tracking) <──────────────────────┘ ──> /obstacles ──> planning
 Camera ──> camera_node ──> yolo_node ──> /camera/detections ──> (camera–LiDAR fusion, in progress)
```

## Packages

| Package | What it does |
|---|---|
| [`apex_msgs`](apex_msgs/msg) | Shared message types: `StateEstimate`, `ObstacleArray` / `Obstacle`, `CameraDetectionArray` / `CameraDetection`. Each message carries a schema version, a sequence id and an observation timestamp. |
| [`apex_track`](apex_track) | Shared track geometry (pure Python, works with or without ROS): global pose ↔ Frenet `(s_abs, e_y, e_psi, kappa)`, track walls and widths, and a `track_hash` so every node is sure it uses the same track. See its [README](apex_track/README.md). |
| `state_estimation` | Wheel odometry, the local and global EKFs, the AMCL map filter, and the `state_estimate` node that builds the contract message. Also contains health checks: stale sensors, wheel slip, IMU dropout, map mismatch, estimator lag, and a standstill (ZUPT) gate. Includes simulator sensor models and evaluation tools. |
| `lidar_perception` | 2D LiDAR perception: a rotating RPLIDAR A1 emulator and scan de-skew, the obstacle list (map difference + tracking + scan alignment), and older clustering and cone-map nodes. |
| `camera_publisher` | Camera node (MJPEG stream from the development camera, an iPhone through Iriun) and a YOLO cone detector that publishes `/camera/detections`, stamped with the time each picture was taken. |
| `sensor_fusion` | Camera–LiDAR projection (LiDAR point ↔ pixel), camera–LiDAR extrinsic calibration with a checkerboard, a simulated "fake camera", and camera latency measurement. |
| `apex_description` | Where each sensor sits on the car: one YAML per platform (`sensor_mounts_car.yaml`, `sensor_mounts_sim.yaml`), published as static TFs and checked with `check_frames`. |
| `apex_bringup` | Starts the **real car**. Loads `config/car.yaml` and checks it first: the launch stops if a value is impossible or a simulator switch is left on. Values that are still placeholders print a warning. Also contains the calibration drives (wheel, steering, slip). |
| `tools/` | Scripts that run outside the ROS nodes: [camera calibration](tools/camera_calibration/README.md), [sensor bench tests](tools/bench/README.md) (LiDAR range offset, IMU noise and bias), and [fusion checks](tools/fusion/README.md). |
| `my_first_ros_package` | My first ROS 2 publisher from when I was learning ROS. Not part of the stack. |

## Hardware

- **LiDAR:** RPLIDAR A1M8 (2D, about 7–10 Hz)
- **IMU:** MPU-6050
- **Wheel speed and steering** from the car's embedded board
- **Camera (for development):** iPhone 14 Pro through Iriun, calibrated (see `tools/camera_calibration`). The car's
  real camera will need its own calibration.

## Build

```bash
# in a ROS 2 Humble workspace (Ubuntu 22.04 / WSL)
cd ~/ros2_ws/src
git clone https://github.com/FatmaAshraf23/grad-perception.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
```

`yolo_node` also needs `ultralytics` (`pip install ultralytics`) and a trained model file. You pass the model's path with the `model` parameter.

## Run

**Simulator** (needs [`f1tenth_gym_ros`](https://github.com/f1tenth/f1tenth_gym_ros)):

```bash
# 1. simulator + RPLIDAR A1 model + de-skew
ros2 launch lidar_perception sim_perception.launch.py
# 2. localization + state estimate (EKF, AMCL, /state_estimate)
ros2 launch state_estimation sim_localization.launch.py
# 3. obstacle list
ros2 launch lidar_perception obstacles.launch.py track:=levine range_offset:=0.022
```

`state_estimation sim_eval.launch.py` drives automatic laps and logs errors against the simulator's ground truth.
Use `run_name:=`, `laps:=` and `v_max:=` to set the run name, number of laps and top speed.

**Real car:**

```bash
ros2 launch apex_bringup car.launch.py                      # LiDAR, de-skew, sensor frames
ros2 launch apex_bringup car.launch.py localization:=true map:=<track map.yaml> track:=<name>
ros2 launch apex_bringup car_calib.launch.py tool:=wheel true_distance:=<m>   # calibration drives
```

**Camera:**

```bash
ros2 run camera_publisher camera_node
ros2 run camera_publisher yolo_node --ros-args -p model:=<best.pt> -p show:=false
```

## Tests

Most tests are plain Python and don't need ROS. Each one prints its result:

```bash
python3 state_estimation/test/test_health.py
python3 apex_track/test/test_frenet.py
python3 tools/bench/test_bench.py
```

You can also run them all with `colcon test` or `pytest`.

## Status

**Working in simulation:**
- Localization: EKF + AMCL with a de-skewed rotating LiDAR model
- The full `StateEstimate` message, including uncertainties and health flags
- Health checks tested with a fault suite: wheel slip, IMU dropout, missing wheel data, map mismatch
- LiDAR obstacle list
- YOLO cone detector publishing ROS messages
- Camera intrinsic calibration
- Camera–LiDAR projection and calibration tools

**In progress:**
- More honest uncertainty in corners after an IMU dropout
- State contract v0.4

**Next:**
- Bench-test the real sensors
- Measure the placeholder values listed in `car.yaml` under `not_measured`
- Camera–LiDAR calibration on the car
- Camera + LiDAR fusion node

## Author

Fatma Ashraf, Perception & State Estimation, Mechatronics Engineering, Cairo University.
