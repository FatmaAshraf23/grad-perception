# Raspberry Pi 5: ROS 2 Humble in Docker

The car's Pi 5 runs Raspberry Pi OS (Debian 13), which is not a Humble platform, so Humble runs in a container (Ubuntu 22.04, Python 3.10, the same as the laptop). Tested on 2026-11-06: workspace build and `car.launch.py localization:=false` with the real RPLIDAR A1M8 (`/scan_a1` about 7.6 Hz).

## One-time setup on the Pi
1. Docker: `sudo apt install -y docker.io`, `sudo usermod -aG docker $USER`, log out and back in.
2. Image (from this folder): `docker build -t apex-humble --build-arg UID=$(id -u) .`
3. Scripts: `cp run_ros.sh ros_shell.sh ~/ && chmod +x ~/run_ros.sh ~/ros_shell.sh`
4. Workspace: this repo is the workspace `src/`: `git clone <repo> ~/ros2_ws/src`
5. LiDAR driver (not in this repo; the car launch refuses the apt driver), inside the container:
   `cd ~/ros2_ws/src && git clone https://github.com/Slamtec/rplidar_ros.git && git -C rplidar_ros checkout 24cc9b6`
6. Build, inside the container: `cd ~/ros2_ws && colcon build --symlink-install --packages-up-to apex_bringup rplidar_ros`

## Daily use
- `~/run_ros.sh` starts the container (`--net=host`, workspace mounted from `~/ros2_ws`, `/dev/ttyUSB0` if present).
- `~/ros_shell.sh` opens a second shell in it.
- `ros2 launch apex_bringup car.launch.py localization:=false` runs the LiDAR stage.

## Notes
- Plug the LiDAR in BEFORE `run_ros.sh`; devices are fixed when the container starts.
- After building a NEW package, run `source install/setup.bash` in shells that were already open.
- New dependency in a `package.xml`? Inside the container: `rosdep update && rosdep install --from-paths src --ignore-src -r -s --rosdistro humble` prints the apt packages; add them to the Dockerfile (before `USER dev`) and rebuild the image.
- Power: with a 5 V / 3 A supply the Pi 5 limits all USB devices to about 600 mA in total.
