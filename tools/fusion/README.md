# tools/fusion -- checks for the camera side of camera-LiDAR fusion

Two scripts that test what the camera sends to the fusion node, before the real car exists.
Run them in WSL (Ubuntu-22.04, ROS 2 Humble) with the workspace sourced:

    source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash

Both read `apex_msgs/CameraDetectionArray` (`/camera/detections`): one message per camera picture, header = the
PICTURE's time stamp + the camera's optical frame, boxes in pixels (`u_min, v_min, u_max, v_max`). The fusion node
needs the picture's time (not the time the boxes were sent) to look up where the car was when the picture was taken.

## yolo_msg_test.py -- does the real YOLO node publish the right boxes?

Publishes the labelled test pictures of the cone dataset one at a time as `/camera/image_compressed`, receives
`camera_publisher yolo_node`'s `/camera/detections` and compares the boxes with the dataset's labels (IoU >= 0.5).
Also checks that each answer carries the picture's stamp, frame_id and size, and draws the first 6 pictures
(green = label, red = YOLO) into the output folder.

    # terminal 1: the YOLO node without its window (the model path is the parameter "model")
    ros2 run camera_publisher yolo_node --ros-args -p show:=false
    # terminal 2: <images dir> needs a sibling labels/ folder with the YOLO txt labels
    python3 ~/ros2_ws/src/tools/fusion/yolo_msg_test.py <dataset>/test/images /tmp/yolo_msg

2026-10-14 (conesV1 test set, 68 pictures, best.pt, RTX 3050): 68 of 68 answered with the right stamp, frame_id and
size; 103 cones found, 27 missed, 32 extra boxes (recall 79 %, precision 76 %), mean IoU 0.81.

## fake_camera_check.py -- score the simulator's fake camera (and preview the fusion)

The simulator has no camera. `sensor_fusion fake_camera` makes the boxes YOLO would give from the cone positions,
the car's pose at the picture time, the camera mount and the lens model (cones behind walls are not seen; misses,
box noise, false boxes and 80 ms latency are ASSUMPTIONS until we measure the real camera). This script scores a
recording of it:

1. timing: picture rate, sequence gaps, latency (receive time - picture stamp)
2. detection: cones the camera should see clearly (in view, line of sight, box >= 25 px tall) -> found or missed
3. hidden cones: cones behind a wall must have no box
4. FUSION PREVIEW: the LiDAR obstacles (`/obstacles`) projected into each picture with the ESTIMATED pose -> how many
   land inside a camera box, and the pixel offset (what the fusion node's association gate must allow)

Recording (simulator with a cone world; `tools/make_cone_world.py` makes one, e.g. `~/apex_maps/closed1_cones/`):

    ros2 launch lidar_perception sim_perception.launch.py config:=<cone world>/sim_closed1_cones.yaml car_map:=<car map>.yaml
    ros2 launch lidar_perception obstacles.launch.py track:=closed1 range_offset:=0.022    # the simulator LiDAR's offset
    ros2 launch sensor_fusion fake_camera.launch.py cones_file:=<cone world>/cones_truth.csv world_map:=<cone world>/closed1_cones.yaml
    ros2 bag record -o ~/obst_bags/fake_camera_test /camera/detections /camera/camera_info /obstacles /state_estimate /ego_racecar/odom
    # drive laps (e.g. state_estimation sim_eval.launch.py), then stop the recording with Ctrl+C

Analysis:

    python3 ~/ros2_ws/src/tools/fusion/fake_camera_check.py ~/obst_bags/fake_camera_test \
        --cones <cone world>/cones_truth.csv --world <cone world>/closed1_cones.yaml \
        --camera $(ros2 pkg prefix camera_publisher)/share/camera_publisher/config/iphone_iriun_640x480.yaml \
        --out /tmp/fake_camera_check

2026-10-14 (closed1 + 3 cones, 2 laps, 1122 pictures): 15.0 Hz, 0 gaps, latency median 82 ms; clearly visible cones
found in 93-100 % of the pictures (box centre error median 2.5-3.7 px); the cone behind a wall (43 pictures) never
boxed; 29 false boxes. Fusion preview: 78 % of the LiDAR obstacles in view land in a box, 87 % of the cone boxes
contain a LiDAR obstacle; offset median 2.6 px / p95 9.8 px horizontally, mostly the localization error (3.0 / 9.2 px
alone) -> the association gate must allow about 1 deg plus the heading uncertainty.
