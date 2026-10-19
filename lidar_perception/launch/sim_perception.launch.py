"""Start the simulator and its RPLIDAR A1 model (/scan_a1, 7 Hz).

Usage:  ros2 launch lidar_perception sim_perception.launch.py
        ros2 launch lidar_perception sim_perception.launch.py cones:=true
        ros2 launch lidar_perception sim_perception.launch.py config:=<sim yaml> car_map:=<map yaml>
        ros2 launch lidar_perception sim_perception.launch.py lidar_model:=instant deskew:=false   (old LiDAR, A/B)
(Drive with teleop_twist_keyboard in a separate terminal.)

lidar_model / lidar_rate / deskew (2026-10-19, task 3b) -- how the A1 is simulated:
  rotating (DEFAULT since 2026-10-19, together with deskew:=true = what the real car will run):
          a1_emulator -- each beam from the moment it is measured during the turn, with the
          rplidar_ros stamp and beam order (like the real sensor at speed: 0.86 m of motion per scan
          at 6 m/s).
  instant (the old way, for A/B): one INSTANTANEOUS simulator scan every 1/lidar_rate s (topic_tools
          throttle) -- all beams from one pose, no motion distortion.
  deskew:=true (default) puts scan_deskew between the LiDAR model (/scan_a1_raw) and /scan_a1, so everything
          that reads /scan_a1 (sim_scan_relay -> AMCL, obstacle_node) gets de-skewed scans; it needs the
          local EKF's odometry (/odom/ekf_local, from sim_localization) and passes scans through
          unchanged until it arrives. deskew_mode: full | restamp | off; deskew_time_order: reverse
          (rplidar_ros) | forward (LaserScan definition). With lidar_model:=instant it passes through.

The track now uses walls instead of cones, so the cone nodes
(scan_cluster_node, cone_map_node) are OFF by default. cones:=true starts them
again, e.g. on the old grad_track map or later for opponent-car detection.

car_map (2026-10-07, obstacle detection) -- the WORLD and the CAR'S MAP are two things:
  The simulator draws its LiDAR scans from the map in its config (map_path) = the WORLD.
  By default the simulator's map_server also publishes that same file on /map, so the car
  knows the world exactly. Fine for localization tests, useless for obstacle detection:
  an obstacle drawn into the world would also be in the car's map and never look "new".
  car_map:=<map yaml> separates them, like on the real car (the slam_toolbox map is made
  of the EMPTY track; cones are put there later):
    - the simulator's own map_server publishes the WORLD on /map_world (checks/viewing only)
    - a second map_server (car_map_server) publishes car_map on /map
  AMCL, the EKF's scan check and the obstacle detector read /map, so they only know the
  walls. Example (closed1 with 3 cones in the world, made by make_cone_world.py):
    config:=$HOME/apex_maps/closed1_cones/sim_closed1_cones.yaml car_map:=$HOME/apex_maps/closed1/closed1.yaml
  Default '' = the old behaviour (one map for both).

sensor_frames (2026-10-14, fusion prep) -- where the sensors sit, as static TFs (apex_description sensor_frames,
apex_description/config/sensor_mounts_sim.yaml), on BOTH cars: ego_racecar/ (the simulator's true car) and est/
(the estimated car):   <car>/base_link -> <car>/camera_link -> <car>/camera_optical_frame
The LiDAR frames are not published here: the simulator (ego_racecar/laser) and sim_scan_relay (est/laser) already
do. Check: ros2 run apex_description check_frames --mounts sim --frame-prefix est/   sensor_frames:=false = off.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            LogInfo, OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


def _simulator(context):
    """The simulator -- alone (default), or with its map moved to /map_world and the car's map on /map."""
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('f1tenth_gym_ros'), 'launch', 'gym_bridge_launch.py')),
        launch_arguments={'open_foxglove': 'false'}.items(),
    )
    car_map = LaunchConfiguration('car_map').perform(context).strip()
    if not car_map:
        return [sim]
    path = os.path.abspath(os.path.expanduser(car_map))
    if not os.path.isfile(path):
        raise RuntimeError(f'car_map:={car_map} -- no such file')
    return [
        # 'map_server:' limits the rule to the node called map_server (the simulator's map server);
        # nothing else in the simulator is renamed
        GroupAction([SetRemap(src='map_server:/map', dst='/map_world'), sim]),
        Node(package='nav2_map_server', executable='map_server', name='car_map_server', output='screen',
             parameters=[{'yaml_filename': path, 'topic_name': 'map', 'frame_id': 'map',
                          'use_sim_time': False}]),
        # map_server is a lifecycle node: our activator switches it on (and retries if a reply is lost)
        Node(package='state_estimation', executable='lifecycle_activator', name='car_map_activator',
             output='screen', parameters=[{'node_names': ['car_map_server']}]),
        LogInfo(msg=f'car_map: simulator world -> /map_world, car map {path} -> /map'),
    ]


def _lidar(context):
    """The A1 model on /scan_a1: instantaneous (throttle) or rotating (a1_emulator), optionally de-skewed."""
    arg = lambda n: LaunchConfiguration(n).perform(context).strip()   # noqa: E731
    model = arg('lidar_model').lower()
    rate = float(arg('lidar_rate'))
    deskew = arg('deskew').lower() in ('1', 'true', 'yes', 'on')
    out = '/scan_a1_raw' if deskew else '/scan_a1'
    if model == 'instant':
        # Simulator publishes /scan at 250 Hz; the RPLIDAR A1 gives ~7 Hz
        nodes = [Node(package='topic_tools', executable='throttle', name='scan_throttle',
                      arguments=['messages', '/scan', f'{rate}', out])]
    elif model == 'rotating':
        nodes = [Node(package='lidar_perception', executable='a1_emulator', name='a1_emulator', output='screen',
                      parameters=[{'in_topic': '/scan', 'out_topic': out, 'rate_hz': rate}])]
    else:
        raise RuntimeError(f'lidar_model:={model} -- use instant or rotating')
    if deskew:
        nodes.append(Node(package='lidar_perception', executable='scan_deskew', name='scan_deskew', output='screen',
                          parameters=[{'in_topic': '/scan_a1_raw', 'out_topic': '/scan_a1',
                                       'odom_topic': '/odom/ekf_local', 'mode': arg('deskew_mode'),
                                       'time_order': arg('deskew_time_order'),
                                       'laser_x': 0.275, 'laser_y': 0.0, 'laser_yaw': 0.0}]))
    chain = f'LiDAR model: {model} {rate:g} Hz'
    if deskew:
        chain += f' -> scan_deskew ({arg("deskew_mode")})'
    nodes.append(LogInfo(msg=chain + ' -> /scan_a1'))
    return nodes


def generate_launch_description():
    cones = LaunchConfiguration('cones')

    scan_cluster = Node(
        package='lidar_perception', executable='scan_cluster_node',
        name='scan_cluster_node', output='screen',
        parameters=[{'scan_topic': '/scan_a1'}],
        condition=IfCondition(cones),
    )

    cone_map = Node(
        package='lidar_perception', executable='cone_map_node',
        name='cone_map_node', output='screen',
        condition=IfCondition(cones),
    )

    # where the sensors sit (static TFs) on the true car (ego_racecar/) and on the estimated car (est/) -- 2026-10-14
    sensor_frames = Node(
        package='apex_description', executable='sensor_frames', name='sensor_frames', output='screen',
        parameters=[{'mounts_file': 'sim', 'prefixes': 'ego_racecar/,est/', 'publish_laser': False}],
        condition=IfCondition(LaunchConfiguration('sensor_frames')),
    )

    return LaunchDescription([
        DeclareLaunchArgument('cones', default_value='false'),
        DeclareLaunchArgument('car_map', default_value='',
                              description="Map the car's software gets on /map ('' = the simulator's own map)"),
        DeclareLaunchArgument('sensor_frames', default_value='true',
                              description='static TFs of the sensors (apex_description) on ego_racecar/ and est/'),
        # A1 model (2026-10-19, docstring): instant | rotating; turns per second; de-skew on / off
        DeclareLaunchArgument('lidar_model', default_value='rotating'),
        DeclareLaunchArgument('lidar_rate', default_value='7.0'),
        DeclareLaunchArgument('deskew', default_value='true'),
        DeclareLaunchArgument('deskew_mode', default_value='full'),
        DeclareLaunchArgument('deskew_time_order', default_value='reverse'),
        OpaqueFunction(function=_simulator),
        OpaqueFunction(function=_lidar),
        scan_cluster, cone_map, sensor_frames,
    ])
