"""Start the simulator and the 7 Hz scan throttle (RPLIDAR A1 rate).

Usage:  ros2 launch lidar_perception sim_perception.launch.py
        ros2 launch lidar_perception sim_perception.launch.py cones:=true
        ros2 launch lidar_perception sim_perception.launch.py config:=<sim yaml> car_map:=<map yaml>
(Drive with teleop_twist_keyboard in a separate terminal.)

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


def generate_launch_description():
    cones = LaunchConfiguration('cones')

    # Simulator publishes /scan at 250 Hz; the RPLIDAR A1 gives ~7 Hz
    throttle = Node(
        package='topic_tools', executable='throttle', name='scan_throttle',
        arguments=['messages', '/scan', '7.0', '/scan_a1'],
    )

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

    return LaunchDescription([
        DeclareLaunchArgument('cones', default_value='false'),
        DeclareLaunchArgument('car_map', default_value='',
                              description="Map the car's software gets on /map ('' = the simulator's own map)"),
        OpaqueFunction(function=_simulator),
        throttle, scan_cluster, cone_map,
    ])
