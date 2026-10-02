"""Start the simulator and the 7 Hz scan throttle (RPLIDAR A1 rate).

Usage:  ros2 launch lidar_perception sim_perception.launch.py
        ros2 launch lidar_perception sim_perception.launch.py cones:=true
(Drive with teleop_twist_keyboard in a separate terminal.)

The track now uses walls instead of cones, so the cone nodes
(scan_cluster_node, cone_map_node) are OFF by default. cones:=true starts them
again, e.g. on the old grad_track map or later for opponent-car detection.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cones = LaunchConfiguration('cones')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('f1tenth_gym_ros'), 'launch', 'gym_bridge_launch.py')),
        launch_arguments={'open_foxglove': 'false'}.items(),
    )

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
        sim, throttle, scan_cluster, cone_map,
    ])
