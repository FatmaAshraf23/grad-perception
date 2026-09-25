"""Start the simulator, the 7 Hz scan throttle and the LiDAR perception nodes.

Usage:  ros2 launch lidar_perception sim_perception.launch.py
(Drive with teleop_twist_keyboard in a separate terminal.)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
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
    )

    cone_map = Node(
        package='lidar_perception', executable='cone_map_node',
        name='cone_map_node', output='screen',
    )

    return LaunchDescription([sim, throttle, scan_cluster, cone_map])
