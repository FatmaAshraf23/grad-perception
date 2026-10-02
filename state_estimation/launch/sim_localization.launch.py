"""Localization on the wall map: fake sensors + local EKF + AMCL + global EKF.

Run the simulator first (ros2 launch lidar_perception sim_perception.launch.py),
then:  ros2 launch state_estimation sim_localization.launch.py
Speed-scale calibration: both EKFs start from the k saved in scale_file
(default ~/.ros/ekf_speed_scale.yaml; ekf_global rewrites it). To start
uncalibrated (k = 1):  scale_file:=none
Start pose (must match sx, sy, stheta in sim.yaml):
       ros2 launch state_estimation sim_localization.launch.py x0:=0.0 y0:=0.0 yaw0:=0.0

Data flow
  fake_vehicle_sensors ─ /vehicle/measured ─┐
  fake_imu ──────────── /imu/data ──────────┼─> ekf_local  (odom frame) ─ TF odom->est/base_link
                                            │                               │
  /scan_a1 ─> sim_scan_relay ─ /scan_amcl ──┼──────────────> AMCL <─────────┘  (+ /map)
                                            │                  │ /amcl_pose, TF map->odom
                                            └─> ekf_global (map frame) <┘ ─> /odom/ekf
                                                                                 │
  apex_track (Frenet) ─────────────────────────────────> state_estimate <────────┘
                                                         ─> /state_estimate (apex_msgs/StateEstimate)
StateEstimate node: on by default; state_estimate:=false leaves it out.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    x0 = LaunchConfiguration('x0')
    y0 = LaunchConfiguration('y0')
    yaw0 = LaunchConfiguration('yaw0')
    use_amcl = LaunchConfiguration('use_amcl')
    f = lambda s: ParameterValue(s, value_type=float)  # noqa: E731

    amcl_yaml = os.path.join(get_package_share_directory('state_estimation'),
                             'config', 'amcl_sim.yaml')

    def se(name, executable=None, params=None):
        return Node(package='state_estimation', executable=executable or name, name=name,
                    output='screen', parameters=params or [])

    return LaunchDescription([
        DeclareLaunchArgument('x0', default_value='0.0'),
        DeclareLaunchArgument('y0', default_value='0.0'),
        DeclareLaunchArgument('yaw0', default_value='0.0'),
        DeclareLaunchArgument('use_amcl', default_value='true'),
        DeclareLaunchArgument('amcl_min_sigma_pos', default_value='0.10'),
        DeclareLaunchArgument('scale_file', default_value='~/.ros/ekf_speed_scale.yaml'),
        DeclareLaunchArgument('state_estimate', default_value='true'),

        # Simulated car sensors
        se('fake_vehicle_sensors'),
        se('fake_imu'),
        # Old wheel-only odometry, kept for comparison (red path)
        se('wheel_odometry', params=[{'x0': f(x0), 'y0': f(y0), 'yaw0': f(yaw0)}]),

        # LOCAL EKF: odometry for AMCL, starts at the origin of the odom frame
        se('ekf_local', 'ekf_node', [{
            'mode': 'odom', 'frame_id': 'odom', 'child_frame_id': 'est/base_link',
            'publish_tf': True, 'use_amcl': False,
            'x0': 0.0, 'y0': 0.0, 'yaw0': 0.0,
            'odom_topic': '/odom/ekf_local', 'path_topic': '/odom/ekf_local_path',
            'diag_prefix': '/ekf_local',
            'scale_file': LaunchConfiguration('scale_file')}]),

        # Scan in the estimated car's frame (simulation only)
        se('sim_scan_relay'),

        # AMCL (a lifecycle node, started by its lifecycle manager)
        Node(package='nav2_amcl', executable='amcl', name='amcl', output='screen',
             parameters=[amcl_yaml, {'initial_pose.x': f(x0), 'initial_pose.y': f(y0),
                                     'initial_pose.yaw': f(yaw0)}]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_localization', output='screen',
             parameters=[{'autostart': True, 'node_names': ['amcl']}]),

        # GLOBAL EKF: the pose planning/MPC will use
        se('ekf_global', 'ekf_node', [{
            'mode': 'map', 'frame_id': 'map', 'child_frame_id': 'est/base_link',
            'publish_tf': False,
            'use_amcl': ParameterValue(use_amcl, value_type=bool),
            'amcl_min_sigma_pos': f(LaunchConfiguration('amcl_min_sigma_pos')),
            'x0': f(x0), 'y0': f(y0), 'yaw0': f(yaw0),
            'odom_topic': '/odom/ekf', 'path_topic': '/odom/ekf_path',
            'diag_prefix': '/ekf',
            'scale_file': LaunchConfiguration('scale_file')}]),

        # StateEstimate for control: /odom/ekf + apex_track -> /state_estimate (contract v0.2)
        Node(package='state_estimation', executable='state_estimate', name='state_estimate',
             output='screen', condition=IfCondition(LaunchConfiguration('state_estimate'))),
    ])
