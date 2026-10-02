"""One evaluation run: localization + automatic test driver + CSV logger.

Run the simulator first (ros2 launch lidar_perception sim_perception.launch.py),
then:
    ros2 launch state_estimation sim_eval.launch.py run_name:=baseline laps:=3
The car waits 3 s, drives `laps` laps of the Levine loop, stops. The logger
writes ~/eval_logs/<run_name>_<date>.csv until you press Ctrl+C.
Then:  python3 plot_eval.py ~/eval_logs/<file>.csv --map <levine.yaml>

Arguments: run_name, laps, v_max, amcl_min_sigma_pos (EKF trust in AMCL, m,
default 0.10 since 2026-10-01), scale_file (speed-scale calibration;
scale_file:=none = start uncalibrated with k = 1, the old behaviour).
Do NOT run teleop at the same time (both would send drive commands).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('state_estimation')
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'sim_localization.launch.py')),
        launch_arguments={'amcl_min_sigma_pos': LaunchConfiguration('amcl_min_sigma_pos'),
                          'scale_file': LaunchConfiguration('scale_file')}.items(),
    )
    driver = Node(
        package='state_estimation', executable='test_driver', name='test_driver', output='screen',
        parameters=[{'laps': ParameterValue(LaunchConfiguration('laps'), value_type=int),
                     'v_max': ParameterValue(LaunchConfiguration('v_max'), value_type=float)}],
    )
    logger = Node(
        package='state_estimation', executable='eval_logger', name='eval_logger', output='screen',
        parameters=[{'run_name': LaunchConfiguration('run_name')}],
    )
    return LaunchDescription([
        DeclareLaunchArgument('run_name', default_value='run'),
        DeclareLaunchArgument('laps', default_value='3'),
        DeclareLaunchArgument('v_max', default_value='1.5'),
        DeclareLaunchArgument('amcl_min_sigma_pos', default_value='0.10'),
        DeclareLaunchArgument('scale_file', default_value='~/.ros/ekf_speed_scale.yaml'),
        localization, driver, logger,
    ])
