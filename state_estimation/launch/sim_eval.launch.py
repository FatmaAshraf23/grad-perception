"""One evaluation run: localization + automatic test driver + CSV logger.

Run the simulator first (ros2 launch lidar_perception sim_perception.launch.py),
then:
    ros2 launch state_estimation sim_eval.launch.py run_name:=baseline laps:=3
The car waits 3 s, drives `laps` laps of the Levine loop, stops. The logger
writes ~/eval_logs/<run_name>_<date>.csv until you press Ctrl+C.
Then:  python3 plot_eval.py ~/eval_logs/<file>.csv --map <levine.yaml>

Arguments: run_name, laps, v_max, amcl_min_sigma_pos (EKF trust in AMCL, m,
default 0.10 since 2026-10-01), numpy_threads (BLAS threads per node, default 1),
pipeline (separate = 3 processes, merged = ONE process for both EKFs + state_estimate),
sim_sensors (v1 | v3: simulated gyro + wheel-speed model, see sim_localization),
scale_file (speed-scale calibration;
scale_file:=none = start uncalibrated with k = 1, the old behaviour).
Do NOT run teleop at the same time (both would send drive commands).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('state_estimation')
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'sim_localization.launch.py')),
        launch_arguments={'amcl_min_sigma_pos': LaunchConfiguration('amcl_min_sigma_pos'),
                          'scale_file': LaunchConfiguration('scale_file'),
                          'numpy_threads': LaunchConfiguration('numpy_threads'),
                          'ekf_path_period': LaunchConfiguration('ekf_path_period'),
                          'ekf_truth_topic': LaunchConfiguration('ekf_truth_topic'),
                          'pipeline': LaunchConfiguration('pipeline'),
                          'sim_sensors': LaunchConfiguration('sim_sensors'),
                          'steer_calib_file': LaunchConfiguration('steer_calib_file')}.items(),
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
        DeclareLaunchArgument('numpy_threads', default_value='1'),
        DeclareLaunchArgument('ekf_path_period', default_value='0.0'),
        DeclareLaunchArgument('ekf_truth_topic', default_value='none'),
        DeclareLaunchArgument('pipeline', default_value='merged'),     # or separate (3 processes)
        DeclareLaunchArgument('sim_sensors', default_value='v3'),      # or v1 (old gyro + speed models)
        DeclareLaunchArgument('steer_calib_file', default_value='~/.ros/steering_calibration.yaml'),
        # CPU rule (also for test_driver and eval_logger, which start here)
        SetEnvironmentVariable('OPENBLAS_NUM_THREADS', LaunchConfiguration('numpy_threads')),
        SetEnvironmentVariable('OMP_NUM_THREADS', LaunchConfiguration('numpy_threads')),
        SetEnvironmentVariable('MKL_NUM_THREADS', LaunchConfiguration('numpy_threads')),
        localization, driver, logger,
    ])
