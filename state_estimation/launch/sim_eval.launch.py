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
scale_file:=none = start uncalibrated with k = 1, the old behaviour),
track (apex_track track for the StateEstimate AND the logger's true Frenet state, default levine)
and waypoints (the test driver's path, '' = levine_centerline.csv) -- 2026-10-06, for other maps:
    track:=closed1 waypoints:=$(ros2 pkg prefix apex_track)/share/apex_track/tracks/closed1/centerline.csv
    (the simulator must run the same map: sim_perception.launch.py config:=<sim yaml with that map>)
ekf_test_fault (TEST ONLY, 2026-10-06): 't_s,dx,dy,dyaw_deg' moves ekf_global's pose once -> tests
the robust jump to AMCL; '' (default) = off.
Test driver speed profile (2026-10-18, racing-speed runs): a_lat (sideways acceleration allowed in
corners, m/s^2, default 1.5), v_min (default 0.7), a_brake / a_accel (m/s^2; default 0 = the old
"corner speed 2 m ahead" rule; > 0 = brake early enough at a_brake, accelerate at most a_accel):
    v_max:=6.0 a_lat:=4.0 a_brake:=5.0 a_accel:=4.0
Sideslip model (2026-10-18, passed to both EKFs and state_estimate): beta_model (steer | slip),
slip_k (s^2/m; < 0 = from slip_file), slip_file. Default: slip with K from ~/.ros/vehicle_slip.yaml
(written by slip_calib); beta_model:=steer = the old kinematic model.
AMCL along-track slide (2026-10-19, passed to sim_localization): wheel_delay (s, the wheel speed's own
delay for both EKFs' delay compensation; default 0.010 = the simulator; -1 = off) and amcl_thin_walls
(default true: AMCL on its thin-wall copy of the map; false = the old /map).
amcl_corridor_sigma (2026-10-19, passed to sim_localization -> ekf_global): AMCL poses wider than
amcl_max_sigma are still used when their ellipse's narrow axis sigma <= this (default 0.10 m; 0 = off).
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
                          'steer_calib_file': LaunchConfiguration('steer_calib_file'),
                          'track': LaunchConfiguration('track'),
                          'ekf_test_fault': LaunchConfiguration('ekf_test_fault'),
                          'beta_model': LaunchConfiguration('beta_model'),
                          'slip_k': LaunchConfiguration('slip_k'),
                          'slip_file': LaunchConfiguration('slip_file'),
                          'wheel_delay': LaunchConfiguration('wheel_delay'),
                          'amcl_thin_walls': LaunchConfiguration('amcl_thin_walls'),
                          'amcl_corridor_sigma': LaunchConfiguration('amcl_corridor_sigma')}.items(),
    )
    driver = Node(
        package='state_estimation', executable='test_driver', name='test_driver', output='screen',
        parameters=[{'laps': ParameterValue(LaunchConfiguration('laps'), value_type=int),
                     'v_max': ParameterValue(LaunchConfiguration('v_max'), value_type=float),
                     'v_min': ParameterValue(LaunchConfiguration('v_min'), value_type=float),
                     'a_lat': ParameterValue(LaunchConfiguration('a_lat'), value_type=float),
                     'a_brake': ParameterValue(LaunchConfiguration('a_brake'), value_type=float),
                     'a_accel': ParameterValue(LaunchConfiguration('a_accel'), value_type=float),
                     'waypoints': ParameterValue(LaunchConfiguration('waypoints'), value_type=str)}],
    )
    logger = Node(
        package='state_estimation', executable='eval_logger', name='eval_logger', output='screen',
        parameters=[{'run_name': LaunchConfiguration('run_name'),
                     'track': ParameterValue(LaunchConfiguration('track'), value_type=str)}],
    )
    return LaunchDescription([
        DeclareLaunchArgument('run_name', default_value='run'),
        DeclareLaunchArgument('laps', default_value='3'),
        DeclareLaunchArgument('v_max', default_value='1.5'),
        DeclareLaunchArgument('v_min', default_value='0.7'),
        DeclareLaunchArgument('a_lat', default_value='1.5'),
        DeclareLaunchArgument('a_brake', default_value='0.0'),       # 0 = old 2-m preview rule
        DeclareLaunchArgument('a_accel', default_value='0.0'),       # 0 = no acceleration limit
        DeclareLaunchArgument('amcl_min_sigma_pos', default_value='0.10'),
        DeclareLaunchArgument('scale_file', default_value='~/.ros/ekf_speed_scale.yaml'),
        DeclareLaunchArgument('numpy_threads', default_value='1'),
        DeclareLaunchArgument('ekf_path_period', default_value='0.0'),
        DeclareLaunchArgument('ekf_truth_topic', default_value='none'),
        DeclareLaunchArgument('pipeline', default_value='merged'),     # or separate (3 processes)
        DeclareLaunchArgument('sim_sensors', default_value='v3'),      # or v1 (old gyro + speed models)
        DeclareLaunchArgument('steer_calib_file', default_value='~/.ros/steering_calibration.yaml'),
        DeclareLaunchArgument('track', default_value='levine'),         # apex_track track (2026-10-06)
        DeclareLaunchArgument('waypoints', default_value=''),           # '' = levine_centerline.csv
        DeclareLaunchArgument('ekf_test_fault', default_value=''),      # TEST ONLY, see sim_localization
        DeclareLaunchArgument('beta_model', default_value='slip'),      # sideslip model (2026-10-18): slip | steer
        DeclareLaunchArgument('slip_k', default_value='-1.0'),          # s^2/m; < 0 = from slip_file
        DeclareLaunchArgument('slip_file', default_value='~/.ros/vehicle_slip.yaml'),
        DeclareLaunchArgument('wheel_delay', default_value='0.010'),    # s, -1 = off (2026-10-19)
        DeclareLaunchArgument('amcl_thin_walls', default_value='true'), # AMCL on /map_amcl (2026-10-19)
        DeclareLaunchArgument('amcl_corridor_sigma', default_value='0.10'),  # m, 0 = off (2026-10-19)
        # CPU rule (also for test_driver and eval_logger, which start here)
        SetEnvironmentVariable('OPENBLAS_NUM_THREADS', LaunchConfiguration('numpy_threads')),
        SetEnvironmentVariable('OMP_NUM_THREADS', LaunchConfiguration('numpy_threads')),
        SetEnvironmentVariable('MKL_NUM_THREADS', LaunchConfiguration('numpy_threads')),
        localization, driver, logger,
    ])
