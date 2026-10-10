"""obstacles.launch.py -- the LiDAR obstacle list (lidar_perception obstacle_node) -> /obstacles.

Needs the LiDAR scan, /state_estimate (state_estimation sim_localization.launch.py, or the real car's) and the
car's map on /map (in the simulator with obstacles: sim_perception.launch.py car_map:=...).
  ros2 launch lidar_perception obstacles.launch.py track:=closed1
laser_x / laser_y / laser_yaw = the LiDAR relative to the StateEstimate point (the CG): sim 0.275 / 0 / 0.
align_scans: true = put every scan on the map's walls before detection (step 3.2); false = step 3.1 behaviour.
range_offset: how much too FAR the LiDAR measures [m] (step 3.3), subtracted from the returns used by the alignment and
  every aligned scan (no effect with align_scans:=false). Default 0 (none); simulator (f1tenth_gym):
  range_offset:=0.022; REAL CAR: measure it (e.g. the car at a taped distance from a flat wall). Write it with a dot.
numpy_threads: BLAS threads (CPU rule, 2026-10-03) -- 1.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    f = lambda n: ParameterValue(LaunchConfiguration(n), value_type=float)  # noqa: E731
    return LaunchDescription([
        DeclareLaunchArgument('track', default_value='levine'),
        DeclareLaunchArgument('scan_topic', default_value='/scan_a1'),
        DeclareLaunchArgument('laser_x', default_value='0.275'),
        DeclareLaunchArgument('laser_y', default_value='0.0'),
        DeclareLaunchArgument('laser_yaw', default_value='0.0'),
        DeclareLaunchArgument('align_scans', default_value='true'),
        DeclareLaunchArgument('range_offset', default_value='0.0'),
        DeclareLaunchArgument('numpy_threads', default_value='1'),
        SetEnvironmentVariable('OPENBLAS_NUM_THREADS', LaunchConfiguration('numpy_threads')),
        SetEnvironmentVariable('OMP_NUM_THREADS', LaunchConfiguration('numpy_threads')),
        SetEnvironmentVariable('MKL_NUM_THREADS', LaunchConfiguration('numpy_threads')),
        Node(package='lidar_perception', executable='obstacle_node', name='obstacle_node', output='screen',
             parameters=[{'track': LaunchConfiguration('track'), 'scan_topic': LaunchConfiguration('scan_topic'),
                          'laser_x': f('laser_x'), 'laser_y': f('laser_y'), 'laser_yaw': f('laser_yaw'),
                          'align_scans': ParameterValue(LaunchConfiguration('align_scans'), value_type=bool),
                          'range_offset_m': f('range_offset')}]),
    ])
