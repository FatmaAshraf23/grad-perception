"""sensor_frames.launch.py -- static TFs of the sensors from a mount YAML (apex_description sensor_frames).

  REAL CAR:   ros2 launch apex_description sensor_frames.launch.py            (mounts:=car, the default)
              -> base_link -> laser,  base_link -> camera_link -> camera_optical_frame
  simulator:  started by lidar_perception sim_perception.launch.py (mounts sim, prefixes ego_racecar/,est/, no laser)
Check:        ros2 run apex_description check_frames --mounts car
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mounts', default_value='car', description="'car', 'sim' or a path to a YAML"),
        DeclareLaunchArgument('prefixes', default_value='', description="comma separated frame prefixes"),
        DeclareLaunchArgument('publish_laser', default_value='true'),
        Node(package='apex_description', executable='sensor_frames', name='sensor_frames', output='screen',
             parameters=[{'mounts_file': ParameterValue(LaunchConfiguration('mounts'), value_type=str),
                          'prefixes': ParameterValue(LaunchConfiguration('prefixes'), value_type=str),
                          'publish_laser': ParameterValue(LaunchConfiguration('publish_laser'), value_type=bool)}]),
    ])
