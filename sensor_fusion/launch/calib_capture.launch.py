"""calib_capture.launch.py -- record a camera-LiDAR calibration session (sensor_fusion calib_capture).

  ros2 launch sensor_fusion calib_capture.launch.py                   (real car: /scan, /camera/image_compressed)
  ros2 launch sensor_fusion calib_capture.launch.py scan_topic:=/scan_a1 out_dir:=/tmp/calib
Needs: the LiDAR driver, the camera node (with /camera/camera_info) and, for the live "LiDAR sees the board" check,
apex_description sensor_frames (ros2 launch apex_description sensor_frames.launch.py).
Then:  ros2 run sensor_fusion calib_solve <out_dir>/session_<time> --mounts car
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    def arg(name):
        return LaunchConfiguration(name)

    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/camera/image_compressed'),
        DeclareLaunchArgument('image_compressed', default_value='true'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera_info'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('out_dir', default_value='~/calib_sessions'),
        DeclareLaunchArgument('still_s', default_value='1.0', description='hold still this long before a pose is saved'),
        DeclareLaunchArgument('max_poses', default_value='20'),
        Node(package='sensor_fusion', executable='calib_capture', name='calib_capture', output='screen',
             parameters=[{'image_topic': arg('image_topic'),
                          'image_compressed': ParameterValue(arg('image_compressed'), value_type=bool),
                          'camera_info_topic': arg('camera_info_topic'), 'scan_topic': arg('scan_topic'),
                          'out_dir': arg('out_dir'),
                          'still_s': ParameterValue(arg('still_s'), value_type=float),
                          'max_poses': ParameterValue(arg('max_poses'), value_type=int)}]),
    ])
