"""fusion.launch.py -- camera confirmation of the LiDAR obstacles (sensor_fusion fusion_node) -> /obstacles/fused.

Needs /obstacles (lidar_perception obstacles.launch.py), /camera/detections + /camera/camera_info (simulator:
sensor_fusion fake_camera.launch.py; car: camera_publisher yolo_node) and /state_estimate.
  ros2 launch sensor_fusion fusion.launch.py                       # simulator (mount 'sim', camera_info topic)
  ros2 launch sensor_fusion fusion.launch.py mounts:=car camera_info_file:=<car camera yaml> camera_delay_s:=<measured>
camera_delay_s: the stamp offset latency_probe measured on the car's camera (simulator 0: fake_camera stamps exactly).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    def f(n):
        return ParameterValue(LaunchConfiguration(n), value_type=float)

    return LaunchDescription([
        DeclareLaunchArgument('mounts', default_value='sim', description="apex_description mount file: sim | car | path"),
        DeclareLaunchArgument('camera_info_file', default_value='', description="'' = the /camera/camera_info topic"),
        DeclareLaunchArgument('camera_delay_s', default_value='0.0'),
        DeclareLaunchArgument('cone_height', default_value='0.25'),
        DeclareLaunchArgument('gate_k', default_value='3.0'),
        DeclareLaunchArgument('sigma_mount_deg', default_value='0.5'),
        DeclareLaunchArgument('publish_markers', default_value='true'),
        DeclareLaunchArgument('numpy_threads', default_value='1'),
        SetEnvironmentVariable('OPENBLAS_NUM_THREADS', LaunchConfiguration('numpy_threads')),
        SetEnvironmentVariable('OMP_NUM_THREADS', LaunchConfiguration('numpy_threads')),
        SetEnvironmentVariable('MKL_NUM_THREADS', LaunchConfiguration('numpy_threads')),
        Node(package='sensor_fusion', executable='fusion_node', name='fusion_node', output='screen',
             parameters=[{'mounts': LaunchConfiguration('mounts'),
                          'camera_info_file': LaunchConfiguration('camera_info_file'),
                          'camera_delay_s': f('camera_delay_s'), 'cone_height': f('cone_height'),
                          'gate_k': f('gate_k'), 'sigma_mount_deg': f('sigma_mount_deg'),
                          'publish_markers': ParameterValue(LaunchConfiguration('publish_markers'), value_type=bool)}]),
    ])
