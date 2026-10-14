"""fake_camera.launch.py -- the simulator's camera + YOLO (sensor_fusion fake_camera) -> /camera/detections.

  ros2 launch sensor_fusion fake_camera.launch.py cones_file:=$HOME/apex_maps/closed1_cones/cones_truth.csv \
      world_map:=$HOME/apex_maps/closed1_cones/closed1_cones.yaml
Run it next to the simulator with a cone world (lidar_perception sim_perception.launch.py config:=... car_map:=...).
frame_id est/camera_optical_frame = the estimated car's camera (what the fusion uses, as on the car).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    def f(n):
        return ParameterValue(LaunchConfiguration(n), value_type=float)

    return LaunchDescription([
        DeclareLaunchArgument('cones_file', description='cones_truth.csv of the cone world (label, x, y, size_m)'),
        DeclareLaunchArgument('world_map', default_value='', description="the simulator's WORLD map yaml (walls hide cones)"),
        DeclareLaunchArgument('frame_id', default_value='est/camera_optical_frame'),
        DeclareLaunchArgument('latency_s', default_value='0.08'),
        DeclareLaunchArgument('fps', default_value='15.0'),
        DeclareLaunchArgument('p_miss', default_value='0.05'),
        DeclareLaunchArgument('false_per_picture', default_value='0.02'),
        DeclareLaunchArgument('cone_height', default_value='0.25'),
        DeclareLaunchArgument('seed', default_value='0'),
        Node(package='sensor_fusion', executable='fake_camera', name='fake_camera', output='screen',
             parameters=[{'cones_file': LaunchConfiguration('cones_file'),
                          'world_map': LaunchConfiguration('world_map'),
                          'frame_id': LaunchConfiguration('frame_id'), 'latency_s': f('latency_s'), 'fps': f('fps'),
                          'p_miss': f('p_miss'), 'false_per_picture': f('false_per_picture'),
                          'cone_height': f('cone_height'),
                          'seed': ParameterValue(LaunchConfiguration('seed'), value_type=int)}]),
    ])
