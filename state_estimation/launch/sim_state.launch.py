"""Start all state-estimation nodes for the simulator.

Run the simulator first (ros2 launch lidar_perception sim_perception.launch.py),
then in a second terminal:  ros2 launch state_estimation sim_state.launch.py
Optional: calibrate:=false to integrate the raw (uncalibrated) gyro.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    calibrate = DeclareLaunchArgument('calibrate', default_value='true')

    def node(name, params=None):
        return Node(package='state_estimation', executable=name, name=name,
                    output='screen', parameters=params or [])

    return LaunchDescription([
        calibrate,
        node('fake_vehicle_sensors'),
        node('fake_imu'),
        node('wheel_odometry'),
        node('imu_heading_check', [{
            'calibrate': ParameterValue(LaunchConfiguration('calibrate'), value_type=bool)}]),
        node('ekf_node'),
    ])
