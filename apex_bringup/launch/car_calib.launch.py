"""car_calib.launch.py -- the REAL car's calibration drives, with every path and car number from car.yaml.

    ros2 launch apex_bringup car_calib.launch.py tool:=wheel true_distance:=10.0   (straight drive, tape measure)
    ros2 launch apex_bringup car_calib.launch.py tool:=steer                        (slow drive, straights + curves)
    (slip constant, offline: ros2 run apex_bringup car_slip_calib <recordings> --map <map.yaml>)
Results go to car.yaml's calibration_dir (~/.ros/apex_car), where car.launch.py's EKFs read them -- never to the
simulator's ~/.ros. HOW to drive each one: the docstrings of state_estimation wheel_calib.py / steer_calib.py.
Needs the car's /vehicle/measured (and /imu/data for steer) from the ESP32 bridge.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from apex_bringup.car_calib import steer_calib_spec, wheel_calib_spec
from apex_bringup.car_config import load_car_config
from apex_bringup.car_nodes import safety_problems


def _calib(context):
    import os
    arg = lambda n: LaunchConfiguration(n).perform(context).strip()   # noqa: E731
    cfg, warnings = load_car_config(arg('config') or None)
    tool = arg('tool').lower()
    if tool == 'wheel':
        spec = wheel_calib_spec(cfg, float(arg('true_distance') or 0.0))
    elif tool == 'steer':
        spec = steer_calib_spec(cfg)
    else:
        raise RuntimeError(f"tool:={tool!r} -- use tool:=wheel (with true_distance:=<m>) or tool:=steer")
    problems = safety_problems([spec])
    if problems:
        raise RuntimeError('calibration refused, unsafe settings: ' + '; '.join(problems))
    os.makedirs(cfg['calibration_dir'], exist_ok=True)
    out = [LogInfo(msg=f"CAR CALIBRATION {tool}: settings {cfg['_path']} -> writes into {cfg['calibration_dir']}")]
    out += [LogInfo(msg=f'CAR WARNING: {w}') for w in warnings if tool == 'steer' or not w.startswith(('lf ', 'lr '))]
    return out + [Node(package=spec['package'], executable=spec['executable'], name=spec['name'], output='screen',
                       parameters=[spec['params']])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('tool', default_value='', description='wheel | steer'),
        DeclareLaunchArgument('true_distance', default_value='', description='wheel: tape-measured distance [m]'),
        DeclareLaunchArgument('config', default_value='', description="car settings YAML ('' = apex_bringup's)"),
        OpaqueFunction(function=_calib),
    ])
