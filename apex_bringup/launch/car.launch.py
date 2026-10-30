"""car.launch.py -- start the REAL car's perception + state estimation (apex_bringup, 2026-10-30).

    ros2 launch apex_bringup car.launch.py                         LiDAR only (today: RPLIDAR + de-skew + frames)
    ros2 launch apex_bringup car.launch.py localization:=true map:=<map.yaml> track:=<track> x0:=.. y0:=.. yaw0:=..
                                                                   + map, AMCL, both EKFs, StateEstimate
    ros2 launch apex_bringup car.launch.py config:=<other car.yaml>   (default: this package's config/car.yaml)
Check the settings without starting anything:  ros2 run apex_bringup check_car_config

Before anything starts, the launch
  1. reads + checks car.yaml (car_config.py): unknown keys, wrong types / sizes, simulator / test switches -> STOP,
  2. reads + checks the sensor mounts (apex_description sensor_mounts_car.yaml, or mounts:=<path>),
  3. builds the node list (car_nodes.py) and checks it once more (no test switch, no simulator node, no simulator
     calibration file) -> STOP on a problem,
  4. prints every placeholder (car.yaml not_measured, mounts status placeholder) as a WARNING.
What the nodes are, which frames and why: apex_bringup/car_nodes.py.
NOT here (simulator only, by design): fake_imu, fake_vehicle_sensors, sim_scan_relay, test_driver, eval_logger,
allow_test_faults, /sim/wheel_ideal.
"""
import os
import tempfile

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from apex_bringup.car_config import load_car_config
from apex_bringup.car_nodes import (lidar_specs, localization_specs, map_resolution, safety_problems)


class _QuotedStr(str):
    """A string written in quotes, so the ROS YAML parser never reads it as a bool / number."""


yaml.add_representer(_QuotedStr, lambda dumper, s: dumper.represent_scalar(
    'tag:yaml.org,2002:str', s, style='"'), Dumper=yaml.SafeDumper)


def _params_file(sections):
    """{node_name: {param: value}} -> a ROS params file with one section per node (state_pipeline = 3 nodes)."""
    doc = {f'/{name}': {'ros__parameters': {k: _QuotedStr(v) if isinstance(v, str) else v
                                            for k, v in params.items()}}
           for name, params in sections.items()}
    fd, path = tempfile.mkstemp(prefix='car_pipeline_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.dump(doc, f, Dumper=yaml.SafeDumper, default_flow_style=False)
    return path


def _node(spec):
    if 'merged' in spec:   # ONE process, three nodes; names come from the code, parameters from a file
        return Node(package=spec['package'], executable=spec['executable'], output='screen',
                    parameters=[_params_file(spec['merged'])])
    params = ([spec['params_file']] if spec.get('params_file') else []) + [spec.get('params', {})]
    return Node(package=spec['package'], executable=spec['executable'], name=spec['name'], output='screen',
                parameters=params, remappings=spec.get('remap', []))


def _check_rplidar():
    """The RPLIDAR driver must be OUR source build (Slamtec ros2 branch 24cc9b6, checked against a1_emulator /
    scan_deskew in job 163, used for the real tests on 22 Oct). Job 215: the laptop ALSO has the apt package
    ros-humble-rplidar-ros (/opt/ros/humble, same version number 2.1.4 but an older build: it cannot set the scan
    frequency). Whichever is sourced last wins, silently -> refuse the apt one."""
    from ament_index_python.packages import get_package_prefix
    hint = 'source the driver workspace first: source ~/drivers_ws/install/local_setup.bash'
    try:
        prefix = get_package_prefix('rplidar_ros')
    except PackageNotFoundError:
        raise RuntimeError(f'package rplidar_ros not found -- {hint}')
    if prefix.startswith('/opt/ros/'):
        raise RuntimeError(f'rplidar_ros comes from {prefix} (the apt package, NOT the driver build we checked) -- '
                           f'{hint}')
    return prefix


def _car(context):
    arg = lambda n: LaunchConfiguration(n).perform(context).strip()   # noqa: E731
    yes = lambda n: arg(n).lower() in ('1', 'true', 'yes', 'on')      # noqa: E731
    from apex_description.mounts import load_mounts

    cfg, warnings = load_car_config(arg('config') or None)
    mounts = load_mounts(arg('mounts'))
    if mounts['status'] == 'simulator':
        raise RuntimeError(f"mounts {mounts['_path']}: these are the SIMULATOR's sensor positions -- the car "
                           'needs sensor_mounts_car.yaml')
    if mounts['status'] == 'placeholder':
        warnings.append(f"sensor mounts {mounts['_path']} are PLACEHOLDERS (not measured on this car)")

    specs, info = [], []
    if yes('lidar'):
        info.append(LogInfo(msg=f'CAR: RPLIDAR driver {_check_rplidar()}'))
        specs += lidar_specs(cfg, mounts)
    if yes('localization'):
        missing = [n for n in ('map', 'track', 'x0', 'y0', 'yaw0') if not arg(n)]
        if missing:
            raise RuntimeError(f"localization:=true needs {', '.join(n + ':=...' for n in missing)} -- no defaults "
                               "on the car (a wrong map / track / start pose would be used silently)")
        map_yaml = os.path.abspath(os.path.expanduser(arg('map')))
        amcl_yaml = os.path.join(get_package_share_directory('apex_bringup'), 'config', 'amcl_car.yaml')
        specs += localization_specs(cfg, mounts, map_yaml, map_resolution(map_yaml), arg('track'),
                                    float(arg('x0')), float(arg('y0')), float(arg('yaw0')), amcl_yaml)
    problems = safety_problems(specs)
    if problems:
        raise RuntimeError('car launch refused, unsafe settings: ' + '; '.join(problems))

    os.makedirs(cfg['calibration_dir'], exist_ok=True)
    out = [LogInfo(msg=f"CAR: settings {cfg['_path']}, mounts {mounts['_path']} (status {mounts['status']}), "
                       f"calibrations {cfg['calibration_dir']}, stages: lidar {yes('lidar')}, "
                       f"localization {yes('localization')}")] + info
    out += [LogInfo(msg=f'CAR WARNING: {w}') for w in warnings]
    if not specs:
        out.append(LogInfo(msg='CAR: nothing to start (lidar:=false localization:=false)'))
    return out + [_node(s) for s in specs]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value='', description="car settings YAML ('' = apex_bringup's)"),
        DeclareLaunchArgument('mounts', default_value='car', description="sensor mounts: 'car' or a path"),
        DeclareLaunchArgument('lidar', default_value='true'),
        DeclareLaunchArgument('localization', default_value='false',
                              description='map + AMCL + EKFs + StateEstimate (needs /imu/data, /vehicle/measured)'),
        DeclareLaunchArgument('map', default_value='', description='map_server YAML of the real track'),
        DeclareLaunchArgument('track', default_value='', description='apex_track track name for the StateEstimate'),
        DeclareLaunchArgument('x0', default_value=''),
        DeclareLaunchArgument('y0', default_value=''),
        DeclareLaunchArgument('yaw0', default_value=''),
        # CPU rule (sim_localization): one math-library thread per node, the Pi has 4 cores
        SetEnvironmentVariable('OPENBLAS_NUM_THREADS', '1'),
        SetEnvironmentVariable('OMP_NUM_THREADS', '1'),
        SetEnvironmentVariable('MKL_NUM_THREADS', '1'),
        OpaqueFunction(function=_car),
    ])
