"""Localization on the wall map: fake sensors + local EKF + AMCL + global EKF.

Run the simulator first (ros2 launch lidar_perception sim_perception.launch.py),
then:  ros2 launch state_estimation sim_localization.launch.py
Speed-scale calibration: both EKFs start from the k saved in scale_file
(default ~/.ros/ekf_speed_scale.yaml; ekf_global rewrites it). To start
uncalibrated (k = 1):  scale_file:=none
Start pose (must match sx, sy, stheta in sim.yaml):
       ros2 launch state_estimation sim_localization.launch.py x0:=0.0 y0:=0.0 yaw0:=0.0

Data flow
  fake_vehicle_sensors ─ /vehicle/measured ─┐
  fake_imu ──────────── /imu/data ──────────┼─> ekf_local  (odom frame) ─ TF odom->est/base_link
                                            │                               │
  /scan_a1 ─> sim_scan_relay ─ /scan_amcl ──┼──────────────> AMCL <─────────┘  (+ /map)
                                            │                  │ /amcl_pose, TF map->odom
                                            └─> ekf_global (map frame) <┘ ─> /odom/ekf
                                                                                 │
  apex_track (Frenet) ─────────────────────────────────> state_estimate <────────┘
                                                         ─> /state_estimate (apex_msgs/StateEstimate)
StateEstimate node: on by default; state_estimate:=false leaves it out.
numpy_threads (default 1): BLAS/OpenMP threads for every node started here (CPU rule,
2026-10-03). numpy's OpenBLAS otherwise starts one busy-waiting thread per core: ekf_node
used ~190 % CPU for 5x5 matrices. Same setting on the real car (Pi 4 has only 4 cores).
pipeline (2026-10-03):
  merged (default)   = ekf_local, ekf_global and state_estimate in ONE process (executable
                       state_pipeline; job 038: the three 80 % -> 36 % of one laptop core,
                       StateEstimate latency p95 8.2 -> 3.2 ms, accuracy unchanged):
                       the IMU and wheel messages are received once instead of 3 times, and
                       ekf_global hands its output to state_estimate directly. Same
                       parameters, topics and TF.
  separate           = the same three nodes as 3 processes (the old layout, for comparison)
sim_sensors (2026-10-03): model of the simulated gyro (fake_imu) and wheel speed
  (fake_vehicle_sensors). v1 = gyro from the pose difference per reading (0x/2x
  readings in turns), speed = the simulator's speed state (keeps "driving" in a
  simulator pause); v3 = pause_proof.PauseProofRate for both: smooth and pause-proof.
steer_calib_file (2026-10-03): steering calibration written by steer_calib, read by both
  EKFs and state_estimate (default ~/.ros/steering_calibration.yaml; none = no correction).
track (2026-10-06): apex_track track the StateEstimate projects onto (Frenet s, e_y, e_psi);
  default levine. Must match the simulator's map (closed1 = the closed-corner test track).
lifecycle (2026-10-03): who switches AMCL on. activator (default) = our lifecycle_activator,
  which re-reads AMCL's state and retries when a reply is lost; nav2 = nav2's lifecycle
  manager, which waited forever when its first reply was lost (jobs 040/041).
AMCL along-track slide (2026-10-19, see ekf_node / amcl_map.py / amcl_sim.yaml):
  wheel_delay (s): the wheel speed's own delay for both EKFs' delay compensation (ekf_node
    wheel_delay_s): simulator 0.010 (default here); real car: measure it; -1 = off (the old odometry).
  amcl_thin_walls (true): AMCL gets its own copy of /map with only the wall SURFACE occupied
    (node amcl_map_filter -> /map_amcl); the EKF's scan check and the obstacle detector keep /map.
    false = AMCL on /map (the old behaviour).
  amcl_offset (0.025 m = half the map cell): nav2 AMCL's map is shifted by half a cell; the global EKF
    adds it to AMCL's poses (ekf_node amcl_offset_m) and AMCL's START pose gets it subtracted, so AMCL
    starts exactly where the car is. (Its particles start with no spread; with the small alphas a
    2.5 cm start error took until the first corner to disappear.)
"""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


class _QuotedStr(str):
    """A string written in quotes, so the ROS YAML parser never reads it as a bool/number."""


yaml.add_representer(_QuotedStr, lambda dumper, s: dumper.represent_scalar(
    'tag:yaml.org,2002:str', s, style='"'), Dumper=yaml.SafeDumper)


def _params_file(sections):
    """{node_name: {param: value}} -> YAML params file with one section per node; its path."""
    doc = {f'/{name}': {'ros__parameters': {k: _QuotedStr(v) if isinstance(v, str) else v
                                            for k, v in params.items()}}
           for name, params in sections.items()}
    fd, path = tempfile.mkstemp(prefix='state_pipeline_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.dump(doc, f, Dumper=yaml.SafeDumper, default_flow_style=False)
    return path


def _ekfs_and_state_estimate(context):
    """ekf_local + ekf_global (+ state_estimate): 3 processes, or ONE (pipeline:=merged)."""
    arg = lambda n: LaunchConfiguration(n).perform(context).strip()   # noqa: E731
    yes = lambda n: arg(n).lower() in ('1', 'true', 'yes', 'on')      # noqa: E731
    slip = {'beta_model': arg('beta_model'), 'slip_k': float(arg('slip_k')), 'slip_file': arg('slip_file')}
    common = dict(slip, **{'path_period_s': float(arg('ekf_path_period')), 'scale_file': arg('scale_file'),
                           'steer_calib_file': arg('steer_calib_file'),
                           'wheel_delay_s': float(arg('wheel_delay'))})
    se = dict(slip, **{'steer_calib_file': arg('steer_calib_file'), 'track': arg('track')})
    # LOCAL EKF: odometry for AMCL, starts at the origin of the odom frame
    local = dict(common, **{
        'mode': 'odom', 'frame_id': 'odom', 'child_frame_id': 'est/base_link',
        'publish_tf': True, 'use_amcl': False,
        'x0': 0.0, 'y0': 0.0, 'yaw0': 0.0,
        'odom_topic': '/odom/ekf_local', 'path_topic': '/odom/ekf_local_path',
        'diag_prefix': '/ekf_local'})
    # GLOBAL EKF: the pose planning/MPC will use
    glob = dict(common, **{
        'mode': 'map', 'frame_id': 'map', 'child_frame_id': 'est/base_link',
        'publish_tf': False, 'use_amcl': yes('use_amcl'), 'amcl_offset_m': float(arg('amcl_offset')),
        'amcl_min_sigma_pos': float(arg('amcl_min_sigma_pos')),
        'x0': float(arg('x0')), 'y0': float(arg('y0')), 'yaw0': float(arg('yaw0')),
        'odom_topic': '/odom/ekf', 'path_topic': '/odom/ekf_path',
        'diag_prefix': '/ekf', 'truth_topic': arg('ekf_truth_topic'),
        'test_fault': arg('ekf_test_fault')})
    with_se = yes('state_estimate')
    pipeline = arg('pipeline').lower()

    if pipeline == 'merged':
        # ONE process. Node names come from the code (a __node remap would rename all three);
        # parameters from a file with one section per node name.
        return [Node(package='state_estimation', executable='state_pipeline', output='screen',
                     parameters=[_params_file({'ekf_local': local, 'ekf_global': glob,
                                               'state_estimate': se})],
                     arguments=[] if with_se else ['--no-state-estimate'])]
    if pipeline != 'separate':
        raise RuntimeError(f"pipeline:={pipeline} -- use 'separate' or 'merged'")
    actions = [
        Node(package='state_estimation', executable='ekf_node', name='ekf_local',
             output='screen', parameters=[local]),
        Node(package='state_estimation', executable='ekf_node', name='ekf_global',
             output='screen', parameters=[glob]),
    ]
    if with_se:
        # StateEstimate for control: /odom/ekf + apex_track -> /state_estimate (contract v0.2)
        actions.append(Node(package='state_estimation', executable='state_estimate',
                            name='state_estimate', output='screen', parameters=[se]))
    return actions


def _amcl(context):
    """AMCL on /map, or (amcl_thin_walls:=true) on its thin-wall copy /map_amcl made by amcl_map_filter."""
    arg = lambda n: LaunchConfiguration(n).perform(context).strip()   # noqa: E731
    thin = arg('amcl_thin_walls').lower() in ('1', 'true', 'yes', 'on')
    off = float(arg('amcl_offset'))                 # AMCL's frame = the true map shifted by half a cell
    amcl_yaml = os.path.join(get_package_share_directory('state_estimation'), 'config', 'amcl_sim.yaml')
    actions = []
    if thin:
        actions.append(Node(package='state_estimation', executable='amcl_map_filter', name='amcl_map_filter',
                            output='screen', parameters=[{'in_topic': '/map', 'out_topic': '/map_amcl'}]))
    # map_topic picks AMCL's map; the remapping makes sure even if an AMCL version ignored map_topic
    actions.append(Node(package='nav2_amcl', executable='amcl', name='amcl', output='screen',
                        parameters=[amcl_yaml, {'initial_pose.x': float(arg('x0')) - off,
                                                'initial_pose.y': float(arg('y0')) - off,
                                                'initial_pose.yaw': float(arg('yaw0')),
                                                'map_topic': '/map_amcl' if thin else '/map'}],
                        remappings=[('/map', '/map_amcl')] if thin else []))
    return actions


def generate_launch_description():
    x0 = LaunchConfiguration('x0')
    y0 = LaunchConfiguration('y0')
    yaw0 = LaunchConfiguration('yaw0')
    f = lambda s: ParameterValue(s, value_type=float)  # noqa: E731

    def se(name, executable=None, params=None):
        return Node(package='state_estimation', executable=executable or name, name=name,
                    output='screen', parameters=params or [])

    return LaunchDescription([
        DeclareLaunchArgument('x0', default_value='0.0'),
        DeclareLaunchArgument('y0', default_value='0.0'),
        DeclareLaunchArgument('yaw0', default_value='0.0'),
        DeclareLaunchArgument('use_amcl', default_value='true'),
        DeclareLaunchArgument('amcl_min_sigma_pos', default_value='0.10'),
        DeclareLaunchArgument('scale_file', default_value='~/.ros/ekf_speed_scale.yaml'),
        DeclareLaunchArgument('state_estimate', default_value='true'),
        DeclareLaunchArgument('numpy_threads', default_value='1'),
        # View/sim-only extras of the EKFs (CPU, 2026-10-03): path for Foxglove (s, 0 = off)
        # and the simulator truth in ekf_global's 2-s report ('none' = off). Both OFF by default:
        # job 036 -> truth (250 Hz) doubled ekf_global's CPU (44 % -> 22 % without it);
        # viewing: ekf_path_period:=0.2 ekf_truth_topic:=/ego_racecar/odom
        DeclareLaunchArgument('ekf_path_period', default_value='0.0'),
        DeclareLaunchArgument('ekf_truth_topic', default_value='none'),
        # merged = ONE process (state_pipeline, default since job 038); separate = 3 processes
        DeclareLaunchArgument('pipeline', default_value='merged'),
        # simulated gyro + wheel-speed model (simulation only): v3 (pause-proof, default since
        # job 041: StateEstimate r error p95 0.31 -> 0.011 rad/s) or v1 (old)
        DeclareLaunchArgument('sim_sensors', default_value='v3'),
        # who switches AMCL on: activator (retries, default) or nav2 (old lifecycle manager);
        # activator_timeout = seconds without a reply before it re-reads the state (tests: 0.05)
        DeclareLaunchArgument('lifecycle', default_value='activator'),
        # steering calibration (steer_calib) for both EKFs + state_estimate; none = no correction
        DeclareLaunchArgument('steer_calib_file', default_value='~/.ros/steering_calibration.yaml'),
        # sideslip model (2026-10-18, vehicle_model.py): steer | slip; slip_k < 0 = from slip_file
        DeclareLaunchArgument('beta_model', default_value='slip'),
        DeclareLaunchArgument('slip_k', default_value='-1.0'),
        DeclareLaunchArgument('slip_file', default_value='~/.ros/vehicle_slip.yaml'),
        DeclareLaunchArgument('activator_timeout', default_value='2.0'),
        # apex_track track for the StateEstimate (2026-10-06): levine | closed1 | <our track>
        DeclareLaunchArgument('track', default_value='levine'),
        # TEST ONLY (2026-10-06): 't_s,dx,dy,dyaw_deg' moves ekf_global's pose once, t_s s after its
        # first IMU message -> tests the robust jump to AMCL (ekf_node RECOVERY). '' = off (always, normally)
        DeclareLaunchArgument('ekf_test_fault', default_value=''),
        # AMCL along-track slide (2026-10-19, docstring): wheel-speed delay [s] for both EKFs
        # (-1 = compensation off) and AMCL's thin-wall map copy
        DeclareLaunchArgument('wheel_delay', default_value='0.010'),
        DeclareLaunchArgument('amcl_thin_walls', default_value='true'),
        DeclareLaunchArgument('amcl_offset', default_value='0.025'),   # m, half the map cell (docstring)

        # CPU rule: limit numpy's math-library threads for every node below
        SetEnvironmentVariable('OPENBLAS_NUM_THREADS', LaunchConfiguration('numpy_threads')),
        SetEnvironmentVariable('OMP_NUM_THREADS', LaunchConfiguration('numpy_threads')),
        SetEnvironmentVariable('MKL_NUM_THREADS', LaunchConfiguration('numpy_threads')),

        # Simulated car sensors
        se('fake_vehicle_sensors', params=[{'model': LaunchConfiguration('sim_sensors')}]),
        se('fake_imu', params=[{'model': LaunchConfiguration('sim_sensors')}]),
        # Old wheel-only odometry, kept for comparison (red path)
        se('wheel_odometry', params=[{'x0': f(x0), 'y0': f(y0), 'yaw0': f(yaw0)}]),

        # The two EKFs + StateEstimate: 3 processes or one (see pipeline above)
        OpaqueFunction(function=_ekfs_and_state_estimate),

        # Scan in the estimated car's frame (simulation only)
        se('sim_scan_relay'),

        # AMCL (a lifecycle node, started by its lifecycle manager) + its thin-wall map (_amcl)
        OpaqueFunction(function=_amcl),
        # AMCL is a lifecycle node: it waits to be configured and activated. Default: our
        # lifecycle_activator, which re-reads the state and retries when a reply is lost. nav2's
        # lifecycle manager waited forever then (jobs 040/041: AMCL never activated, no fixes;
        # starting it 1.5 s later did not help). lifecycle:=nav2 = the old manager.
        Node(package='state_estimation', executable='lifecycle_activator', name='lifecycle_activator',
             output='screen', condition=LaunchConfigurationEquals('lifecycle', 'activator'),
             parameters=[{'node_names': ['amcl'],
                          'call_timeout_s': f(LaunchConfiguration('activator_timeout'))}]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_localization', output='screen',
             condition=LaunchConfigurationEquals('lifecycle', 'nav2'),
             parameters=[{'autostart': True, 'node_names': ['amcl']}]),
    ])
