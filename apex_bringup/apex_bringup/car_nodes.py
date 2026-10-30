"""car_nodes.py -- WHAT the real car starts, as plain data (no ROS imports): car.launch.py turns these specs into
Node actions, and test/test_car_nodes.py checks them without ROS.

A spec = {'package', 'executable', 'name', 'params': {...}, 'remap': [(from, to)], 'merged': {...} (state_pipeline)}.

Frames on the car (the same names the simulator's estimator uses, so AMCL / EKFs / obstacle node run exactly as
tested):
    map --(AMCL)--> odom --(ekf_local)--> est/base_link --(sensor_frames, static)--> est/laser
                                                                                  \\-> est/camera_link -> ..._optical
Stages (launch arguments):
    lidar         RPLIDAR driver (/scan_a1_raw, frame est/laser) -> scan_deskew -> /scan_a1; sensor_frames
    localization  map_server + AMCL (thin-wall copy) + ekf_local + ekf_global + state_estimate (one process)
                  -- needs /imu/data and /vehicle/measured (the ESP32 bridge, not written yet)
The LiDAR's position comes ONLY from the mounts YAML: the same numbers go to sensor_frames (TF), scan_deskew and
both EKFs (map check). Simulator / test switches are never passed; the two whose node default is not already safe
are passed explicitly safe (allow_test_faults false, sim_wheel_ideal_topic '').
"""
import math
import os

SCAN_RAW, SCAN = '/scan_a1_raw', '/scan_a1'
BASE, LASER = 'est/base_link', 'est/laser'
LOCAL_ODOM = '/odom/ekf_local'


def laser_pose(mounts):
    """(x, y, yaw_rad) of the LiDAR on the car from a checked mounts dict (apex_description.mounts.load_mounts)."""
    s = mounts['laser']
    return float(s['x']), float(s['y']), math.radians(float(s['yaw']))


def calib_files(cfg):
    d = cfg['calibration_dir']
    return {'scale_file': os.path.join(d, 'ekf_speed_scale.yaml'),
            'scale_learned_file': os.path.join(d, 'ekf_speed_scale_learned.yaml'),
            'steer_calib_file': os.path.join(d, 'steering_calibration.yaml'),
            'slip_file': os.path.join(d, 'vehicle_slip.yaml')}


def lidar_specs(cfg, mounts):
    lx, ly, lyaw = laser_pose(mounts)
    return [
        {'package': 'rplidar_ros', 'executable': 'rplidar_node', 'name': 'rplidar_node',
         'params': {'channel_type': 'serial', 'serial_port': cfg['serial_port'],
                    'serial_baudrate': int(cfg['serial_baudrate']), 'frame_id': LASER, 'inverted': False,
                    'angle_compensate': True, 'scan_mode': cfg['scan_mode'],
                    'scan_frequency': float(cfg['scan_frequency']), 'topic_name': SCAN_RAW}},
        {'package': 'lidar_perception', 'executable': 'scan_deskew', 'name': 'scan_deskew',
         'params': {'in_topic': SCAN_RAW, 'out_topic': SCAN, 'odom_topic': LOCAL_ODOM,
                    'mode': cfg['deskew_mode'], 'time_order': cfg['deskew_time_order'],
                    'laser_x': lx, 'laser_y': ly, 'laser_yaw': lyaw}},
        {'package': 'apex_description', 'executable': 'sensor_frames', 'name': 'sensor_frames',
         'params': {'mounts_file': mounts['_path'], 'prefixes': 'est/', 'publish_laser': True}},
    ]


def ekf_common(cfg, mounts):
    """Parameters both EKFs get (the car's numbers + calibration files + the LiDAR pose + safe test switches)."""
    lx, ly, lyaw = laser_pose(mounts)
    p = {k: cfg[k] for k in ('lf', 'lr', 'imu_yaw_deg', 'beta_model', 'slip_k', 'wheel_delay_s', 'slip_lag_s',
                             'imu_gap_s', 'slip_check', 'slip_use_observer', 'slip_r0', 'imu_gap_predict',
                             'imu_gap_kus', 'zupt_gate', 'zupt_turn_max_s')}
    p.update(calib_files(cfg))
    p.update({'laser_x': lx, 'laser_y': ly, 'laser_yaw': lyaw, 'scan_topic': SCAN, 'map_topic': '/map',
              'path_period_s': 0.0, 'truth_topic': 'none',
              # explicitly SAFE (simulator / test only): never anything else on the car
              'sim_wheel_ideal_topic': '', 'allow_test_faults': False})
    return p


def localization_specs(cfg, mounts, map_yaml, resolution, track, x0, y0, yaw0, amcl_yaml):
    """map_server, AMCL (+ thin-wall map), lifecycle activator, and the merged estimator process."""
    off = resolution / 2.0          # nav2 AMCL's map frame is shifted by half a cell (sim_localization docstring)
    common = ekf_common(cfg, mounts)
    local = dict(common, **{'mode': 'odom', 'frame_id': 'odom', 'child_frame_id': BASE, 'publish_tf': True,
                            'use_amcl': False, 'x0': 0.0, 'y0': 0.0, 'yaw0': 0.0, 'map_check': False,
                            'odom_topic': LOCAL_ODOM, 'path_topic': '/odom/ekf_local_path',
                            'diag_prefix': '/ekf_local'})
    glob = dict(common, **{'mode': 'map', 'frame_id': 'map', 'child_frame_id': BASE, 'publish_tf': False,
                           'use_amcl': True, 'amcl_offset_m': off, 'map_check': cfg['map_check'],
                           'amcl_corridor_sigma': cfg['amcl_corridor_sigma'],
                           'amcl_min_sigma_pos': cfg['amcl_min_sigma_pos'],
                           'x0': x0, 'y0': y0, 'yaw0': yaw0, 'odom_topic': '/odom/ekf', 'path_topic': '/odom/ekf_path',
                           'diag_prefix': '/ekf', 'initialpose_topic': '/initialpose'})
    se = {k: common[k] for k in ('lf', 'lr', 'beta_model', 'slip_k', 'slip_file', 'steer_calib_file')}
    se.update({'track': track, 'initialpose_topic': '/initialpose'})
    return [
        {'package': 'nav2_map_server', 'executable': 'map_server', 'name': 'map_server',
         'params': {'yaml_filename': map_yaml, 'topic_name': 'map', 'frame_id': 'map'}},
        {'package': 'state_estimation', 'executable': 'amcl_map_filter', 'name': 'amcl_map_filter',
         'params': {'in_topic': '/map', 'out_topic': '/map_amcl'}},
        {'package': 'nav2_amcl', 'executable': 'amcl', 'name': 'amcl', 'params_file': amcl_yaml,
         'params': {'initial_pose.x': x0 - off, 'initial_pose.y': y0 - off, 'initial_pose.yaw': yaw0,
                    'map_topic': '/map_amcl', 'scan_topic': SCAN, 'base_frame_id': BASE},
         'remap': [('/map', '/map_amcl')]},
        {'package': 'state_estimation', 'executable': 'lifecycle_activator', 'name': 'lifecycle_activator',
         'params': {'node_names': ['map_server', 'amcl'], 'call_timeout_s': 2.0}},
        {'package': 'state_estimation', 'executable': 'state_pipeline', 'name': 'state_pipeline',
         'merged': {'ekf_local': local, 'ekf_global': glob, 'state_estimate': se}},
    ]


def map_resolution(map_yaml):
    """The map's cell size [m] from its map_server YAML."""
    import yaml
    with open(os.path.expanduser(map_yaml)) as f:
        m = yaml.safe_load(f)
    r = float(m['resolution'])
    if not 0.005 <= r <= 0.5:
        raise ValueError(f'{map_yaml}: resolution {r} m is not a plausible map cell size')
    return r


def all_params(specs):
    """Every (node, param, value) of a spec list, including the merged process's sections (for checks / tests)."""
    out = []
    for s in specs:
        for k, v in s.get('params', {}).items():
            out.append((s['name'], k, v))
        for node, params in s.get('merged', {}).items():
            for k, v in params.items():
                out.append((node, k, v))
    return out


def safety_problems(specs):
    """[] if no spec could switch on a simulator / test feature or read the simulator's calibrations."""
    problems = []
    sim_ros = os.path.normpath(os.path.expanduser('~/.ros'))
    for node, k, v in all_params(specs):
        if k == 'allow_test_faults' and v is not False:
            problems.append(f'{node}.{k} = {v!r}')
        elif k == 'sim_wheel_ideal_topic' and v != '':
            problems.append(f'{node}.{k} = {v!r}')
        elif k.startswith('test_') and v not in ('', 0, 0.0, False):
            problems.append(f'{node}.{k} = {v!r}')
        elif k.endswith('_file') and isinstance(v, str) and v and \
                os.path.dirname(os.path.normpath(os.path.expanduser(v))) == sim_ros:
            problems.append(f"{node}.{k} = {v!r} (the simulator's calibration folder)")
    for s in specs:
        if s['package'] == 'state_estimation' and s['executable'] in ('fake_imu', 'fake_vehicle_sensors',
                                                                       'sim_scan_relay', 'test_driver', 'eval_logger'):
            problems.append(f"simulator node {s['executable']}")
    return problems
