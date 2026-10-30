#!/usr/bin/env python3
"""Tests for car_config.py (the real car's settings checker) and car_nodes.py (what the car launch starts).
No ROS needed (yaml only). Run: python3 test_car_config.py   (or pytest)"""
import copy
import math
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from apex_bringup.car_config import ALL_NAMES, CarConfigError, _forbidden, check, load_car_config  # noqa: E402
from apex_bringup.car_nodes import (all_params, ekf_common, lidar_specs, localization_specs,  # noqa: E402
                                    map_resolution, safety_problems)

CAR_YAML = os.path.join(HERE, '..', 'config', 'car.yaml')
MOUNTS = {'_path': '/x/sensor_mounts_car.yaml', 'status': 'placeholder',
          'laser': {'x': 0.12, 'y': -0.01, 'z': 0.15, 'yaw': 180.0},
          'camera': {'x': 0.05, 'y': 0.0, 'z': 0.18, 'roll': 0.0, 'pitch': 8.0, 'yaw': 0.0}}


def doc():
    with open(CAR_YAML) as f:
        return yaml.safe_load(f)


def refused(d, *words):
    """check(d) must raise and the message must contain every word."""
    try:
        check(d)
    except CarConfigError as e:
        for w in words:
            assert w in str(e), f'{w!r} not in: {e}'
        return str(e)
    raise AssertionError('not refused')


def loc_specs(cfg):
    return localization_specs(cfg, MOUNTS, '/m/track.yaml', 0.05, 'ourtrack', 1.0, 2.0, 0.5, '/c/amcl_car.yaml')


# ---------------------------------------------------------------- car_config
def test_shipped_file_loads_with_placeholder_warnings():
    cfg, warn = load_car_config(CAR_YAML)
    assert cfg['lf'] == 0.15875 and cfg['beta_model'] == 'steer'
    assert cfg['calibration_dir'].endswith(os.path.join('.ros', 'apex_car'))
    for n in cfg['not_measured']:
        assert any(w.startswith(n + ' = ') for w in warn), n
    # the safe defaults the car starts with
    assert cfg['slip_use_observer'] is False and cfg['imu_gap_predict'] is False and cfg['slip_check'] is True


def test_no_schema_name_looks_forbidden():
    for n in ALL_NAMES:
        assert not _forbidden(n), n


def test_unknown_key_refused():
    d = doc()
    d['estimator']['slip_ro'] = 0.3            # typo of slip_r0
    refused(d, 'estimator.slip_ro', 'unknown key')
    d = doc()
    d['speed_limit'] = 3
    refused(d, 'speed_limit', 'unknown key')


def test_test_and_sim_switches_refused_anywhere():
    for sec, key, val in (('estimator', 'allow_test_faults', True), (None, 'allow_test_faults', False),
                          ('timing', 'test_lag_ms', 12.0), ('lidar', 'sim_wheel_ideal_topic', '/sim/wheel_ideal'),
                          (None, 'fake_imu', True)):
        d = doc()
        (d[sec] if sec else d)[key] = val
        refused(d, key, 'SIMULATOR / TEST')


def test_wrong_types_refused():
    d = doc()
    d['estimator']['slip_check'] = 'true'      # a string, not a bool
    refused(d, 'estimator.slip_check', 'true or false')
    d = doc()
    d['vehicle']['lf'] = '0.16'
    refused(d, 'vehicle.lf', 'number')
    d = doc()
    d['vehicle']['lf'] = True
    refused(d, 'vehicle.lf', 'number')
    d = doc()
    d['vehicle']['lf'] = float('nan')
    refused(d, 'vehicle.lf', 'number')


def test_impossible_sizes_refused():
    d = doc()
    d['vehicle']['lf'] = 160.0                 # mm typed as m
    refused(d, 'vehicle.lf = 160.0', 'outside')
    d = doc()
    d['vehicle']['imu_yaw_deg'] = 270.0
    refused(d, 'imu_yaw_deg')
    d = doc()
    d['lidar']['serial_baudrate'] = 9600
    refused(d, 'serial_baudrate', '115200')
    d = doc()
    d['lidar']['serial_port'] = 'COM3'
    refused(d, 'serial_port', '/dev/')
    d = doc()
    d['vehicle']['beta_model'] = 'dynamic'
    refused(d, 'beta_model')


def test_simulator_calibration_folder_refused():
    for p in ('~/.ros', '~/.ros/', os.path.expanduser('~/.ros')):
        d = doc()
        d['calibration_dir'] = p
        refused(d, 'calibration_dir', 'SIMULATOR')


def test_not_measured_checked():
    d = doc()
    d['not_measured'] = ['lf', 'lff']
    refused(d, "'lff'", 'not a setting name')
    d = doc()
    d['not_measured'] = ['lf', 'lf']
    refused(d, 'twice')
    d = doc()
    d['not_measured'] = []
    cfg, warn = check(d)
    assert not any('PLACEHOLDER' in w for w in warn)


def test_every_problem_listed_at_once():
    d = doc()
    d['vehicle']['lf'] = 160.0
    del d['timing']['imu_gap_s']
    d['lidar']['scan_mode'] = 'Turbo'
    msg = refused(d, 'vehicle.lf', 'timing.imu_gap_s: missing', 'scan_mode')
    assert '3 problem(s)' in msg


def test_missing_section_refused():
    d = doc()
    del d['lidar']
    refused(d, 'lidar: missing section')


def test_risky_choices_warned():
    d = doc()
    d['estimator']['imu_gap_predict'] = True
    d['estimator']['slip_use_observer'] = True
    d['estimator']['map_check'] = False
    _, warn = check(d)
    text = ' | '.join(warn)
    assert 'contract v0.4' in text and 'slip_r0' in text and 'map_check is OFF' in text


# ---------------------------------------------------------------- car_nodes
def test_laser_numbers_come_only_from_the_mounts():
    cfg, _ = load_car_config(CAR_YAML)
    specs = lidar_specs(cfg, MOUNTS) + loc_specs(cfg)
    yaw = math.radians(180.0)
    seen = 0
    for node, k, v in all_params(specs):
        if k in ('laser_x', 'laser_y', 'laser_yaw'):
            want = {'laser_x': 0.12, 'laser_y': -0.01, 'laser_yaw': yaw}[k]
            assert abs(v - want) < 1e-12, (node, k, v)
            seen += 1
    assert seen == 9, seen                      # scan_deskew + ekf_local + ekf_global, 3 numbers each
    frames = [s for s in specs if s['name'] == 'sensor_frames'][0]['params']
    assert frames['mounts_file'] == MOUNTS['_path'] and frames['prefixes'] == 'est/' and frames['publish_laser']


def test_lidar_chain():
    cfg, _ = load_car_config(CAR_YAML)
    s = {x['name']: x['params'] for x in lidar_specs(cfg, MOUNTS)}
    drv, dsk = s['rplidar_node'], s['scan_deskew']
    assert drv['frame_id'] == 'est/laser' and drv['topic_name'] == '/scan_a1_raw' and drv['angle_compensate']
    assert drv['serial_baudrate'] == 115200 and drv['scan_mode'] == 'Sensitivity'
    assert dsk['in_topic'] == '/scan_a1_raw' and dsk['out_topic'] == '/scan_a1'
    assert dsk['odom_topic'] == '/odom/ekf_local' and dsk['time_order'] == 'reverse'


def test_ekfs_get_the_car_numbers_and_car_calibrations():
    cfg, _ = load_car_config(CAR_YAML)
    m = [s for s in loc_specs(cfg) if 'merged' in s][0]['merged']
    for name in ('ekf_local', 'ekf_global'):
        p = m[name]
        assert p['lf'] == cfg['lf'] and p['lr'] == cfg['lr'] and p['beta_model'] == 'steer'
        assert p['wheel_delay_s'] == -1.0 and p['imu_gap_predict'] is False and p['slip_use_observer'] is False
        assert p['allow_test_faults'] is False and p['sim_wheel_ideal_topic'] == ''
        assert p['scan_topic'] == '/scan_a1' and p['child_frame_id'] == 'est/base_link'
        for k in ('scale_file', 'scale_learned_file', 'steer_calib_file', 'slip_file'):
            assert os.path.dirname(p[k]) == cfg['calibration_dir'], (name, k, p[k])
    assert m['ekf_local']['publish_tf'] is True and m['ekf_global']['publish_tf'] is False
    g = m['ekf_global']
    assert g['use_amcl'] and abs(g['amcl_offset_m'] - 0.025) < 1e-12 and g['map_check'] is True
    assert (g['x0'], g['y0'], g['yaw0']) == (1.0, 2.0, 0.5)
    se = m['state_estimate']
    assert se['track'] == 'ourtrack' and se['lf'] == cfg['lf'] and se['steer_calib_file'] == g['steer_calib_file']


def test_amcl_start_pose_shifted_by_half_a_cell():
    cfg, _ = load_car_config(CAR_YAML)
    a = [s for s in loc_specs(cfg) if s['name'] == 'amcl'][0]
    assert a['params_file'] == '/c/amcl_car.yaml' and ('/map', '/map_amcl') in a['remap']
    assert abs(a['params']['initial_pose.x'] - 0.975) < 1e-12 and abs(a['params']['initial_pose.y'] - 1.975) < 1e-12
    assert a['params']['scan_topic'] == '/scan_a1' and a['params']['base_frame_id'] == 'est/base_link'


def test_shipped_specs_are_safe():
    cfg, _ = load_car_config(CAR_YAML)
    assert safety_problems(lidar_specs(cfg, MOUNTS) + loc_specs(cfg)) == []


def test_safety_check_catches_unsafe_specs():
    cfg, _ = load_car_config(CAR_YAML)
    base = lidar_specs(cfg, MOUNTS) + loc_specs(cfg)
    cases = [('ekf_global', 'allow_test_faults', True), ('ekf_local', 'sim_wheel_ideal_topic', '/sim/wheel_ideal'),
             ('ekf_global', 'test_kick', '0.5,0,0'), ('ekf_global', 'scale_file', '~/.ros/ekf_speed_scale.yaml'),
             ('state_estimate', 'steer_calib_file', os.path.expanduser('~/.ros/steering_calibration.yaml'))]
    for node, k, v in cases:
        specs = copy.deepcopy(base)
        [s for s in specs if 'merged' in s][0]['merged'][node][k] = v
        assert safety_problems(specs), (node, k, v)
    specs = copy.deepcopy(base) + [{'package': 'state_estimation', 'executable': 'fake_imu', 'name': 'fake_imu',
                                    'params': {}}]
    assert any('fake_imu' in p for p in safety_problems(specs))


def test_ekf_common_does_not_share_dicts():
    cfg, _ = load_car_config(CAR_YAML)
    m = [s for s in loc_specs(cfg) if 'merged' in s][0]['merged']
    m['ekf_local']['lf'] = 99.0
    assert m['ekf_global']['lf'] == cfg['lf'] and ekf_common(cfg, MOUNTS)['lf'] == cfg['lf']


def test_map_resolution(tmp=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'm.yaml')
        for r, ok in ((0.05, True), (5.0, False)):
            with open(p, 'w') as f:
                yaml.safe_dump({'image': 'm.pgm', 'resolution': r, 'origin': [0, 0, 0]}, f)
            try:
                assert map_resolution(p) == r and ok
            except ValueError:
                assert not ok


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS  {name}')
        except Exception as ex:                       # noqa: BLE001
            fails += 1
            print(f'FAIL  {name}: {type(ex).__name__}: {ex}')
    print(f'\n{len(tests) - fails} of {len(tests)} passed')
    sys.exit(1 if fails else 0)
