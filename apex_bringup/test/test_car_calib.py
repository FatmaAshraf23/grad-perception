#!/usr/bin/env python3
"""Tests for car_calib.py: the car's calibration tools read and write ONLY the car's calibration folder and use the
car's numbers. No ROS needed. Run: python3 test_car_calib.py   (or pytest)"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from apex_bringup.car_calib import find_slip_calib, slip_calib_argv, steer_calib_spec, wheel_calib_spec  # noqa: E402
from apex_bringup.car_config import load_car_config  # noqa: E402
from apex_bringup.car_nodes import safety_problems  # noqa: E402

CAR_YAML = os.path.join(HERE, '..', 'config', 'car.yaml')


def cfg_with(**kw):
    cfg, _ = load_car_config(CAR_YAML)
    cfg = dict(cfg)
    cfg.update(kw)
    return cfg


def in_dir(path, d):
    return os.path.dirname(path) == d


def test_wheel_writes_into_the_car_folder():
    cfg = cfg_with(calibration_dir='/home/x/.ros/apex_car')
    p = wheel_calib_spec(cfg, 10.0)['params']
    assert p['true_distance'] == 10.0 and p['use_sim_truth'] is False and p['write_file'] is True
    assert p['scale_file'] == '/home/x/.ros/apex_car/ekf_speed_scale.yaml'


def test_wheel_needs_a_tape_distance():
    for d in (0.0, -1.0, None, ''):
        try:
            wheel_calib_spec(cfg_with(), d)
        except ValueError as e:
            assert 'true_distance' in str(e)
            continue
        raise AssertionError(f'accepted {d!r}')


def test_steer_uses_the_car_wheelbase_and_folder():
    cfg = cfg_with(lf=0.14, lr=0.19, calibration_dir='/c')
    p = steer_calib_spec(cfg)['params']
    assert (p['lf'], p['lr']) == (0.14, 0.19) and p['use_sim_truth'] is False
    assert p['out_file'] == '/c/steering_calibration.yaml' and p['scale_file'] == '/c/ekf_speed_scale.yaml'


def test_steer_reads_the_scale_the_ekf_reads():
    """steer_calib's speed anchor must be the SAME file the EKFs start from (car_nodes.calib_files)."""
    from apex_bringup.car_nodes import calib_files
    cfg = cfg_with(calibration_dir='/c')
    assert steer_calib_spec(cfg)['params']['scale_file'] == calib_files(cfg)['scale_file']
    assert wheel_calib_spec(cfg, 5.0)['params']['scale_file'] == calib_files(cfg)['scale_file']


def test_slip_argv_has_the_car_numbers():
    cfg = cfg_with(lf=0.14, lr=0.19, wheel_delay_s=0.012, calibration_dir='/c')
    a = slip_calib_argv(cfg, ['/bags/b1', '/bags/b2'], 0.1)
    assert a[:2] == ['/bags/b1', '/bags/b2']
    kv = dict(zip(a[2::2], a[3::2]))
    assert float(kv['--lf']) == 0.14 and float(kv['--lr']) == 0.19 and float(kv['--wheel-delay']) == 0.012
    assert float(kv['--amcl-offset']) == 0.05   # half of a 0.1 m cell
    assert kv['--scale-file'] == '/c/ekf_speed_scale.yaml' and kv['--write'] == '/c/vehicle_slip.yaml'


def test_shipped_specs_pass_the_safety_check():
    cfg, _ = load_car_config(CAR_YAML)
    assert safety_problems([wheel_calib_spec(cfg, 10.0), steer_calib_spec(cfg)]) == []


def test_simulator_folder_would_be_caught():
    sim = os.path.expanduser('~/.ros')
    cfg = cfg_with(calibration_dir=sim)          # (car_config already refuses this; second line of defence)
    assert safety_problems([wheel_calib_spec(cfg, 10.0)])
    assert safety_problems([steer_calib_spec(cfg)])


def test_find_slip_calib():
    with tempfile.TemporaryDirectory() as ws:
        prefix = os.path.join(ws, 'install', 'state_estimation')
        os.makedirs(prefix)
        old = os.environ.pop('APEX_SLIP_CALIB', None)
        try:
            assert find_slip_calib(prefix) is None
            tool = os.path.join(ws, 'src', 'state_estimation', 'tools', 'slip_calib.py')
            os.makedirs(os.path.dirname(tool))
            open(tool, 'w').close()
            assert find_slip_calib(prefix) == tool
        finally:
            if old is not None:
                os.environ['APEX_SLIP_CALIB'] = old


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
