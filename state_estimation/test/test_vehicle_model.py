#!/usr/bin/env python3
"""Tests for vehicle_model.py (sideslip shared by ekf_node and state_estimate), 2026-10-18.
Run: python3 test_vehicle_model.py   (or pytest)"""
import math
import os
import sys
import tempfile

try:
    from state_estimation.vehicle_model import Sideslip, load_slip_file, sideslip_from_params
except ImportError:                                  # run from the source tree without ROS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'state_estimation'))
    from vehicle_model import Sideslip, load_slip_file, sideslip_from_params

LF, LR = 0.15875, 0.17145


def test_steer_model_is_the_old_formula():
    s = Sideslip('steer', LF, LR)
    for d in (-0.4, -0.1, 0.0, 0.05, 0.3):
        assert abs(s.beta(3.0, 1.0, d) - math.atan(LR / (LF + LR) * math.tan(d))) < 1e-12
    assert s.kinematic


def test_slip_model_formula_and_limits():
    s = Sideslip('slip', LF, LR, slip_k=0.017)
    for vx, r in ((2.0, 1.8), (6.0, 0.5), (1.0, -1.0)):
        assert abs(s.vy(vx, r) - r * (LR - 0.017 * vx * vx)) < 1e-12
    assert not s.kinematic
    s0 = Sideslip('slip', LF, LR, slip_k=0.0)
    assert abs(s0.vy(2.0, 1.0) - LR * 1.0) < 1e-12 and s0.kinematic    # no slip: rear axle has vy 0
    assert s.beta(0.01, 1.0) == 0.0                                      # standing: no division by ~0


def test_reversing_keeps_the_velocity_direction():
    """beta stays within +-90 deg: v * cos(yaw + beta) must still point backwards for vx < 0"""
    s = Sideslip('slip', LF, LR, slip_k=0.0)
    b = s.beta(-1.0, 0.5)
    assert abs(b) < math.pi / 2
    vel_x = -1.0 * math.cos(0.0 + b)                     # yaw 0, v = -1 m/s
    assert vel_x < 0
    assert abs(s.vy(-1.0, 0.5) - LR * 0.5) < 1e-12       # rear axle has no sideways velocity backwards too


def test_bad_values_are_refused():
    for bad in (dict(model='dynamic'), dict(model='slip', slip_k=-0.1), dict(model='slip', slip_k=1.0)):
        try:
            Sideslip(bad.get('model'), LF, LR, bad.get('slip_k', 0.0))
        except ValueError:
            continue
        raise AssertionError(f'accepted {bad}')


def test_slip_file_and_params():
    with tempfile.TemporaryDirectory() as d:
        missing = os.path.join(d, 'none.yaml')
        k, sig, problem = load_slip_file(missing)
        assert k is None and 'no slip calibration file' in problem
        s, msg, warn = sideslip_from_params('slip', LF, LR, -1.0, missing)
        assert s.k == 0.0 and warn and 'K = 0' in msg
        good = os.path.join(d, 'slip.yaml')
        with open(good, 'w') as f:
            f.write('# written by slip_calib\nslip_k: 0.0185\nslip_k_sigma: 0.002\n')
        k, sig, problem = load_slip_file(good)
        assert (k, sig, problem) == (0.0185, 0.002, None)
        s, msg, warn = sideslip_from_params('slip', LF, LR, -1.0, good)
        assert s.k == 0.0185 and not warn
        s, msg, warn = sideslip_from_params('slip', LF, LR, 0.017, good)     # explicit value wins
        assert s.k == 0.017 and not warn
        bad = os.path.join(d, 'bad.yaml')
        with open(bad, 'w') as f:
            f.write('slip_k: 3\n')
        assert load_slip_file(bad)[0] is None
        s, msg, warn = sideslip_from_params('steer', LF, LR, -1.0, missing)  # steer ignores the file
        assert s.model == 'steer' and not warn


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
