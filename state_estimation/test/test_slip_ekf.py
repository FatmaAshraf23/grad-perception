#!/usr/bin/env python3
"""Tests for LocalizationEKF.predict(use_k=False, sigma_v=...) -- the prediction with the IMU speed while the wheel
slips (ekf_node docstring UPDATE 3, Fix 3, 2026-10-26). numpy only. Run: python3 test_slip_ekf.py   (or pytest)"""
import math
import os
import sys

import numpy as np

try:
    from state_estimation.ekf_node import LocalizationEKF
except ImportError:                                  # run from the source tree without ROS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'state_estimation'))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from ekf_node import LocalizationEKF

K = 0.97                                             # the simulator's learned wheel scale


def ekf(yaw=0.3):
    return LocalizationEKF(x0=[0.0, 0.0, yaw, 0.0, K], P0_diag=[0.1, 0.1, math.radians(2.0), 0.05, 0.05],
                           sigma_v=0.05, sigma_w=0.005, sigma_bw=0.0001, sigma_kw=0.0005)


def test_use_k_true_is_the_old_prediction():
    """Default call: unchanged -- the car moves k * v_meas, F has the k column."""
    e = ekf()
    e.predict(4.0, 0.0, 0.0, 0.01)
    assert abs(math.hypot(e.x[0], e.x[1]) - K * 4.0 * 0.01) < 1e-12
    assert abs(e.P[0, 4]) > 1e-6, 'position-scale correlation missing in the normal prediction'


def test_use_k_false_moves_the_true_speed():
    """While slipping: v_meas is already true m/s (the IMU speed) -> moves exactly v * dt, the scale k untouched."""
    e = ekf()
    for _ in range(100):
        e.predict(4.0, 0.0, 0.0, 0.01, use_k=False)
    assert abs(math.hypot(e.x[0], e.x[1]) - 4.0) < 1e-9, f'moved {math.hypot(e.x[0], e.x[1]):.4f} m (4.0 expected)'
    assert e.x[4] == K, 'k changed by the prediction'


def test_use_k_false_no_scale_correlation():
    """No k column in F: the prediction builds no position-scale correlation, so an AMCL fix during a slip cannot
    'learn' a wrong wheel scale through it."""
    e = ekf()
    for _ in range(100):
        e.predict(4.0, 0.0, 0.0, 0.01, use_k=False)
    assert np.all(np.abs(e.P[0:2, 4]) < 1e-15), f'P[x/y, k] = {e.P[0:2, 4]}'
    e2 = ekf()
    for _ in range(100):
        e2.predict(4.0 / K, 0.0, 0.0, 0.01)
    assert np.all(np.abs(e2.P[0:2, 4]) > 1e-4), 'normal prediction should build the correlation'


def test_sigma_v_override_grows_the_uncertainty():
    """sigma_v 0.15 instead of 0.05: the along-track variance grows faster (the IMU speed is less certain)."""
    a, b = ekf(yaw=0.0), ekf(yaw=0.0)
    for _ in range(100):
        a.predict(4.0, 0.0, 0.0, 0.01, use_k=False)
        b.predict(4.0, 0.0, 0.0, 0.01, use_k=False, sigma_v=0.15)
    da, db = a.P[0, 0] - 0.01, b.P[0, 0] - 0.01
    assert db > 5 * da, f'variance growth {db:.2e} vs {da:.2e}'
    assert a.sigma_v == 0.05 and b.sigma_v == 0.05, 'the override must not change the stored sigma_v'


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
