#!/usr/bin/env python3
"""Tests for the global EKF's AMCL gate (ekf_node.amcl_pose_class) and measurement noise
(ekf_node.amcl_measurement_noise), 2026-10-19 -- task 3(a): corridor poses at racing speed.
Run: python3 test_amcl_gate.py   (or pytest)"""
import math
import os
import sys

import numpy as np

try:
    from state_estimation.ekf_node import LocalizationEKF, amcl_measurement_noise, amcl_pose_class
except ImportError:                                  # run from the source tree without ROS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'state_estimation'))
    from ekf_node import LocalizationEKF, amcl_measurement_noise, amcl_pose_class

MAX, CORR = 0.35, 0.10                               # node defaults amcl_max_sigma, amcl_corridor_sigma


def ellipse(sig_long, sig_narrow, angle_deg):
    """x-y covariance of an ellipse with these axis sigmas, long axis at angle_deg from the map's x"""
    a = math.radians(angle_deg)
    V = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return V @ np.diag([sig_long ** 2, sig_narrow ** 2]) @ V.T


def cov6(c2, cyaw=math.radians(1.0) ** 2):
    c = np.zeros((6, 6))
    c[:2, :2] = c2
    c[5, 5] = cyaw
    return c.ravel()


def old_noise(cov, scale, min_pos, min_yaw):
    """the function before 2026-10-19 (floor on the map's x / y diagonal)"""
    c = np.asarray(cov, dtype=float).reshape(6, 6)
    R = c[np.ix_([0, 1, 5], [0, 1, 5])] * scale
    R[0, 0] = max(R[0, 0], min_pos ** 2)
    R[1, 1] = max(R[1, 1], min_pos ** 2)
    R[2, 2] = max(R[2, 2], min_yaw ** 2)
    return 0.5 * (R + R.T)


def cls(c2, corridor=CORR):
    return amcl_pose_class(c2[0, 0], c2[0, 1], c2[1, 1], MAX, corridor)


def test_round_clouds_follow_the_old_rule():
    """a round cloud (how a LOST AMCL looks) is judged exactly as before: <= 0.35 total -> confident, else unsure"""
    assert cls(ellipse(0.20, 0.20, 0)) == 'confident'          # total 0.28
    assert cls(ellipse(0.26, 0.26, 0)) == 'unsure'             # total 0.37
    assert cls(ellipse(0.50, 0.50, 30)) == 'unsure'


def test_corridor_pose_is_used_at_any_angle():
    """closed1's straight at 6 m/s: along ~40 cm, across ~5.5 cm -> 'corridor' (was thrown away)"""
    for ang in (0, 30, 90, 135):
        assert cls(ellipse(0.40, 0.055, ang)) == 'corridor', ang
    assert cls(ellipse(0.40, 0.055, 0), corridor=0.0) == 'unsure'   # 0 = off = the old rule


def test_wide_in_both_directions_stays_unsure():
    """narrow axis above 10 cm -> unsure, however long the ellipse (lost AMCL: narrow 12-150 cm)"""
    assert cls(ellipse(0.50, 0.15, 20)) == 'unsure'
    assert cls(ellipse(2.00, 0.12, 0)) == 'unsure'
    assert cls(ellipse(2.00, 0.09, 0)) == 'corridor'


def test_floor_holds_in_every_direction():
    """long thin ellipse at 30 deg, narrow sigma 3 cm (x2 = 6 cm): the new floor keeps every axis >= 10 cm,
    the old diagonal floor let 6 cm through; axis-aligned: identical to the old function"""
    c = cov6(ellipse(0.40, 0.03, 30))
    narrow_new = math.sqrt(np.linalg.eigvalsh(amcl_measurement_noise(c, 4.0, 0.10, math.radians(1.0))[:2, :2]).min())
    narrow_old = math.sqrt(np.linalg.eigvalsh(old_noise(c, 4.0, 0.10, math.radians(1.0))[:2, :2]).min())
    assert narrow_new >= 0.10 - 1e-9, narrow_new
    assert narrow_old < 0.07, narrow_old
    for c2 in (ellipse(0.40, 0.03, 0), ellipse(0.02, 0.02, 0), ellipse(0.10, 0.04, 90)):
        a = amcl_measurement_noise(cov6(c2), 4.0, 0.10, math.radians(1.0))
        b = old_noise(cov6(c2), 4.0, 0.10, math.radians(1.0))
        assert np.allclose(a, b, atol=1e-12), (a, b)


def test_corridor_fix_corrects_across_not_along():
    """EKF sigma 3.8 cm (6 m/s run); corridor fix along x (along 40 cm, across 5.5 cm), AMCL 30 cm ahead
    and 10 cm to the left: the EKF moves ~1 cm sideways (gain ~0.1) and < 1 mm along (gain ~0.002)"""
    ekf = LocalizationEKF(x0=[0.0, 0.0, 0.0, 0.0, 1.0], P0_diag=[0.038, 0.038, math.radians(0.5), 0.01, 0.01],
                          sigma_v=0.05, sigma_w=0.005, sigma_bw=0.0001, sigma_kw=0.0005)
    R = amcl_measurement_noise(cov6(ellipse(0.40, 0.055, 0)), 4.0, 0.10, math.radians(1.0))
    nis = ekf.pose_nis(np.array([0.30, 0.10, 0.0]), R)
    assert nis < 11.34, nis                                    # passes the NIS gate
    _, dx = ekf.pose_update(np.array([0.30, 0.10, 0.0]), R)
    assert abs(dx[0]) < 0.001, dx[0]                           # along: 0.30 m innovation -> < 1 mm
    assert 0.008 < dx[1] < 0.013, dx[1]                        # across: 0.10 m innovation -> ~1.07 cm


def test_far_off_corridor_pose_fails_the_nis_gate():
    """a corridor pose 1 m off ACROSS the track is not believed (NIS gate), even though its covariance is thin"""
    ekf = LocalizationEKF(x0=[0.0, 0.0, 0.0, 0.0, 1.0], P0_diag=[0.038, 0.038, math.radians(0.5), 0.01, 0.01],
                          sigma_v=0.05, sigma_w=0.005, sigma_bw=0.0001, sigma_kw=0.0005)
    R = amcl_measurement_noise(cov6(ellipse(0.40, 0.055, 0)), 4.0, 0.10, math.radians(1.0))
    assert ekf.pose_nis(np.array([0.0, 1.0, 0.0]), R) > 11.34


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
