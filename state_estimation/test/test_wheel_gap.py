#!/usr/bin/env python3
"""Tests for wheel_gap.py (WheelGap: wheel-data dropout detection + the uncertainty model, Fix 4, 2026-10-30) and for
the EKF covariance it produces (LocalizationEKF.predict with WheelGap.sigma_v_step). No ROS. Run: python3
test_wheel_gap.py   (or pytest)"""
import math
import os
import sys

import numpy as np

try:
    from state_estimation.ekf_node import LocalizationEKF
    from state_estimation.wheel_gap import GAP, OK, WheelGap
except ImportError:                                  # run from the source tree without ROS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'state_estimation'))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from ekf_node import LocalizationEKF
    from wheel_gap import GAP, OK, WheelGap


def feed(g, T, drop=None, imu_rx=None, wheel_rx=None):
    """IMU 100 Hz, wheel 50 Hz (5 ms offset), no wheel samples inside drop = (t0, t1). rx functions give receive times
    (None = no receive times). Returns [(t, kind, state)] in receive order."""
    ev = []
    for i in range(1, int(round(T * 1000)) + 1):
        t = i / 1000.0
        if i % 10 == 0:
            ev.append(((imu_rx or (lambda x: x))(t), 'imu', t))
        if i % 20 == 5 and not (drop and drop[0] <= t < drop[1]):
            ev.append(((wheel_rx or (lambda x: x))(t), 'wheel', t))
    out = []
    for rx, kind, t in sorted(ev):
        r = rx if (imu_rx or wheel_rx) else None
        if kind == 'imu':
            g.imu(t, rx=r)
        else:
            g.wheel(t, rx=r)
        out.append((t, kind, g.state))
    return out


def test_no_gap_before_the_first_wheel_sample():
    g = WheelGap()
    for i in range(1, 200):
        assert not g.imu(i * 0.01)
    assert g.state == OK and g.episodes == 0


def test_detected_after_gap_s():
    g = WheelGap(gap_s=0.05)
    g.wheel(1.000)
    assert not g.imu(1.040) and g.state == OK         # 40 ms: a late sample, not a dropout
    assert not g.imu(1.0499) and g.state == OK        # just under gap_s: still OK (exactly 1.050 - 1.000 is 0.05000000000000004)
    assert g.imu(1.060) and g.state == GAP and g.lost # 60 ms: dropout
    assert not g.imu(1.070) and g.episodes == 1        # the same gap: counted once
    ev = g.pop_events()
    assert ev == [('lost', 1.060, 1.000)], ev
    assert abs(g.gap_len(1.5) - 0.5) < 1e-12


def test_dropout_in_a_stream():
    g = WheelGap()
    rows = feed(g, 3.0, drop=(1.0, 2.0))
    first = next(t for t, k, s in rows if s == GAP)
    assert 0.98 + 0.05 < first <= 0.985 + 0.05 + 0.011, f'detected at {first:.3f} s'   # last wheel 0.985 s
    back = next(t for t, k, s in rows if t > first and s == OK)
    assert abs(back - 2.005) < 1e-9, back
    assert g.episodes == 1 and abs(g.longest - (2.005 - 0.985)) < 1e-9


def test_clean_stream_never_a_gap():
    g = WheelGap()
    feed(g, 5.0)
    assert g.episodes == 0


def test_backlog_is_not_a_dropout():
    """Processing backlog: the IMU messages are handled 0.2 s late (stale) -> no dropout, though the stamps say so."""
    g = WheelGap()
    feed(g, 2.0, imu_rx=lambda t: t + 0.2 if 1.0 <= t < 1.4 else t + 0.002, wheel_rx=lambda t: t + 0.002)
    assert g.episodes == 0, f'{g.episodes} false dropouts in a backlog'


def test_wheel_handled_recently_is_not_a_dropout():
    """Stamps old but the wheel message was HANDLED just now (it arrived late) -> not a dropout."""
    g = WheelGap()
    g.wheel(1.0, rx=1.10)                             # stamp 1.0, received 1.10 (a late message)
    assert not g.imu(1.10, rx=1.101)                  # fresh IMU, but a wheel message was received 1 ms ago
    assert g.imu(1.20, rx=1.201)                      # 100 ms later nothing new: dropout


def test_real_dropout_with_receive_times():
    g = WheelGap()
    feed(g, 3.0, drop=(1.0, 2.0), imu_rx=lambda t: t + 0.002, wheel_rx=lambda t: t + 0.002)
    assert g.episodes == 1


def test_back_event():
    g = WheelGap()
    g.wheel(1.0)
    g.imu(1.1)
    assert g.wheel(1.5) and g.state == OK
    ev = g.pop_events()
    assert ev[-1][0] == 'back' and abs(ev[-1][2] - 0.5) < 1e-12
    assert not g.wheel(1.52)                           # an ordinary sample


def test_sigma_d():
    g = WheelGap(sv0=0.14, sa=0.1)
    assert g.sigma_d(0.0) == 0.0
    assert abs(g.sigma_d(1.0) - 0.19) < 1e-12
    assert abs(g.sigma_d(2.0) - (0.28 + 0.2)) < 1e-12


def test_sigma_v_step_accumulates_to_sigma_d():
    """The per-step speed noise adds (sv * dt)^2 per step -> the sum over a gap = sigma_d(L)^2."""
    g = WheelGap()
    g.wheel(0.0)
    g.imu(0.06)
    var, t, dt = 0.0, 0.0, 0.01
    while t < 1.5 - 1e-9:
        t += dt
        if t <= 0.06 - 1e-9:
            continue
        var += (g.sigma_v_step(t, dt) * dt) ** 2
    var += g.sigma_d(0.06) ** 2                        # the part before detection (the first step covers it)
    assert abs(math.sqrt(var) - g.sigma_d(1.5)) < 0.005, (math.sqrt(var), g.sigma_d(1.5))


def test_sigma_zero_while_ok():
    g = WheelGap()
    g.wheel(1.0)
    assert g.gap_len(1.02) == 0.0 and g.sigma_v_step(1.02, 0.01) == 0.0


def test_ekf_covariance_grows_like_sigma_d():
    """The EKF predicting with sigma_v_step: its position uncertainty ALONG the heading grows by sigma_d(L)."""
    yaw = 0.3
    e = LocalizationEKF(x0=[0.0, 0.0, yaw, 0.0, 0.97], P0_diag=[0.03, 0.03, math.radians(0.5), 0.001, 0.005],
                        sigma_v=0.05, sigma_w=0.005, sigma_bw=0.0001, sigma_kw=0.0005)
    u = np.array([math.cos(yaw), math.sin(yaw)])
    var0 = float(u @ e.P[:2, :2] @ u)
    g = WheelGap()
    g.wheel(0.0)
    g.imu(0.06)
    t, dt = 0.0, 0.01
    while t < 1.0 - 1e-9:
        t += dt
        sv = g.sigma_v_step(t, dt) if t > 0.06 - 1e-9 else 0.0
        e.predict(4.0, 0.0, 0.0, dt, use_k=False, sigma_v=sv)
    grown = math.sqrt(max(float(u @ e.P[:2, :2] @ u) - var0, 0.0))
    assert abs(grown - g.sigma_d(1.0)) < 0.01, f'grew {grown:.3f} m, sigma_d {g.sigma_d(1.0):.3f} m'


def test_bad_parameters():
    for kw in ({'gap_s': 0.0}, {'sv0': -0.1}, {'sa': -0.1}, {'sv0': 0.0, 'sa': 0.0}):
        try:
            WheelGap(**kw)
        except ValueError:
            continue
        raise AssertionError(f'no ValueError for {kw}')


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
