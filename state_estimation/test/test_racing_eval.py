#!/usr/bin/env python3
"""Tests for the racing-speed evaluation (2026-10-18): eval_logger's truth interpolation and the
test_driver's physics speed profile.  Run:  python3 test_racing_eval.py   (or pytest)

Why these tests: at 6 m/s the old "nearest truth message" was up to ~5 cm off, which is as large
as the errors we measure; and the old "corner speed 2 m ahead" profile asked for ~8-9 m/s^2 of
braking.  The tests pin down the new behaviour with synthetic data whose answer we know."""
import math
import os
import sys
from collections import deque

import numpy as np

from state_estimation import eval_logger as EL
from state_estimation import test_driver as TD


def _logger():
    lg = object.__new__(EL.EvalLogger)          # only the truth helpers are used (no ROS node)
    lg.states = deque(maxlen=3000)
    lg.truth_hist = deque(maxlen=1500)
    lg.last_msg_ns = None
    return lg


def _drive(lg, v=6.0, wz=0.8, seconds=4.0, seed=0):
    """simulator-like stream: a new physics state every 10 ms (+-0.3 ms), messages every 4 ms"""
    rng = np.random.default_rng(seed)
    phys = [k * 10e6 + rng.normal(0, 0.3e6) for k in range(int(seconds * 100))]
    k, prev = 0, None
    for tp in np.arange(0, seconds * 1e9, 4e6) + 1.3e6:
        while k + 1 < len(phys) and phys[k + 1] <= tp:
            k += 1
        if phys[k] > tp:
            continue
        msg = (int(tp), v * phys[k] * 1e-9, 0.0, wz * phys[k] * 1e-9, v, 0.0, wz)
        if k != prev:
            lg.states.append(msg)
        lg.truth_hist.append(msg)
        lg.last_msg_ns = int(tp)
        prev = k


def test_interpolation_is_exact_and_known_until():
    lg = _logger()
    _drive(lg)
    lim = lg.known_until()
    assert lim == lg.states[-1][0] - 1                       # moving: just before the newest state
    fp = np.array([s[0] for s in lg.states])
    fx = np.array([s[1] for s in lg.states])
    rng = np.random.default_rng(1)
    for t in rng.uniform(fp[5], lim, 500).astype(np.int64):    # stamps are whole nanoseconds
        (x, *_), kind = lg.truth_interp(int(t))
        i = np.searchsorted(fp, t, side='right') - 1
        w = min(fp[i + 1] - fp[i], 10e6)
        f = min(max((t - (fp[i + 1] - w)) / w, 0.0), 1.0)
        assert abs(x - (fx[i] + f * (fx[i + 1] - fx[i]))) < 1e-9
        assert kind in ('interp', 'hold')


def test_old_nearest_message_method_was_cm_off_at_6_mps():
    lg = _logger()
    _drive(lg)
    fp = np.array([s[0] for s in lg.states])
    rng = np.random.default_rng(2)
    d = [abs(lg.truth_nearest(int(t))[0] - lg.truth_interp(int(t))[0][0])
         for t in rng.uniform(fp[5], lg.known_until(), 500)]
    assert np.percentile(d, 95) > 0.02                        # > 2 cm: why we interpolate


def test_hold_while_standing_and_at_start_up():
    lg = _logger()
    st = (0, 1.0, 2.0, 0.3, 0.0, 0.0, 0.0)
    lg.states.append(st)
    for tp in np.arange(0, 2.0e9, 4e6):
        lg.truth_hist.append((int(tp),) + st[1:])
        lg.last_msg_ns = int(tp)
    assert lg.known_until() == lg.last_msg_ns - EL.PUB_NS
    (x, y, *_), kind = lg.truth_interp(int(1.5e9))
    assert (x, y, kind) == (1.0, 2.0, 'hold')
    lg.states.append((int(2.0e9), 1.06, 2.0, 0.3, 6.0, 0.0, 0.0))
    lg.last_msg_ns = int(2.0e9)
    assert lg.truth_interp(int(1.98e9))[1] == 'hold'           # still standing 20 ms before
    (x, *_), kind = lg.truth_interp(int(1.995e9))             # moving during the last 10 ms
    assert kind == 'interp' and abs(x - 1.03) < 1e-9


def test_yaw_interpolation_across_pi():
    s0 = (0, 0.0, 0.0, math.pi - 0.01, 1.0, 0.0, 0.0)
    s1 = (10_000_000, 1.0, 0.0, -math.pi + 0.01, 1.0, 0.0, 0.0)
    (_, _, yaw, *_), _ = EL.interp_states(s0, s1, 5_000_000)
    assert abs(abs(yaw) - math.pi) < 1e-9


def _centerline():
    from ament_index_python.packages import get_package_share_directory
    p = os.path.join(get_package_share_directory('apex_track'), 'tracks', 'closed1', 'centerline.csv')
    wp = TD.load_waypoints(p)
    return wp, float(np.mean(np.linalg.norm(np.diff(wp, axis=0), axis=1)))


def test_physics_profile_respects_its_limits():
    wp, sp = _centerline()
    k = TD.loop_curvature(wp)
    for v_max, a_lat, a_brake, a_accel in [(3.0, 4.0, 5.0, 4.0), (6.0, 4.0, 5.0, 4.0), (6.0, 6.0, 3.0, 2.0)]:
        v = TD.speed_profile_physics(wp, v_max, 0.7, a_lat, a_brake, a_accel, sp)
        dec = (v ** 2 - np.roll(v, -1) ** 2) / (2 * sp)
        acc = (np.roll(v, -1) ** 2 - v ** 2) / (2 * sp)
        assert v.max() <= v_max + 1e-9 and v.min() >= 0.7 - 1e-9
        assert dec.max() <= a_brake + 1e-6 and acc.max() <= a_accel + 1e-6
        assert np.all(v <= np.maximum(np.sqrt(a_lat / np.maximum(k, 1e-6)), 0.7) + 1e-9)


def test_default_profile_is_the_old_one():
    """a_brake = 0 keeps the 2026-09 profile: corner speed min over the next 2 m"""
    wp, sp = _centerline()
    kappa = TD.loop_curvature(wp)
    v_curve = np.clip(np.sqrt(1.5 / np.maximum(kappa, 1e-6)), 0.7, 1.5)
    n, k = len(wp), max(1, int(2.0 / sp))
    old = np.array([min(v_curve[(i + j) % n] for j in range(k)) for i in range(n)])
    assert np.array_equal(TD.speed_profile(wp, 1.5, 0.7, 1.5, 2.0, sp), old)


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS  {name}')
        except Exception as ex:                               # noqa: BLE001
            fails += 1
            print(f'FAIL  {name}: {type(ex).__name__}: {ex}')
    print(f'\n{len(tests) - fails} of {len(tests)} passed')
    sys.exit(1 if fails else 0)
