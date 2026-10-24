#!/usr/bin/env python3
"""Tests for map_check.py (fast along-track shift search 2026-10-21; MismatchAlarm = contract v0.3, 2026-10-24).
Run: python3 test_map_check.py   (or pytest)
Synthetic track: a straight corridor |y| <= 1 m closed by end walls at x = +-8 m, scans computed
EXACTLY (ray vs the 4 wall lines), so the true answer of every check is known."""
import math
import os
import sys
import time

import numpy as np

try:
    from state_estimation.map_check import (BAD_ALONG, BAD_FIT, CLEAN, NEITHER, NO_RESULT, CpuBudget, DistanceField,
                                            MismatchAlarm, ShiftMonitor, ShiftResult, _distance_cells, make_shifts,
                                            shift_search)
except ImportError:                                  # run from the source tree without ROS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'state_estimation'))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from map_check import (BAD_ALONG, BAD_FIT, CLEAN, NEITHER, NO_RESULT, CpuBudget, DistanceField, MismatchAlarm,
                           ShiftMonitor, ShiftResult, _distance_cells, make_shifts, shift_search)

RES, OX, OY = 0.05, -10.0, -2.0
LASER = (0.275, 0.0, 0.0)
SHIFTS = make_shifts(1.0, 0.05)


def corridor_field(end_x=8.0, half_w=1.0):
    h, w = int(4.0 / RES), int(20.0 / RES)
    occ = np.zeros((h, w), dtype=bool)
    xs = OX + (np.arange(w) + 0.5) * RES
    ys = OY + (np.arange(h) + 0.5) * RES
    for yw in (-half_w, half_w):                             # the cell row containing the wall line
        occ[int(math.floor((yw - OY) / RES)), np.abs(xs) <= end_x] = True
    for xw in (-end_x, end_x):
        occ[np.abs(ys) <= half_w, int(math.floor((xw - OX) / RES))] = True
    return occ


def exact_scan(pose, n=180, end_x=8.0, half_w=1.0, max_range=12.0):
    """Ranges of n beams from the LiDAR at `pose` inside the rectangle |x| <= end_x, |y| <= half_w."""
    x, y, yaw = pose
    lx, ly = x + math.cos(yaw) * LASER[0], y + math.sin(yaw) * LASER[0]
    ang = -math.pi + np.arange(n) * (2 * math.pi / n)
    out = np.full(n, np.inf)
    for i, a in enumerate(ang):
        c, s = math.cos(yaw + a), math.sin(yaw + a)
        ts = []
        if abs(c) > 1e-9:
            ts += [((end_x if c > 0 else -end_x) - lx) / c]
        if abs(s) > 1e-9:
            ts += [((half_w if s > 0 else -half_w) - ly) / s]
        t = min(t for t in ts if t > 0)
        if t <= max_range:
            out[i] = t
    return ang, out


FIELD = DistanceField(corridor_field(), RES, OX, OY)


def test_distance_field_values_and_cell_convention():
    occ = np.zeros((20, 20), dtype=bool)
    occ[10, 10] = True                                       # one wall cell: x in [0.50, 0.55), y in [0.50, 0.55)
    f = DistanceField(occ, RES, 0.0, 0.0, cap_m=0.3)
    assert f.lookup(np.array([0.52]), np.array([0.52]))[0] == 0.0
    assert abs(f.lookup(np.array([0.62]), np.array([0.52]))[0] - 0.10) < 1e-6      # 2 cells right
    assert abs(f.lookup(np.array([0.62]), np.array([0.62]))[0] - 0.10 * math.sqrt(2)) < 1e-6
    assert abs(f.lookup(np.array([0.95]), np.array([0.52]))[0] - 0.30) < 1e-6      # capped
    assert f.lookup(np.array([-5.0]), np.array([99.0]))[0] == np.float32(0.3)      # outside the map: cap


def test_numpy_fallback_equals_scipy_or_brute_force():
    rng = np.random.default_rng(1)
    occ = rng.random((40, 50)) < 0.02
    fast = _distance_cells(occ, 8.0, use_scipy=False)
    wy, wx = np.nonzero(occ)
    yy, xx = np.mgrid[0:40, 0:50]
    brute = np.min(np.hypot(yy[..., None] - wy, xx[..., None] - wx), axis=-1)
    assert np.allclose(fast, np.minimum(brute, 8.0), atol=1e-5)


def test_correct_pose_gives_no_shift():
    truth = (2.0, 0.1, 0.0)
    ang, r = exact_scan(truth)
    res = shift_search(FIELD, truth, LASER, ang, r, SHIFTS)
    assert abs(res.shift) <= 0.025 + 1e-9 and res.gain == 0.0, res        # 'center': half a 5 cm step
    assert res.fit0 > 0.95, res


def test_ekf_ahead_is_measured():
    """EKF 0.5 m AHEAD of the truth (wheel over-read) -> best shift -0.5 (move the pose back)."""
    truth = (2.0, 0.1, 0.0)
    ang, r = exact_scan(truth)
    for err in (0.3, 0.5, 0.8, -0.4):
        ekf = (truth[0] + err, truth[1], truth[2])
        res = shift_search(FIELD, ekf, LASER, ang, r, SHIFTS)
        assert abs(res.shift + err) <= 0.051, (err, res)
        assert res.gain > 0.05 and res.width <= 0.3, (err, res)


def test_heading_not_along_x():
    """Same, with the corridor rotated: the shift is along the EKF's heading, not along x."""
    yaw = math.radians(30.0)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = lambda px, py: (c * px - s * py, s * px + c * py)               # noqa: E731
    # build a rotated corridor map by rotating cell centres back into the corridor frame
    h, w = 400, 400
    ox, oy = -10.0, -10.0
    xs = ox + (np.arange(w) + 0.5) * RES
    ys = oy + (np.arange(h) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    u, v = c * X + s * Y, -s * X + c * Y                                  # corridor frame
    occ = ((np.abs(np.abs(v) - 1.0) < RES * 0.75) & (np.abs(u) <= 8.0)) | \
          ((np.abs(np.abs(u) - 8.0) < RES * 0.75) & (np.abs(v) <= 1.0))
    field = DistanceField(occ, RES, ox, oy)
    ang, r = exact_scan((2.0, 0.1, 0.0))                                  # scan in the corridor frame
    tx, ty = rot(2.0, 0.1)
    ekf = (tx + 0.5 * c, ty + 0.5 * s, yaw)                               # 0.5 m ahead along the heading
    res = shift_search(field, ekf, LASER, ang, r, SHIFTS)
    assert abs(res.shift + 0.5) <= 0.051, res


def test_no_end_walls_in_range_means_no_measurement():
    """Only side walls visible (short range): every shift fits the same -> flat -> no alarm."""
    truth = (0.0, 0.0, 0.0)
    ang, r = exact_scan(truth, max_range=5.0)
    ekf = (0.5, 0.0, 0.0)
    res = shift_search(FIELD, ekf, LASER, ang, r, SHIFTS)
    mon = ShiftMonitor()
    assert not mon.valid(res), res
    assert res.width >= 1.0, res


def test_sideways_error_does_not_fake_an_along_error():
    truth = (2.0, 0.0, 0.0)
    ang, r = exact_scan(truth)
    res = shift_search(FIELD, (2.0, 0.3, 0.0), LASER, ang, r, SHIFTS)
    assert not ShiftMonitor().valid(res), res


def test_too_few_returns():
    ang = np.linspace(-1, 1, 30)
    r = np.full(30, np.inf)
    r[:5] = 3.0
    assert shift_search(FIELD, (0.0, 0.0, 0.0), LASER, ang, r, SHIFTS) is None


def test_monitor_needs_k_in_a_row_and_ignores_the_edge():
    SR = ShiftResult
    m = ShiftMonitor(k=2)
    good = SR(-0.5, 0.08, 0.1, 0.8, 0.88, 150)
    assert not m.update(good) and m.update(good)
    assert not m.update(SR(-1.0, 0.2, 0.05, 0.6, 0.8, 150))              # at the +-1 m edge: no measurement
    assert not m.update(good)                                             # run restarted
    assert not ShiftMonitor().valid(SR(-0.5, 0.02, 0.1, 0.8, 0.82, 150))  # gain too small
    assert not ShiftMonitor().valid(SR(-0.5, 0.08, 0.6, 0.8, 0.88, 150))  # flat optimum
    assert not ShiftMonitor().valid(SR(0.2, 0.08, 0.1, 0.8, 0.88, 150))   # small error: fine
    assert not ShiftMonitor().valid(None)


def test_cpu_budget_thins_out_and_recovers():
    b = CpuBudget(budget_ms=2.0, max_every=5, alpha=1.0)
    b.add(1.0)
    assert b.every_n == 1
    b.add(5.0)
    assert b.every_n == 3
    b.add(50.0)
    assert b.every_n == 5
    b.add(0.5)
    assert b.every_n == 1
    b2 = CpuBudget(budget_ms=1.0, alpha=1.0)
    b2.add(2.5)                                                           # every 3rd scan
    assert [b2.due() for _ in range(6)] == [False, False, True, False, False, True]


def test_speed():
    """One check = 41 shifts x 180 beams. Laptop ~0.2 ms; must stay far below the 143 ms scan period."""
    truth = (2.0, 0.1, 0.0)
    ang, r = exact_scan(truth)
    shift_search(FIELD, truth, LASER, ang, r, SHIFTS)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        shift_search(FIELD, (2.4, 0.1, 0.0), LASER, ang, r, SHIFTS)
    ms = (time.perf_counter() - t0) * 1000.0 / n
    print(f'      shift_search: {ms:.3f} ms per scan (41 x 180)')
    assert ms < 20.0


def test_state_history_nearest_equals_at():
    """ekf_node.StateHistory.nearest (backwards search, per scan) == the old full search at()."""
    try:
        from state_estimation.ekf_node import StateHistory
    except ImportError:
        from ekf_node import StateHistory
    h = StateHistory(2.0)
    rng = np.random.default_rng(3)
    t = 0
    for _ in range(300):
        t += int(rng.integers(8_000_000, 12_000_000))           # ~100 Hz with jitter
        h.add(t, np.array([t * 1e-9, 0.0, 0.0, 0.0, 1.0]))
    for q in list(rng.integers(h.buf[0][0] - 10_000_000, t + 50_000_000, 200)) + [t, h.buf[0][0]]:
        got = h.nearest(int(q))
        ref = h.at(int(q))
        assert got is not None and np.array_equal(got[1], ref), q
    assert h.nearest(h.buf[0][0] - 30_000_000) is None


def test_mismatch_classes_on_exact_scans():
    """Exact corridor scans: the right pose is CLEAN, along errors BAD (along), heading / sideways errors
    BAD (fit), and a pose 3 m off (only the end walls tell it apart) is NOT clean."""
    m = MismatchAlarm()
    truth = (2.0, 0.1, 0.0)
    ang, r = exact_scan(truth)

    def cls(pose):
        return m.classify(shift_search(FIELD, pose, LASER, ang, r, SHIFTS))
    assert cls(truth) == CLEAN
    assert cls((2.5, 0.1, 0.0)) == BAD_ALONG and cls((1.6, 0.1, 0.0)) == BAD_ALONG
    assert cls((2.0, 0.1, math.radians(10.0))) == BAD_FIT
    assert cls((2.0, 0.4, 0.0)) == BAD_FIT
    assert cls((5.0, 0.1, 0.0)) != CLEAN
    assert m.classify(None) == NO_RESULT


def test_mismatch_alarm_hysteresis():
    """ON after 2 BAD in a row; anything that is not CLEAN restarts the clean count and keeps it ON;
    OFF after 4 CLEAN in a row; once OFF, single BAD scans and 'between' values do not switch it on."""
    SR = ShiftResult
    bad = SR(-0.5, 0.08, 0.1, 0.8, 0.88, 150)                 # 0.5 m along error, clear optimum
    clean = SR(0.025, 0.0, 0.05, 0.97, 0.97, 170)
    between = SR(0.025, 0.03, 0.05, 0.97, 1.0, 170)           # g_clear < gain < g_min: neither
    dt = 1.0 / 7.0                                             # A1M8 scan period
    m = MismatchAlarm(k_on=2, k_clear=4, min_on_s=0.5)
    assert [m.update(x, i * dt) for i, x in enumerate([bad, clean, bad, bad])] == [False, False, False, True]
    assert m.cause == BAD_ALONG
    t = 3 * dt
    for x in (clean, clean, clean, between, clean, clean, None, clean, clean, clean):
        t += dt
        assert m.update(x, t), 'cleared too early'
    t += dt
    assert not m.update(clean, t)                              # 4th CLEAN in a row, on > 0.5 s -> OFF
    for x in (between, bad, clean, between, None, bad, clean):
        t += dt
        assert not m.update(x, t)
    m.update(bad, t + dt)
    m.reset()
    assert not m.alarm and m.n_bad == 0 and m.n_clean == 0


def test_mismatch_alarm_min_on_time():
    """Fast scans (40 Hz): 4 CLEAN scans arrive 0.1 s after ON -> stays ON until min_on_s has passed."""
    m = MismatchAlarm(k_on=2, k_clear=4, min_on_s=0.5)
    bad = ShiftResult(0.6, 0.1, 0.1, 0.7, 0.8, 150)
    clean = ShiftResult(0.0, 0.0, 0.05, 0.95, 0.95, 150)
    m.update(bad, 0.0)
    assert m.update(bad, 0.025) and m.t_on == 0.025
    states = [m.update(clean, 0.025 * (i + 2)) for i in range(30)]   # t = 0.05, 0.075, ...
    assert all(states[:18]) and not states[20], states                # OFF from t = 0.525 s


def test_far_off_pose_is_bad_not_clean():
    """A pose far off fits badly at EVERY shift -> gain ~0 like a correct pose; only fit0 tells them apart."""
    m = MismatchAlarm()
    assert m.classify(ShiftResult(0.0, 0.0, 2.0, 0.3, 0.3, 150)) == BAD_FIT
    assert m.classify(ShiftResult(0.0, 0.0, 2.0, 0.8, 0.8, 150)) == NEITHER     # between fit_bad and fit_clear
    assert m.classify(ShiftResult(0.0, 0.0, 0.05, 0.95, 0.95, 150)) == CLEAN


def test_mismatch_alarm_needs_a_gap():
    for kw in (dict(fit_bad=0.9, fit_clear=0.9), dict(g_clear=0.05, g_min=0.05)):
        try:
            MismatchAlarm(**kw)
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
