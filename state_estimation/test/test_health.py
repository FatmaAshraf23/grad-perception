#!/usr/bin/env python3
"""Tests for health.py (LagMonitor = the estimator-lag self-check of contract v0.3), 2026-10-24.
Run: python3 test_health.py   (or pytest). No ROS needed."""
import os
import random
import sys

try:
    from state_estimation.health import LagMonitor
except ImportError:                                  # run from the source tree without ROS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'state_estimation'))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from health import LagMonitor

DT = 0.01                                            # 100 Hz, one estimate per IMU message


def feed(mon, ages, t0=0.0):
    return [mon.update(a, t0 + i * DT) for i, a in enumerate(ages)]


def test_normal_ages_never_flag():
    rnd = random.Random(1)
    mon = LagMonitor()
    assert not any(feed(mon, [rnd.uniform(1.5, 9.0) for _ in range(3000)]))
    assert mon.episodes == 0


def test_single_late_messages_are_ignored():
    mon = LagMonitor()
    assert not any(feed(mon, [3, 3, 60, 3, 3, 26, 3, 80, 4]))


def test_on_after_two_and_holds_in_the_band():
    mon = LagMonitor()
    assert feed(mon, [3, 30, 31]) == [False, False, True] and mon.episodes == 1
    assert all(feed(mon, [15.0] * 300, t0=0.03))     # 10..25 ms: between the thresholds -> keeps its state
    assert mon.worst_ms == 31


def test_off_only_after_clear_s_below_off_ms():
    mon = LagMonitor(on_ms=25, k_on=2, off_ms=10, clear_s=1.0)
    feed(mon, [40, 40])
    states = feed(mon, [5.0] * 150, t0=0.02)         # first young message at t = 0.02 s
    assert all(states[:99]) and not states[101], 'must clear 1.0 s after the first young message'


def test_clear_timer_restarts():
    mon = LagMonitor()
    feed(mon, [40, 40])
    assert all(feed(mon, [5.0] * 80, t0=0.02))       # 0.8 s young ...
    assert mon.update(12.0, 0.82)                    # ... one message >= off_ms restarts the timer
    states = feed(mon, [5.0] * 150, t0=0.83)
    assert all(states[:99]) and not states[101]


def test_f07_like_overload_is_flagged_at_once():
    """F07: 12 ms of work per 10 ms IMU period -> each estimate leaves 2 ms later than the one before."""
    mon = LagMonitor()
    ages = [2.0 + 2.0 * i for i in range(100)]
    out = feed(mon, ages)
    assert out.index(True) == 13 and ages[12] > 25 >= ages[11]   # the 2nd message above 25 ms


def test_negative_age_is_no_lag():
    """A clock that stepped back gives negative ages -- no lag."""
    mon = LagMonitor()
    assert not any(feed(mon, [-500, -500, -30, -30]))


def test_needs_a_gap():
    try:
        LagMonitor(on_ms=10, off_ms=10)
    except ValueError:
        return
    raise AssertionError('no ValueError for off_ms == on_ms')


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
