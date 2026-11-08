#!/usr/bin/env python3
"""Tests for wheel_stuck.py (F12 "wheel stuck at 0 while driving", 2026-11-08). No ROS.
Run: python3 test_wheel_stuck.py

The streams are 50 Hz wheel samples (the simulator's rate) with noise like fake_vehicle_sensors (sigma 0.02 m/s);
v_imu is the IMU (observer) speed = the true speed + a small error, as the slip check gives it.
"""
import random
import sys

try:
    from state_estimation.wheel_stuck import StuckWheel
except ImportError:                                    # run from the staging folder
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from wheel_stuck import StuckWheel

RESULTS = []
DT = 0.02


def check(name, cond, info=''):
    RESULTS.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f'   [{info}]' if info else ''))


def raises(f):
    try:
        f()
    except ValueError:
        return True
    return False


def run(sw, profile, n, stuck_from=None, stuck_to=None, noise=0.02, imu_err=0.1, seed=1):
    """feed n samples of profile(t); the wheel reads EXACTLY 0 in [stuck_from, stuck_to) (as the fixed
    fake_vehicle_sensors does). -> list of (t, true v, wheel v, dropped)"""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        t = i * DT
        v = profile(t)
        stuck = stuck_from is not None and stuck_from <= t < (stuck_to if stuck_to is not None else 1e9)
        w = 0.0 if stuck else (v + rnd.gauss(0.0, noise) if v != 0.0 else 0.0)
        out.append((t, v, w, sw.check(t, w, v + imu_err)))
    return out


def racing(t):
    """6 m/s, brake at 8 m/s^2 to 2, accelerate at 4 m/s^2 back, period 3 s"""
    p = t % 3.0
    if p < 1.0:
        return 6.0
    if p < 1.5:
        return 6.0 - 8.0 * (p - 1.0)
    if p < 2.0:
        return 2.0
    return min(6.0, 2.0 + 4.0 * (p - 2.0))


def stop_and_go(t):
    """4 m/s, brake hard (10 m/s^2 = more than this car can do) to 0, parked 2 s, launch at 4 m/s^2"""
    if t < 1.0:
        return 4.0
    if t < 1.4:
        return max(0.0, 4.0 - 10.0 * (t - 1.0))
    if t < 3.4:
        return 0.0
    return min(4.0, 4.0 * (t - 3.4))


def main():
    # 1. normal racing (10 s), noise: never stuck
    sw = StuckWheel()
    r = run(sw, racing, 500)
    check('racing 10 s (6 <-> 2 m/s, braking 8 m/s^2): nothing dropped', sw.dropped_total == 0 and not sw.events,
          f'dropped {sw.dropped_total}')

    # 2. a REAL stop (even harder braking than the car can do), parked, launch: never stuck, ZUPT stays possible
    sw = StuckWheel()
    r = run(sw, stop_and_go, 300, imu_err=0.05)
    check('real stop at 10 m/s^2 + 2 s parked + launch: nothing dropped (the wheel slows THROUGH v_min)',
          sw.dropped_total == 0, f'dropped {sw.dropped_total}')
    sw = StuckWheel()
    r = run(sw, stop_and_go, 300, imu_err=0.3)
    check('... also when the IMU speed reads 0.3 m/s high while parked', sw.dropped_total == 0,
          f'dropped {sw.dropped_total}')

    # 3. stuck at 6 m/s for 1 s, then back
    sw = StuckWheel()
    r = run(sw, lambda t: 6.0, 150, stuck_from=1.0, stuck_to=2.0)
    first = next(i for i, x in enumerate(r) if x[0] >= 1.0)
    in_fault = [x for x in r if 1.0 <= x[0] < 2.0]
    check('stuck at 6 m/s: the FIRST zero is dropped (nothing reaches the EKF)', r[first][3])
    check('stuck: every zero during the fault is dropped', all(x[3] for x in in_fault), f'{len(in_fault)} samples')
    ev = sw.pop_events()
    check("events: 'stuck' at ON with the last good speed, then 'back'",
          len(ev) == 2 and ev[0][0] == 'stuck' and abs(ev[0][1] - 1.0) < 1e-9 and abs(ev[0][2] - 6.0) < 0.1
          and ev[1][0] == 'back', str([(e[0], round(e[1], 2)) for e in ev]))
    back = [x for x in r if x[0] >= 2.0 - 1e-9]
    check('back: the first 2 good samples are dropped, the 3rd (t = OFF + 40 ms) is delivered',
          back[0][3] and back[1][3] and not back[2][3] and not any(x[3] for x in back[2:]),
          f'back event at {ev[1][1]:.2f} s, stuck for {ev[1][2]:.2f} s, {ev[1][3]} dropped')
    check('pop_events empties the list', sw.pop_events() == [])
    check('counters: dropped = fault samples + 2', sw.dropped_total == len(in_fault) + 2,
          f'{sw.dropped_total} vs {len(in_fault)} + 2')

    # 4. stays stuck while the car REALLY stops (a dead sensor must not produce "parked")
    sw = StuckWheel()
    r = run(sw, stop_and_go, 300, stuck_from=0.5)
    check('stuck at 4 m/s, then the car really stops and launches: stays stuck (all dropped, no wrong ZUPT)',
          sw.stuck and all(x[3] for x in r if x[0] >= 0.5))

    # 5. a single spike while stuck does not end it; nor does an implausible sequence
    sw = StuckWheel()
    sw.check(0.0, 6.0, 6.0)
    seq = [0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 3.1, 0.0]
    d = [sw.check(DT * (i + 1), v, 6.0) for i, v in enumerate(seq)]
    check('single good-looking readings between zeros do not end it', all(d) and sw.stuck)
    sw = StuckWheel()
    sw.check(0.0, 6.0, 6.0)
    seq = [0.0] + [6.0, 1.0] * 10
    d = [sw.check(DT * (i + 1), v, 6.0) for i, v in enumerate(seq)]
    check('jumping readings (6, 1, 6, 1 ... = a broken signal, |dv| > a_max dt + tol) never end it', all(d) and sw.stuck)

    # 6. dead from the start: no good wheel sample yet, the IMU speed says the car moves
    sw = StuckWheel()
    d = [sw.check(DT * i, 0.0, 3.0) for i in range(20)]
    check('dead encoder from the start (only the IMU says 3 m/s): stuck on the first sample', all(d) and sw.stuck)
    sw = StuckWheel()
    d = [sw.check(DT * i, 0.0, None) for i in range(20)]
    check('no IMU speed (observer not synced) and no good sample: never stuck (no evidence)', not any(d))

    # 7. reverse driving
    sw = StuckWheel()
    r = run(sw, lambda t: -1.5, 100, stuck_from=1.0, imu_err=0.0)
    check('reversing at -1.5 m/s, then stuck: detected (|v| is used)', sw.stuck and r[50][3] and not r[49][3])

    # 8. the documented limit: stuck at a crawl (below v_min) is NOT caught here (the 1-s wheel-lost rule cannot
    #    fire either: the samples look like a real stop -> the old behaviour, small at that speed)
    sw = StuckWheel()
    run(sw, lambda t: 0.3, 100, stuck_from=1.0, imu_err=0.0)
    check('LIMIT: stuck at 0.3 m/s (< v_min 0.5) is not detected', not sw.stuck)

    # 9. parameters
    check('bad parameters raise', raises(lambda: StuckWheel(v_zero=0.6, v_min=0.5)) and raises(lambda: StuckWheel(k_back=0))
          and raises(lambda: StuckWheel(a_max=0.0)) and raises(lambda: StuckWheel(dv_tol=-1.0)))
    sw = StuckWheel(k_back=1)
    sw.check(0.0, 6.0, 6.0)
    sw.check(0.02, 0.0, 6.0)
    check('k_back 1: the first good sample is delivered', sw.check(0.04, 6.0, 6.0) is False and not sw.stuck)

    print(f'\n{sum(RESULTS)} of {len(RESULTS)} passed')
    return 0 if all(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
