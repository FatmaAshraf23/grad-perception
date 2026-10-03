#!/usr/bin/env python3
"""Self-test of steer_calib.SteeringCalibrator on a SYNTHETIC drive with known answers.

Straights and curves to both sides, servo rate limit, yaw rate lagging the steering,
noise on all sensors, a gyro bias, and UNDERSTEER like the simulator's car (it turns
~3 % less than r = v tan(delta)/L at 0.8 m/s; job 045). The calibrator must find the
steering offset always, the actuator gain only from slow driving, and must NOT report
the understeer as a gain. Run: python3 steer_calib_selftest.py
"""
import math
import os
import random
import sys

try:
    from state_estimation.steer_calib import SteeringCalibrator
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from steer_calib import SteeringCalibrator  # noqa: E402

L = 0.3302
K_US = 0.0176      # understeer: r = v tan(delta) / (L + K_US v^2) -> 3.3 % less at 0.8 m/s


def drive(offset, gain, speeds, k_true=0.9709, k_used=0.9709, bias=0.012, seed=1, duration=170.0):
    rnd = random.Random(seed)
    cal = SteeringCalibrator(L, speed_scale=k_used)
    plan = [(6, 0.0), (4, 0.15), (5, 0.0), (4, -0.25), (5, 0.0), (3, 0.35), (6, 0.0), (4, -0.1)]
    dt, t, v, d_act, r = 0.01, 0.0, 0.0, 0.0, 0.0
    seg_t, seg = 0.0, 0
    while t < duration:
        if t < 3.0:
            v_target, cmd = 0.0, 0.0                     # standing still first (gyro bias)
        else:
            v_target = speeds[int((t - 3.0) // 40.0) % len(speeds)]   # 40 s per speed
            seg_t += dt
            if seg_t > plan[seg][0]:
                seg, seg_t = (seg + 1) % len(plan), 0.0
            cmd = plan[seg][1]
        v += max(-0.8 * dt, min(0.8 * dt, v_target - v))
        d_act += max(-3.2 * dt, min(3.2 * dt, gain * cmd - d_act))   # servo rate limit
        r_ss = v * math.tan(d_act) / (L + K_US * v * v)
        r += (r_ss - r) * dt / 0.1                         # yaw rate lags 0.1 s
        gyro = r + bias + rnd.gauss(0.0, 0.005)
        v_meas = v / k_true + (rnd.gauss(0.0, 0.02) if v > 1e-3 else 0.0)
        delta_meas = cmd + offset + rnd.gauss(0.0, 0.005)  # what the STM32 / fake sensor reports
        cal.add(t, v_meas, delta_meas, gyro)
        t += dt
    return cal.estimate()


#        name                                         offset  gain  speeds       k_used  expect gain
cases = [('sim-like, slow + normal driving',          0.010, 1.00, [0.4, 0.85], 0.9709, 'determined'),
         ('servo trim -0.020, linkage gain 0.92',     -0.020, 0.92, [0.4, 0.85], 0.9709, 'determined'),
         ('wheel speed NOT calibrated (k used 1.0)',   0.010, 1.00, [0.4, 0.85], 1.0,    'off by k'),
         ('only normal speed (0.85 m/s)',              0.010, 1.00, [0.85],      0.9709, 'not determined')]
ok = True
for name, off, g, speeds, ku, expect in cases:
    est = drive(off, g, speeds, k_used=ku)
    e_off = est['offset'] - off
    good_off = abs(e_off) < 0.001
    if expect == 'determined':
        good_g = est['gain_determined'] and abs(est['gain'] - g) < 0.02
    elif expect == 'not determined':
        good_g = (not est['gain_determined']) and est['gain'] == 1.0
    else:
        good_g = est['gain_determined']                  # k wrong: the gain may be off by the speed error
    ok &= good_off and good_g
    app = ('-' if est['apparent_gain'] is None else
           f"{est['apparent_gain']:.3f} at {est['apparent_speed']:.2f} m/s")
    print(f"{name:42s} -> offset {est['offset']:+.5f} (error {e_off:+.5f}) {'OK' if good_off else 'FAIL'} | "
          f"gain {est['gain']:.3f} {'(from ' + str(est['n_slow']) + ' slow pairs)' if est['gain_determined'] else '(not determined)'} "
          f"true {g:.2f}, expected {expect}: {'OK' if good_g else 'FAIL'} | apparent gain {app}")
print('SELF-TEST', 'PASSED' if ok else 'FAILED')
sys.exit(0 if ok else 1)
