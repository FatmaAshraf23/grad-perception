#!/usr/bin/env python3
"""Tests for wheel_delay.py (wheel-speed delay compensation used by both EKFs), 2026-10-19.
Run: python3 test_wheel_delay.py   (or pytest)"""
import os
import sys

try:
    from state_estimation.wheel_delay import WheelSpeedCompensator
except ImportError:                                  # run from the source tree without ROS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'state_estimation'))
    from wheel_delay import WheelSpeedCompensator


def drive(comp, accel, tau_w, t_end=2.0, imu_dt=0.01, wheel_dt=0.02, bias=0.0, stand=0.5):
    """the car stands for `stand` s, then accelerates at `accel`; the wheel speed (50 Hz) is the true
    speed tau_w before its stamp, the IMU (100 Hz) measures accel + bias.  Returns the largest error of
    (wheel value + compensation) vs the true speed while driving, and the same without compensation."""
    v_true = lambda t: max(0.0, t - stand) * accel  # noqa: E731
    t, last_w, v_w = 0.0, -1.0, 0.0
    worst = worst_raw = 0.0
    while t < t_end:
        t = round(t + imu_dt, 9)
        if t - last_w >= wheel_dt - 1e-9:            # a wheel message (arrives before the IMU at the same time)
            last_w = t
            v_w = v_true(t - tau_w)
            comp.on_wheel(t)
        standing = t < stand - 0.05
        a = (accel if t > stand else 0.0) + bias
        dv = comp.on_imu(t, a, imu_dt, standing)
        if t > stand + 0.1:
            worst = max(worst, abs(v_w + dv - v_true(t)))
            worst_raw = max(worst_raw, abs(v_w - v_true(t)))
    return worst, worst_raw


def test_constant_acceleration_is_bridged_exactly():
    """4 m/s^2 with a 10 ms wheel delay + 20 ms wheel period: without it up to 0.08 m/s behind, with it ~0"""
    worst, worst_raw = drive(WheelSpeedCompensator(tau_w=0.010), 4.0, 0.010)
    assert worst_raw > 0.07, worst_raw
    assert worst < 0.005, worst


def test_bias_is_learned_while_standing():
    """0.3 m/s^2 accelerometer bias: learned during the stop, so it does not leak into the speed"""
    comp = WheelSpeedCompensator(tau_w=0.010)
    worst, _ = drive(comp, 4.0, 0.010, bias=0.3)
    assert abs(comp.b_ax - 0.3) < 1e-6, comp.b_ax
    assert worst < 0.005, worst


def test_no_bias_learning_means_bounded_error():
    """bias never learned (no stop): the error stays bias x (tau_w + wheel period), it does not grow"""
    comp = WheelSpeedCompensator(tau_w=0.010)
    worst, _ = drive(comp, 4.0, 0.010, bias=0.3, stand=0.0)
    assert worst < 0.3 * 0.031 + 1e-6, worst


def test_old_wheel_value_is_not_extrapolated_forever():
    """wheel messages stop (link lost): the gain freezes after max_age instead of integrating on"""
    comp = WheelSpeedCompensator(tau_w=0.010, max_age=0.1)
    comp.on_imu(0.0, 0.0, 0.01, True)                # bias 0, ready
    comp.on_wheel(1.0)
    gained = [comp.on_imu(1.0 + 0.01 * i, 2.0, 0.01, False) for i in range(1, 50)]
    assert abs(gained[-1] - gained[9]) < 1e-12        # frozen from ~90 ms on (100 ms incl. tau_w)
    assert 0.15 < gained[-1] < 0.2, gained[-1]


def test_standing_resets_the_gain_and_no_wheel_yet_gives_zero():
    comp = WheelSpeedCompensator(tau_w=0.010)
    assert comp.on_imu(0.0, 1.0, 0.01, False) == 0.0  # no wheel message yet
    comp.on_wheel(0.0)
    assert comp.on_imu(0.01, 1.0, 0.01, False) > 0.0
    assert comp.on_imu(0.02, 1.0, 0.01, True) == 0.0  # standing
    assert comp.dv == 0.0


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
