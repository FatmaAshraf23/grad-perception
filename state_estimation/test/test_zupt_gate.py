#!/usr/bin/env python3
"""Tests for the standstill gate (LocalizationEKF.standstill + StandstillTurns, 2026-10-24): with the wheels standing
still the gyro reading is learned as bias -- unless it is far from the known bias, which means the car is being TURNED
(by hand) -- and a "turn" that never ends is a gyro-bias change, not a turn.
numpy only. Run: python3 test_zupt_gate.py   (or pytest)"""
import math
import os
import sys

import numpy as np

try:
    from state_estimation.ekf_node import LocalizationEKF, StandstillTurns
except ImportError:                                  # run from the source tree without ROS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'state_estimation'))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from ekf_node import LocalizationEKF, StandstillTurns

DT = 0.01                                            # 100 Hz IMU
SIGMA_W = 0.005                                      # rad/s per sample (simulator gyro noise; ekf_node sigma_w)
B_TRUE = math.radians(0.6)                           # the simulator's gyro bias
GATE = 16.0


def ekf(sigma_bias0=0.05):
    return LocalizationEKF(x0=[0.0, 0.0, 0.0, 0.0, 1.0], P0_diag=[0.1, 0.1, math.radians(2.0), sigma_bias0, 0.05],
                           sigma_v=0.05, sigma_w=SIGMA_W, sigma_bw=0.0001, sigma_kw=0.0005)


def still(f, n, rng, gate=GATE, b=B_TRUE):
    turned = 0
    for _ in range(n):
        turned += f.standstill(b + rng.normal(0.0, SIGMA_W), DT, gate)[0]
    return turned


def hand_turn(f, rng, deg=15.0, secs=1.0, gate=GATE):
    """A half-sine turn of `deg` degrees in `secs` seconds (peak 23.6 deg/s for 15 deg in 1 s), wheels still."""
    n = int(round(secs / DT))
    peak = math.radians(deg) * math.pi / (2.0 * secs)
    turned = 0
    for i in range(n):
        rate = peak * math.sin(math.pi * (i + 0.5) / n)
        turned += f.standstill(B_TRUE + rate + rng.normal(0.0, SIGMA_W), DT, gate)[0]
    return turned


def test_standing_still_learns_the_bias():
    rng = np.random.default_rng(1)
    f = ekf()
    turned = still(f, 1000, rng)
    assert abs(f.x[3] - B_TRUE) < 0.001, math.degrees(f.x[3])
    assert turned <= 2, turned                       # 4-sigma gate: noise is almost never taken for a turn
    assert abs(f.x[2]) < math.radians(0.05)          # standing still: the heading does not move


def test_hand_turn_keeps_the_bias_and_the_heading_follows():
    rng = np.random.default_rng(2)
    f = ekf()
    still(f, 300, rng)
    yaw0, b0 = f.x[2], f.x[3]
    turned = hand_turn(f, rng)
    still(f, 200, rng)
    assert turned > 80, turned                       # most of the 100 turn samples were seen as a turn
    assert abs(f.x[3] - B_TRUE) < 0.0015, math.degrees(f.x[3])
    assert abs(f.x[3] - b0) < 0.001
    assert abs(math.degrees(f.x[2] - yaw0) - 15.0) < 0.5, math.degrees(f.x[2] - yaw0)


def test_old_rule_learns_the_turn_as_bias():
    """gate 0 = the rule before 2026-10-24: reproduces job 195's F10 (bias several deg/s, heading frozen)."""
    rng = np.random.default_rng(2)
    f = ekf()
    still(f, 300, rng, gate=0.0)
    yaw0 = f.x[2]
    hand_turn(f, rng, gate=0.0)
    assert f.x[3] - B_TRUE > math.radians(1.0), math.degrees(f.x[3])
    assert abs(math.degrees(f.x[2] - yaw0)) < 1.0


def test_unknown_start_bias_is_still_learned():
    """At start-up the bias is uncertain (P_bb from sigma_bias0) -> the gate is wide: a 5.7 deg/s bias is learned."""
    rng = np.random.default_rng(3)
    f = ekf(sigma_bias0=0.05)
    b_big = 0.1
    turned = sum(f.standstill(b_big + rng.normal(0.0, SIGMA_W), DT, GATE)[0] for _ in range(300))
    assert abs(f.x[3] - b_big) < 0.002, math.degrees(f.x[3])
    assert turned <= 2, turned


def test_bias_change_after_driving_is_accepted():
    """After a drive P_bb has grown; a 0.5 deg/s bias change found at the next stop is learned, not taken for a turn."""
    rng = np.random.default_rng(4)
    f = ekf()
    still(f, 300, rng)
    f.P[3, 3] = 0.003 ** 2                           # as after a few minutes of driving
    b_new = B_TRUE + math.radians(0.5)
    turned = still(f, 300, rng, b=b_new)
    assert abs(f.x[3] - b_new) < 0.001, math.degrees(f.x[3])
    assert turned <= 2, turned


def test_a_slow_turn_is_detected_above_the_threshold():
    """3 deg/s for 2 s with the wheels still (well above 4 sigma_w ~ 1.15 deg/s): a turn, not bias."""
    rng = np.random.default_rng(5)
    f = ekf()
    still(f, 300, rng)
    yaw0 = f.x[2]
    turned = sum(f.standstill(B_TRUE + math.radians(3.0) + rng.normal(0.0, SIGMA_W), DT, GATE)[0] for _ in range(200))
    assert turned > 180, turned
    assert abs(math.degrees(f.x[2] - yaw0) - 6.0) < 0.5, math.degrees(f.x[2] - yaw0)
    assert abs(f.x[3] - B_TRUE) < 0.001


def run(turns, f, rates, rng):
    """Feed StandstillTurns gyro readings (true turn rate per sample + B_TRUE + noise) -> its events."""
    return [ev for r in rates for ev in [turns.step(f, B_TRUE + r + rng.normal(0.0, SIGMA_W), DT)] if ev]


def half_sine(deg, secs):
    n = int(round(secs / DT))
    peak = math.radians(deg) * math.pi / (2.0 * secs)
    return [peak * math.sin(math.pi * (i + 0.5) / n) for i in range(n)]


def test_turn_events_one_start_one_end():
    rng = np.random.default_rng(6)
    f, turns = ekf(), StandstillTurns(gate=GATE, max_turn_s=3.0, sigma_bias0=0.05)
    assert run(turns, f, [0.0] * 300, rng) == []
    ev = run(turns, f, half_sine(15.0, 1.0) + [0.0] * 100, rng)
    assert [e[0] for e in ev] == ['start', 'end'], ev
    assert ev[1][2] == 'standing still again'
    assert abs(ev[1][1] - 15.0) < 0.5, ev[1][1]
    assert turns.turn_samples > 80 and turns.bias_changes == 0


def test_old_rule_gives_no_events():
    rng = np.random.default_rng(6)
    f, turns = ekf(), StandstillTurns(gate=0.0, max_turn_s=3.0)
    assert run(turns, f, [0.0] * 300 + half_sine(15.0, 1.0) + [0.0] * 100, rng) == []
    assert turns.turn_samples == 0 and turns.zupts == 500


def test_driving_off_ends_the_turn():
    rng = np.random.default_rng(7)
    f, turns = ekf(), StandstillTurns(gate=GATE)
    run(turns, f, [0.0] * 300, rng)
    ev = run(turns, f, half_sine(15.0, 1.0)[:50], rng)
    assert [e[0] for e in ev] == ['start']
    end = turns.moving()
    assert end[0] == 'end' and end[2] == 'the car set off' and abs(end[1] - 7.5) < 0.5, end
    assert turns.moving() is None                    # nothing open any more


def test_long_steady_turn_is_a_bias_change():
    """The bias jumps by 3 deg/s while the car stands still (e.g. warmed up while driving): the first 3 s look like
    a turn (the heading follows), then the rule takes it as a bias change -> heading restored, new bias learned."""
    rng = np.random.default_rng(8)
    f, turns = ekf(), StandstillTurns(gate=GATE, max_turn_s=3.0, sigma_bias0=0.05)
    run(turns, f, [0.0] * 300, rng)
    yaw0 = f.x[2]
    step = math.radians(3.0)                          # fed as a "turn rate" = bias change, the car does not turn
    ev = run(turns, f, [step] * 800, rng)
    assert [e[0] for e in ev] == ['start', 'bias_change'], ev
    assert abs(ev[1][1] - 9.0) < 0.5 and 3.0 < ev[1][2] < 3.1, ev[1]
    assert abs(math.degrees(f.x[2] - yaw0)) < 0.2, math.degrees(f.x[2] - yaw0)
    assert abs(f.x[3] - (B_TRUE + step)) < math.radians(0.05), math.degrees(f.x[3])
    assert turns.bias_changes == 1


def test_a_two_second_hand_turn_is_not_a_bias_change():
    rng = np.random.default_rng(9)
    f, turns = ekf(), StandstillTurns(gate=GATE, max_turn_s=3.0)
    run(turns, f, [0.0] * 300, rng)
    yaw0 = f.x[2]
    ev = run(turns, f, half_sine(30.0, 2.0) + [0.0] * 100, rng)
    assert [e[0] for e in ev] == ['start', 'end'], ev
    assert abs(math.degrees(f.x[2] - yaw0) - 30.0) < 0.6, math.degrees(f.x[2] - yaw0)
    assert abs(f.x[3] - B_TRUE) < 0.0015


if __name__ == '__main__':
    tests = [(n, fn) for n, fn in sorted(globals().items()) if n.startswith('test_') and callable(fn)]
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
