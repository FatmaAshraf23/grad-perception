#!/usr/bin/env python3
"""Tests for slip_check.py (SlipCheck + SimGate = the wheel-slip check, Fix 3), 2026-10-26.
Run: python3 test_slip_check.py   (or pytest). No ROS needed.

A small synthetic car: a true speed profile -> IMU at 100 Hz (the true acceleration through a 40 ms
first-order lag like fake_imu, + bias + noise) and wheel at 50 Hz (the true speed 10 ms late, x a
fault factor, + noise). Seeded, so every run gives the same numbers."""
import os
import random
import sys

try:
    from state_estimation.slip_check import SimGate, SlipCheck
except ImportError:                                  # run from the source tree without ROS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'state_estimation'))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from slip_check import SimGate, SlipCheck


def profile(t, stand=1.0, accel=3.0, cruise=4.0):
    """Stand still, accelerate, cruise."""
    if t < stand:
        return 0.0
    return min(cruise, accel * (t - stand))


def lively(t):
    """Stand 1 s, +4 m/s^2 to 6 m/s, cruise, brake -5 m/s^2 to 2 m/s, +3 m/s^2 back to 5 m/s, cruise."""
    pts = [(0.0, 0.0), (1.0, 0.0), (2.5, 6.0), (6.0, 6.0), (6.8, 2.0), (9.0, 2.0), (10.0, 5.0), (99.0, 5.0)]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return pts[-1][1]


def drive(T, prof=profile, fault=None, bias=0.08, imu_noise=0.05, wheel_noise=0.03, imu_gap=None,
          dip=None, gate=None, mark_gated=False, seed=1, sc=None):
    """Feed a SlipCheck. fault = (t0, t1, factor) on the wheel; imu_gap = (t0, t1) no IMU samples;
    dip = (t0, t1, factor): a SIMULATOR artifact (the wheel AND the error-free wheel dip, the IMU does not);
    gate: a SimGate fed with /sim/wheel_ideal; mark_gated: pass gated=True for the dip samples instead.
    Returns (sc, rows) with rows (t, on, v_obs, v_true, v_wheel)."""
    rnd = random.Random(seed)
    sc = sc or SlipCheck(sim_gate=gate)
    a_lp, v_prev, rows = 0.0, prof(0.0), []
    h = 0.001
    n = int(round(T / h))
    for i in range(1, n + 1):
        t = i * h
        v = prof(t)
        a = (v - v_prev) / h
        v_prev = v
        a_lp += h / 0.040 * (a - a_lp)                         # fake_imu's 40 ms smoothing
        if i % 10 == 0 and not (imu_gap and imu_gap[0] <= t < imu_gap[1]):          # IMU 100 Hz
            sc.imu(t, a_lp + bias + rnd.gauss(0.0, imu_noise))
        if i % 20 == 5:                                         # wheel 50 Hz, 10 ms late
            vt = prof(t - 0.010)
            f = fault[2] if fault and fault[0] <= t < fault[1] else 1.0
            in_dip = bool(dip and dip[0] <= t < dip[1])
            if in_dip:
                f *= dip[2]
            vw = vt * f + (rnd.gauss(0.0, wheel_noise) if abs(vt * f) > 1e-3 else 0.0)   # locked/stopped: exactly 0
            if gate is not None:                                # /sim/wheel_ideal of the same tick
                gate.update(t, vt * (dip[2] if in_dip else 1.0), vt)
            sc.wheel(t, vw, gated=mark_gated and in_dip)
            rows.append((t, sc.on, sc.observer_speed, vt, vw))
    return sc, rows


def first_on(rows, after):
    return next((r[0] for r in rows if r[0] >= after and r[1]), None)


def first_off(rows, after):
    return next((r[0] for r in rows if r[0] >= after and not r[1]), None)


def check_fault(fault, T=8.0):
    t0, t1, _ = fault
    sc, rows = drive(T, fault=fault)
    on = first_on(rows, t0)
    assert on is not None, 'slip never detected'
    assert on - t0 <= 0.080, f'detected {1000 * (on - t0):.0f} ms after the fault started (> 80 ms)'
    off = first_off(rows, t1)
    assert off is not None and off - t1 <= 0.350, f'flag still on {1000 * ((off or T) - t1):.0f} ms after the fault'
    during = [r for r in rows if t0 <= r[0] < t1]
    share = sum(r[1] for r in during) / len(during)
    assert share >= 0.85, f'only {100 * share:.0f} % of the fault flagged'
    flagged = [r for r in during if r[1]]
    e_obs = sum(abs(r[2] - r[3]) for r in flagged) / len(flagged)
    e_whl = sum(abs(r[4] - r[3]) for r in flagged) / len(flagged)
    assert e_obs < 0.1 and e_obs < 0.3 * e_whl, f'observer {e_obs:.3f} m/s off vs wheel {e_whl:.3f} m/s off'
    outside = [r for r in rows if r[1] and not (t0 <= r[0] <= t1 + 0.4)]
    assert not outside, f'flag on outside the fault at t = {outside[0][0]:.2f} s'
    assert sc.episodes == 1, f'{sc.episodes} episodes (1 expected)'
    return sc


def test_spin():
    """F05-like wheelspin: wheel x 1.15 for 1 s at 4 m/s."""
    check_fault((5.0, 6.0, 1.15))


def test_skid():
    """F06-like skid: wheel x 0.85 for 0.6 s."""
    check_fault((5.0, 5.6, 0.85))


def test_locked_wheels_skid():
    """Hard braking (-3 m/s^2 from 4 m/s) with the wheels LOCKED for 0.6 s: the encoder reads exactly 0 while
    the car still slides. A wheel at 0 must not count as 'parked' while slipping (else: v_obs forced to 0,
    the braking learned as bias -0.36 m/s^2, and a false slip with negative speed after the wheels unlock)."""
    prof = lambda t: 4.0 if t < 5.0 else max(1.0, 4.0 - 3.0 * (t - 5.0))      # noqa: E731
    sc, rows = drive(7.0, prof=prof, fault=(5.0, 5.6, 0.0))
    on = first_on(rows, 5.0)
    assert on is not None and on - 5.0 <= 0.080, 'locked wheels not detected'
    during = [r for r in rows if 5.1 <= r[0] < 5.6]
    assert all(r[1] for r in during), 'flag dropped while the wheels were locked'
    assert max(abs(r[2] - r[3]) for r in during) < 0.15, 'observer does not follow the sliding car'
    # the car keeps braking until 6.0 s; the 40 ms IMU smoothing keeps wheel and observer ~0.1 m/s apart
    # until then, so the flag clears ~0.5 s after the wheels unlock. Harmless: what matters is that the
    # speed the EKF uses while flagged is right.
    off = first_off(rows, 5.6)
    assert off is not None and off - 5.6 <= 0.700, 'flag stuck after the wheels unlocked'
    flagged = [r for r in rows if r[1]]
    assert max(abs(r[2] - r[3]) for r in flagged) < 0.15, 'observer speed wrong while flagged'
    assert sc.episodes == 1, f'{sc.episodes} episodes (1 expected)'
    assert abs(sc.bias - 0.08) < 0.03, f'bias {sc.bias:.3f}: the braking was learned as bias'


def test_spin_while_accelerating():
    """Wheelspin in the middle of a 3 m/s^2 acceleration (the c_a * |a| margin must not hide it)."""
    sc, rows = drive(6.0, prof=lambda t: profile(t, cruise=9.0), fault=(2.0, 2.6, 1.15))
    on = first_on(rows, 2.0)
    assert on is not None and on - 2.0 <= 0.080, f'detected after {on}'


def test_no_false_alarm():
    """20 s of hard acceleration / braking / cruising with noise and bias, 5 seeds: never on."""
    for seed in range(5):
        sc, rows = drive(20.0, prof=lively, seed=seed)
        bad = [r[0] for r in rows if r[1]]
        assert not bad, f'seed {seed}: false slip at t = {bad[0]:.2f} s'


def pulses(t, a=7.0, on=0.2):
    """Stand 1 s, +a to 5 m/s, then speed-controller-like pulses: +a for `on` s, -a for `on` s, ..."""
    if t < 1.0:
        return 0.0
    t1 = 1.0 + 5.0 / a
    if t < t1:
        return a * (t - 1.0)
    ph = (t - t1) % (2 * on)
    return 5.0 + (a * ph if ph < on else a * (2 * on - ph))


def test_no_false_alarm_hard_pulses():
    """+-7 m/s^2 acceleration / braking pulses of 0.2 s, 5 seeds: never on. This is where the lag
    compensation matters (the IMU is ~30 ms later than the wheel): without it 28 false episodes.
    Known limit: +-9.5 m/s^2 (the simulator's maximum) reversing every 0.15 s still gives false alarms --
    the 40 ms IMU smoothing flattens such fast changes, no time shift undoes that. The recorded racing
    runs never drive like that (0 false alarms in 30 clean runs)."""
    for seed in range(5):
        sc, rows = drive(10.0, prof=pulses, seed=seed)
        assert sc.episodes == 0, f'seed {seed}: {sc.episodes} false slip episodes'


def test_start_while_moving():
    """The node (re)starts while the car already drives 4 m/s: the observer starts at the wheel, no alarm."""
    sc, rows = drive(5.0, prof=lambda t: 4.0)
    assert not any(r[1] for r in rows), 'false slip at start-up'


def test_slow_slip_not_detected():
    """A +0.10 m/s wheel error (x 1.025 at 4 m/s) is below r0 = 0.20: not detected (known limit; the
    complementary pull absorbs it, AMCL / the speed scale k have to handle it)."""
    sc, rows = drive(10.0, fault=(5.0, 9.0, 1.025))
    assert not any(r[1] for r in rows), 'a slip below the threshold was flagged'


def test_imu_gap_resyncs():
    """An 80 ms IMU gap (> gap 50 ms) during a slip: flag off ('IMU gap'), observer re-synced to the wheel."""
    sc, rows = drive(8.0, fault=(5.0, 6.0, 1.15), imu_gap=(5.5, 5.58))
    assert any(r[1] for r in rows if 5.0 <= r[0] < 5.5), 'slip not detected before the gap'
    evs = [e for e in sc.events if not e[1]]
    assert evs and 'IMU gap' in evs[0][2], f'first OFF event: {evs[:1]}'
    after = [r for r in rows if 5.6 <= r[0] < 5.9]
    assert all(abs(r[2] - r[4]) < 0.25 for r in after), 'observer not re-synced to the wheel'


def test_imu_gap_no_false_alarm():
    """A clean run with an 80 ms IMU gap while accelerating: no false alarm afterwards."""
    sc, rows = drive(8.0, imu_gap=(1.8, 1.88))
    assert not any(r[1] for r in rows), 'false slip after an IMU gap'
    assert sc.resyncs == 1, f'{sc.resyncs} re-syncs (1 expected)'


def test_simulator_gate():
    """Simulator artifact: the wheel dips x 0.70 for 0.5 s with the IMU unaware. With the SimGate fed
    from /sim/wheel_ideal: never on. Without it the same dip IS flagged (so the test means something)."""
    sc, rows = drive(8.0, dip=(5.0, 5.5, 0.70), gate=SimGate())
    assert not any(r[1] for r in rows), 'simulator artifact flagged as slip despite the gate'
    assert sc.gated_n >= 20, f'only {sc.gated_n} samples gated'
    sc2, rows2 = drive(8.0, dip=(5.0, 5.5, 0.70))
    assert any(r[1] for r in rows2), 'without the gate the dip should look like a skid'


def test_gated_flag():
    """The same dip marked gated=True by the caller (offline use): never on."""
    sc, rows = drive(8.0, dip=(5.0, 5.5, 0.70), mark_gated=True)
    assert not any(r[1] for r in rows)


def test_real_fault_not_gated():
    """In the simulator a REAL fault (only the measured wheel is wrong, /sim/wheel_ideal is fine) must
    still be detected with the gate active."""
    sc, rows = drive(8.0, fault=(5.0, 6.0, 1.15), gate=SimGate())
    on = first_on(rows, 5.0)
    assert on is not None and on - 5.0 <= 0.080


def test_bias_learned_at_standstill():
    """Parked 6.5 s with an accelerometer bias of 0.15 m/s^2: the bias is learned (time constant 2 s),
    the observer stays at 0, no alarm."""
    sc, rows = drive(6.5, prof=lambda t: 0.0, bias=0.15)
    assert abs(sc.bias - 0.15) < 0.01, f'bias {sc.bias:.3f} (0.15 expected)'
    assert not any(r[1] for r in rows), 'alarm while parked'
    # the first stop_s (0.3 s) the check cannot know yet that the car is parked -> v_obs creeps a little
    assert max(abs(r[2]) for r in rows if r[0] > 0.4) < 1e-9, 'observer not held at 0 while parked'


def test_bias_tracked_while_driving():
    """The car never stops (started while driving), the IMU bias is 0.10 m/s^2 and was never learned:
    ki_bias learns it from the wheel while driving; a wheelspin at 12 s then clears normally."""
    sc, rows = drive(14.0, prof=lambda t: 4.0, bias=0.10, fault=(12.0, 13.0, 1.15))
    b_before = sc.bias
    assert abs(b_before - 0.10) < 0.02, f'bias {b_before:.3f} (0.10 expected)'
    off = first_off(rows, 13.0)
    assert off is not None and off - 13.0 <= 0.350, 'flag stuck after the spin'


def test_spin_soon_after_bias_change():
    """A wheelspin only 3 s after driving off with an unlearned 0.10 m/s^2 bias: the driving bias learning
    has not converged yet (~6 s), so the IMU-only speed drifts during the spin; the OFF threshold that
    grows with the slip's duration still lets the flag clear (without it: stuck until max_on)."""
    sc, rows = drive(6.5, prof=lambda t: 4.0, bias=0.10, fault=(3.0, 4.0, 1.15))
    off = first_off(rows, 4.0)
    assert off is not None and off - 4.0 <= 0.350, f'flag stuck after the spin (off at {off})'


def test_bias_change_while_driving_old_rule_sticks():
    """The same with the prototype rule (ki_bias 0, r_off_rate 0): the flag stays on until max_on --
    the weakness the two additions fix (keeps the comparison honest)."""
    sc, rows = drive(14.0, prof=lambda t: 4.0, bias=0.10, fault=(9.0, 10.0, 1.15),
                     sc=SlipCheck(ki_bias=0.0, r_off_rate=0.0))
    offs = [e for e in sc.events if not e[1]]
    assert offs and 'max_on' in offs[0][2], f'OFF events: {offs}'


def test_max_on():
    """A 5 s 'slip' (longer than max_on 3 s): given up after ~3 s, re-synced to the wheel, then off."""
    sc, rows = drive(12.0, fault=(5.0, 10.0, 1.15))
    offs = [e for e in sc.events if not e[1]]
    assert offs and 'max_on' in offs[0][2], f'OFF events: {offs}'
    assert 7.9 <= offs[0][0] <= 8.2, f'gave up at {offs[0][0]:.2f} s (~8.0 expected)'


def test_sim_gate_window():
    g = SimGate(ratio=0.97, min_speed=0.5, hold_s=0.3)
    g.update(1.00, 3.0, 4.0)            # dip
    g.update(1.02, 0.3, 0.4)            # slow: no dip (below min_speed)
    g.update(1.04, 3.95, 4.0)           # no dip
    assert g.gated(1.00) and g.gated(1.25) and not g.gated(1.31) and not g.gated(0.99)


def test_needs_a_gap():
    try:
        SlipCheck(r0=0.1, r_off=0.1)
    except ValueError:
        return
    raise AssertionError('no ValueError for r_off == r0')


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
