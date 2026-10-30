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


def late_wheel_stream(stale_s=0.09, bias=0.08, seed=3):
    """Job 208 G2 (2026-10-27): the estimator got NO wheel messages for the first seconds (its subscription connected
    late) while the IMU ran normally; the first wheel messages then arrived in a burst, ~90 ms stale, just as the car
    launched at 9.5 m/s^2. Events are handled in RECEIVE order. Returns (SlipCheck, rows (t, on, v_obs, v_true))."""
    rnd = random.Random(seed)
    prof = lambda t: 0.0 if t < 2.6 else min(6.0, 9.5 * (t - 2.6))      # noqa: E731
    ev, a_lp, v_prev = [], 0.0, 0.0
    for i in range(1, 5001):                                             # 1 ms truth steps, 5 s
        t = i * 0.001
        v = prof(t)
        a_lp += 0.001 / 0.040 * ((v - v_prev) / 0.001 - a_lp)
        v_prev = v
        if i % 10 == 0:
            ev.append((t + 0.002, 'imu', t, a_lp + bias + rnd.gauss(0.0, 0.05)))
        if i % 20 == 5:
            vt = prof(t - 0.010)
            vw = vt + (rnd.gauss(0.0, 0.03) if vt > 1e-3 else 0.0)
            t_first = 2.6 + stale_s
            if t < 2.6:
                continue                                                 # never delivered (not connected yet)
            rx = t_first + (t - 2.6) * 0.01 + 0.002 if t < t_first else t + 0.002   # burst, then normal
            ev.append((rx, 'wheel', t, vw))
    sc, rows = SlipCheck(), []
    for rx, kind, t, val in sorted(ev):
        if kind == 'imu':
            sc.imu(t, val)
        else:
            sc.wheel(t, val)
            rows.append((t, sc.on, sc.observer_speed, prof(t)))
    return sc, rows


def test_wheel_stream_starts_late():
    """The observer must not start from a STALE wheel sample (job 208: started 0.87 m/s low -> a false 3 s 'slip'
    at launch and the estimate was lost). It waits for a current one: no false slip, observer right."""
    sc, rows = late_wheel_stream()
    assert sc.episodes == 0, f'{sc.episodes} false slip episodes at launch'
    late = [r for r in rows if r[0] > 3.5]          # after the 9.5 m/s^2 launch (ends 3.23 s); during it the observer
    assert max(abs(r[2] - r[3]) for r in late) < 0.1, 'observer speed off after the late start'   # lags ~a * 0.05 s


def test_wheel_gap_while_driving_resyncs():
    """The wheel stream stops for 0.5 s while driving and comes back with a stale burst: re-sync, no false slip."""
    rnd = random.Random(5)
    sc = SlipCheck()
    prof = lambda t: min(5.0, 4.0 * max(0.0, t - 1.0))                   # noqa: E731
    ev, a_lp, v_prev = [], 0.0, 0.0
    for i in range(1, 6001):
        t = i * 0.001
        v = prof(t)
        a_lp += 0.001 / 0.040 * ((v - v_prev) / 0.001 - a_lp)
        v_prev = v
        if i % 10 == 0:
            ev.append((t + 0.002, 'imu', t, a_lp + 0.08 + rnd.gauss(0.0, 0.05)))
        if i % 20 == 5:
            vt = prof(t - 0.010)
            vw = vt + (rnd.gauss(0.0, 0.03) if vt > 1e-3 else 0.0)
            if 1.5 <= t < 2.0:
                rx = 2.09 + (t - 1.5) * 0.01                              # held back, then a burst 90 ms late
            else:
                rx = t + 0.002
            ev.append((rx, 'wheel', t, vw))
    for rx, kind, t, val in sorted(ev):
        sc.imu(t, val) if kind == 'imu' else sc.wheel(t, val)
    assert sc.episodes == 0, f'{sc.episodes} false slip episodes after a wheel gap'


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


# ------------------------------------------------------------------ wheel data lost (Fix 4, 2026-10-30)
def drive_lost(T, prof, lost, call_lost=True, bias=0.08, fault=None, imu_gap=None, seed=7, bias_step=None):
    """Like drive(), but NO wheel samples inside lost = (t0, t1); like ekf_node, wheel_lost(t) is called at the first
    IMU sample more than 50 ms after the last wheel sample (call_lost False = the old code). Returns (sc, rows) with
    rows (t, v_obs, v_true) at every IMU sample, and the slip ON / OFF events."""
    rnd = random.Random(seed)
    sc = SlipCheck()
    a_lp, v_prev, rows, t_w = 0.0, prof(0.0), [], None
    h = 0.001
    for i in range(1, int(round(T / h)) + 1):
        t = i * h
        v = prof(t)
        a = (v - v_prev) / h
        v_prev = v
        a_lp += h / 0.040 * (a - a_lp)
        if i % 10 == 0 and not (imu_gap and imu_gap[0] <= t < imu_gap[1]):
            if call_lost and t_w is not None and t - t_w > 0.05:
                sc.wheel_lost(t)
            db = bias_step[2] if bias_step and bias_step[0] <= t < bias_step[1] else 0.0
            sc.imu(t, a_lp + bias + db + rnd.gauss(0.0, 0.05))
            rows.append((t, sc.observer_speed, v))
        if i % 20 == 5 and not (lost[0] <= t < lost[1]):
            vt = prof(t - 0.010)
            f = fault[2] if fault and fault[0] <= t < fault[1] else 1.0
            vw = vt * f + (rnd.gauss(0.0, 0.03) if abs(vt * f) > 1e-3 else 0.0)
            sc.wheel(t, vw)
            t_w = t
    return sc, rows


def dist_err(rows, t0, t1):
    """Distance from the observer speed minus the true distance over [t0, t1] [m]."""
    sel = [r for r in rows if t0 <= r[0] <= t1]
    return sum((b[1] - b[2] + a[1] - a[2]) / 2 * (b[0] - a[0]) for a, b in zip(sel, sel[1:]))


def test_wheel_lost_while_parked_then_drive():
    """The wheel data dies while PARKED, then the car sets off (3 m/s^2): the observer must follow the IMU, not stay
    at 0 (the old code: still 'parked' -> 0 m/s). Bias frozen meanwhile (the accelerations are not bias)."""
    prof = lambda t: profile(t, stand=2.0, accel=3.0, cruise=4.0)        # noqa: E731
    sc_new, rows = drive_lost(3.5, prof, lost=(1.5, 9.0))
    v_new = rows[-1][1]
    sc_old, rows_old = drive_lost(3.5, prof, lost=(1.5, 9.0), call_lost=False)
    assert abs(v_new - prof(3.5)) < 0.3, f'observer {v_new:.2f} m/s, true {prof(3.5):.2f}'
    assert abs(rows_old[-1][1]) < 0.05, f'the old code should stay parked (got {rows_old[-1][1]:.2f})'
    sc_at_loss, _ = drive_lost(1.56, prof, lost=(1.5, 9.0))           # the same run, stopped just after the loss
    assert abs(sc_new.bias - sc_at_loss.bias) < 1e-12, (f'bias changed during the dropout: {sc_at_loss.bias:.4f} -> '
                                                         f'{sc_new.bias:.4f}')
    assert abs(sc_old.bias - sc_at_loss.bias) > 0.5, 'the old code should learn the set-off as accelerometer bias'
    assert abs(dist_err(rows, 2.0, 3.5)) < 0.15, f'distance error {dist_err(rows, 2.0, 3.5):.2f} m'


def test_wheel_lost_while_braking():
    """1 s without wheel data while braking 6 -> 2 m/s (F11b-like): distance error within the model's 1 sigma
    (wheel_gap.WheelGap.sigma_d(1 s) = 0.19 m); held speed: > 1 m. Most of what remains is this synthetic IMU's 40 ms
    smoothing: 5 m/s^2 x 40 ms ~ 0.2 m/s behind for the 0.8 s of braking ~ 16 cm."""
    sc, rows = drive_lost(9.0, lively, lost=(6.0, 7.0))
    e = dist_err(rows, 6.0, 7.0)
    assert abs(e) < 0.19, f'distance error {e:.2f} m over the 1 s dropout'
    held = sum((6.0 - lively(t / 100.0)) * 0.01 for t in range(600, 700))
    assert held > 1.0, held                            # what holding 6 m/s would have cost


def test_wheel_back_resyncs_no_false_slip():
    """After the dropout the first wheel sample re-syncs the observer: no false WHEEL SLIP (job 221 F11c / F11d)."""
    for lost in ((6.0, 7.0), (2.0, 5.0), (0.5, 3.0)):
        sc, rows = drive_lost(12.0, lively, lost=lost)
        assert sc.episodes == 0, f'{sc.episodes} false slip episodes after a dropout {lost}'
        assert not sc.lost


def test_wheel_back_after_drift_resyncs():
    """A 2 s dropout during which the accelerometer bias jumps by 0.3 m/s^2 (a knock, warm-up): the IMU speed drifts
    0.6 m/s away. The returning wheel must RE-SYNC it -- deciding on it would be a false slip (r > r0 0.20 m/s)."""
    sc, rows = drive_lost(10.0, profile, lost=(5.0, 7.0), bias_step=(5.0, 7.0, 0.3))
    drift = next(r[1] - r[2] for r in rows if r[0] >= 6.95)
    assert abs(drift) > 0.4, f'the IMU speed should have drifted (got {drift:+.2f} m/s)'
    assert sc.episodes == 0, f'{sc.episodes} false slip episodes after a drifted dropout'
    after = [r for r in rows if 7.2 <= r[0] <= 7.5]
    assert max(abs(r[1] - r[2]) for r in after) < 0.15, 'observer not back on the wheel speed after the re-sync'


def test_wheel_back_old_code_false_slip():
    """The same without wheel_lost (the old code) after a dropout while parked and setting off: a false slip -- the
    reason for the re-sync."""
    sc, _ = drive_lost(4.0, lambda t: profile(t, stand=2.0), lost=(1.5, 3.0), call_lost=False)
    assert sc.episodes >= 1, 'expected the old false alarm'


def test_wheel_lost_ends_a_slip():
    """A wheelspin in progress when the wheel data dies: the flag ends ('wheel data lost'); the dropout is the EKF's
    and state_estimate's business from there."""
    sc, rows = drive_lost(8.0, profile, lost=(5.3, 6.5), fault=(5.0, 6.0, 1.15))
    ev = [e for e in sc.pop_events()]
    assert any((not on) and 'wheel data lost' in why for _, on, why, _ in ev), ev
    assert sc.episodes == 1


def test_imu_gap_during_wheel_loss_keeps_the_speed():
    """IMU AND wheel silent: the observer must not jump to the STALE wheel speed 6 m/s (that would undo the braking
    the IMU did see) -- it HOLDS its speed through the IMU gap. (It cannot know the braking it did not measure: with
    both sensors silent the estimate is LOST after 50 ms anyway, Fix 2.)"""
    sc, rows = drive_lost(9.0, lively, lost=(6.0, 7.5), imu_gap=(6.6, 6.7))
    v_before = [r[1] for r in rows if r[0] < 6.6][-1]
    v_after = next(r[1] for r in rows if r[0] >= 6.7)
    assert abs(v_after - v_before) < 0.1, f'observer {v_before:.2f} -> {v_after:.2f} m/s across the IMU gap'
    assert abs(v_after - 6.0) > 1.0, 'jumped back to the stale wheel speed 6 m/s'


def test_no_bias_learning_while_lost():
    """Parked with the wheel lost: stopped is never true -> the bias is not learned from whatever the IMU reads."""
    sc, _ = drive_lost(3.0, lambda t: 0.0, lost=(1.0, 9.0), bias=0.08)
    b1 = sc.bias
    sc2, _ = drive_lost(6.0, lambda t: 0.0, lost=(1.0, 9.0), bias=0.08)
    assert abs(sc2.bias - b1) < 1e-12, (b1, sc2.bias)


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
