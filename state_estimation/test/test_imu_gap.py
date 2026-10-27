#!/usr/bin/env python3
"""Tests for imu_gap.py (ImuGap: IMU-dropout detection, wheel + steering yaw rate, validation of the returning gyro;
Fix 2, 2026-10-27). No ROS, no numpy. Run: python3 test_imu_gap.py   (or pytest)"""
import math
import os
import sys

try:
    from state_estimation.imu_gap import GAP, OK, RECOVERING, ImuGap
except ImportError:                                  # run from the source tree without ROS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'state_estimation'))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from imu_gap import GAP, OK, RECOVERING, ImuGap

L = 0.3302


def feed(g, T, v=4.0, steer=0.1, drop=None, gyro=None, t0=0.0):
    """IMU 100 Hz (none inside drop = (t_start, t_end)), wheel 50 Hz (5 ms offset). gyro(t) -> yaw rate the IMU
    reports (default: the true kinematic yaw rate). Returns [(t, kind, state, trusted_or_None)]."""
    true_r = v * math.tan(steer) / L
    gyro = gyro or (lambda t: true_r)
    ev = []
    for i in range(int(round(T / 0.005)) + 1):
        t = t0 + i * 0.005
        if i % 2 == 0 and not (drop and drop[0] <= t < drop[1]):
            ok = g.imu(t, gyro(t))
            ev.append((t, 'imu', g.state, ok))
        if i % 4 == 1:
            g.wheel(t, v, steer)
            ev.append((t, 'wheel', g.state, None))
    return ev


def test_no_gap_with_regular_imu():
    g = ImuGap()
    ev = feed(g, 5.0)
    assert all(e[2] == OK for e in ev) and g.episodes == 0


def test_gap_detected_after_gap_s():
    """IMU silent from 1.0 s: GAP at the first wheel sample more than 25 ms after the last IMU sample (<= 45 ms)."""
    g = ImuGap()
    ev = feed(g, 1.5, drop=(1.0, 9.0))
    first = next(e[0] for e in ev if e[2] == GAP)
    last_imu = max(e[0] for e in ev if e[1] == 'imu')
    assert 0.025 < first - last_imu <= 0.045 + 1e-9, f'detected {1000 * (first - last_imu):.0f} ms after the last sample'


def test_one_missing_sample_is_no_gap():
    """One lost IMU message (20 ms spacing) is not a dropout."""
    g = ImuGap()
    feed(g, 2.0, drop=(1.0, 1.01))
    assert g.episodes == 0


def test_model_kinematic_and_understeer():
    g = ImuGap()
    assert abs(g.model_r(4.0, 0.1) - 4.0 * math.tan(0.1) / L) < 1e-12
    g2 = ImuGap(kus=0.01)
    assert g2.model_r(6.0, 0.1) < g.model_r(6.0, 0.1), 'understeer must reduce the yaw rate at speed'
    assert g.model_r(0.0, 0.3) == 0.0


def test_offset_fades_from_the_last_gyro():
    """The model is 0.3 rad/s off at the start: yaw_rate starts at the last gyro and ends at the model."""
    g = ImuGap()
    true_r = 4.0 * math.tan(0.1) / L
    feed(g, 1.2, drop=(1.0, 9.0), gyro=lambda t: true_r + 0.3)
    assert g.state == GAP
    assert abs(g.yaw_rate(g.t_gap0) - (true_r + 0.3)) < 1e-9
    assert abs(g.yaw_rate(g.t_gap0 + 1.0) - true_r) < 0.3 * math.exp(-5) + 1e-9


def test_heading_through_a_1s_gap_in_a_corner():
    """Integrating yaw_rate through a 1 s gap in a steady corner (truth = the model) gives the true heading change."""
    g = ImuGap()
    feed(g, 1.0, drop=(1.0, 9.0))
    t, psi, dt = 1.0, 0.0, 0.02
    feed(g, 0.06, drop=(0.0, 99.0), t0=1.0)
    while t < 2.0 - 1e-9:
        psi += g.yaw_rate(t) * dt
        t += dt
    true = 4.0 * math.tan(0.1) / L * 1.0
    assert abs(math.degrees(psi - true)) < 0.5, f'{math.degrees(psi):.2f} vs {math.degrees(true):.2f} deg'


def test_recovery_needs_rec_n_consistent_samples():
    """The returning gyro is not trusted before rec_n (10) consistent samples; then OK and trusted."""
    g = ImuGap(rec_n=10)
    ev = feed(g, 2.0, drop=(1.0, 1.3))
    back = [e for e in ev if e[1] == 'imu' and e[0] >= 1.3]
    assert [e[3] for e in back[:9]] == [False] * 9, 'gyro trusted too early'
    assert back[9][3] is True and back[9][2] == OK
    kinds = [e[0] for e in g.events]
    assert kinds == ['gap', 'back', 'valid'], kinds
    assert abs(g.events[2][2] - 0.09) < 0.011, f'recovery took {g.events[2][2]:.3f} s'


def test_garbage_gyro_never_trusted():
    """The IMU comes back but reports 0 while the car turns at 1.2 rad/s: never trusted, stays RECOVERING."""
    g = ImuGap()
    feed(g, 3.0, drop=(1.0, 1.3), gyro=lambda t: 0.0 if t >= 1.3 else 4.0 * math.tan(0.1) / L)
    assert g.state == RECOVERING and g.rejects > 100
    assert not any(e[0] == 'valid' for e in g.events)


def test_one_bad_sample_restarts_the_count():
    g = ImuGap(rec_n=10)
    true_r = 4.0 * math.tan(0.1) / L
    feed(g, 1.5, drop=(1.0, 1.3), gyro=lambda t: true_r + (2.0 if abs(t - 1.35) < 1e-6 else 0.0))
    valid = [e for e in g.events if e[0] == 'valid']
    assert valid and abs(valid[0][2] - 0.14) < 0.011, f'recovery {valid[0][2] if valid else None}'


def test_silent_again_while_recovering():
    """IMU back for 30 ms, then silent again: GAP again, and the untrusted gyro gives no offset."""
    g = ImuGap()
    feed(g, 1.5, drop=(1.0, 1.3), gyro=lambda t: 5.0 if t >= 1.3 else 0.0)
    g2 = ImuGap()
    ev = feed(g2, 2.0, drop=(1.0, 1.3))
    assert g2.state == OK
    g3 = ImuGap()
    true_r = 4.0 * math.tan(0.1) / L
    feed(g3, 1.33, drop=(1.0, 1.3), gyro=lambda t: true_r + 0.3)    # model 0.3 off: the first gap HAS an offset
    assert abs(g3.off0 - 0.3) < 1e-9
    feed(g3, 0.2, drop=(0.0, 99.0), t0=1.335)
    assert g3.state == GAP and g3.off0 == 0.0, f'offset {g3.off0} from an untrusted gyro'
    assert [e[0] for e in g3.events] == ['gap', 'back', 'gap']


def test_gap_len_and_stats():
    g = ImuGap()
    feed(g, 3.0, drop=(1.0, 1.5))
    feed(g, 2.0, drop=(3.5, 3.7), t0=3.0)         # drop times are absolute
    assert g.episodes == 2 and abs(g.longest - 0.51) < 0.011, (g.episodes, g.longest)
    assert g.gap_len(10.0) == 0.0


def process(g, msgs, v=0.0, steer=0.0, gyro=0.0):
    """msgs = [(rx, stamp, 'imu'|'wheel')], handled in RECEIVE order (like an executor). Returns the states seen."""
    seen = []
    for rx, t, kind in sorted(msgs):
        if kind == 'imu':
            g.imu(t, gyro, rx=rx)
        else:
            g.wheel(t, v, steer, rx=rx)
        seen.append(g.state)
    return seen


def stream(T, imu_rx, wheel_rx, drop=None):
    """IMU 100 Hz / wheel 50 Hz stamps over T s; receive time = imu_rx(stamp) / wheel_rx(stamp)."""
    m = [(imu_rx(i * 0.01), i * 0.01, 'imu') for i in range(int(T / 0.01)) if not (drop and drop[0] <= i * 0.01 < drop[1])]
    m += [(wheel_rx(0.005 + j * 0.02), 0.005 + j * 0.02, 'wheel') for j in range(int(T / 0.02))]
    return m


def test_backlog_is_no_dropout():
    """Start-up backlog (job 207): the IMU is handled up to 0.25 s late (its queue falls behind the wheel's), but it
    keeps arriving. The stamps alone say 'dropout'; with receive times: no dropout."""
    lag = lambda t: 0.25 * min(1.0, t / 0.5) if t < 1.0 else max(0.0, 0.25 - 0.5 * (t - 1.0))   # noqa: E731
    msgs = stream(2.0, imu_rx=lambda t: t + lag(t) + 0.002, wheel_rx=lambda t: t + 0.002)
    g = ImuGap()
    process(g, msgs)
    assert g.episodes == 0, f'{g.episodes} false dropouts with receive times'
    g_old = ImuGap()
    for rx, t, kind in sorted(msgs):                  # the same without receive times = job 207's false dropouts
        g_old.imu(t, 0.0) if kind == 'imu' else g_old.wheel(t, 0.0, 0.0)
    assert g_old.episodes > 0, 'the scenario should fool the stamp-only rule'


def stall_release(order):
    """IMU 100 Hz + wheel 50 Hz; nothing handled from 1.0 to 1.16 s (busy CPU), then the queued messages are handled
    'roundrobin' (one per topic per executor cycle, IMU first -- what a single-threaded executor does) or 'wheel_first'
    (every queued wheel message before any queued IMU message: the worst order). Returns [(rx, stamp, kind)]."""
    msgs = [(i * 0.01, 'imu') for i in range(200)] + [(0.005 + j * 0.02, 'wheel') for j in range(100)]
    msgs.sort()
    before = [(t + 0.002, t, k) for t, k in msgs if t < 0.9975]
    after = [(t + 0.002, t, k) for t, k in msgs if t >= 1.1625]
    q_imu = [t for t, k in msgs if 0.9975 <= t < 1.1625 and k == 'imu']
    q_whl = [t for t, k in msgs if 0.9975 <= t < 1.1625 and k == 'wheel']
    rx, out = 1.16, []
    if order == 'roundrobin':
        while q_imu or q_whl:
            for q, k in ((q_imu, 'imu'), (q_whl, 'wheel')):
                if q:
                    rx += 0.0002
                    out.append((rx, q.pop(0), k))
    else:
        for q, k in ((q_whl, 'wheel'), (q_imu, 'imu')):
            for t in q:
                rx += 0.0002
                out.append((rx, t, k))
    return before + out + [(max(r, rx + 0.001), t, k) for r, t, k in after]


def test_process_stall_is_no_dropout():
    """Nothing is handled for 160 ms (busy CPU), then the queues drain the way an executor drains them: no dropout."""
    g = ImuGap()
    process(g, stall_release('roundrobin'))
    assert g.episodes == 0, f'{g.episodes} false dropouts after a process stall'


def test_stall_old_wheel_messages_prove_nothing():
    """KNOWN LIMIT, worst order (all queued wheel messages before the queued IMU messages): the OLD queued wheel
    messages must not declare a dropout (they are > gap_s old); only the newest one (fresh, and no IMU handled for
    160 ms of real time -- indistinguishable from a real dropout) does, and the IMU right after it ends it."""
    g = ImuGap()
    first_gap = None
    for rx, t, kind in sorted(stall_release('wheel_first')):
        g.imu(t, 0.0, rx=rx) if kind == 'imu' else g.wheel(t, 0.0, 0.0, rx=rx)
        if g.state == GAP and first_gap is None:
            first_gap = (rx, t)
    assert first_gap is not None and first_gap[0] - first_gap[1] <= g.gap_s, \
        f'a dropout was declared by an old queued wheel message (age {1000 * (first_gap[0] - first_gap[1]):.0f} ms)'
    assert g.episodes == 1 and g.state == OK


def test_real_dropout_with_receive_times():
    """A real dropout (no IMU messages 1.0-1.3 s) is still detected when receive times are given."""
    g = ImuGap()
    seen = process(g, stream(2.0, imu_rx=lambda t: t + 0.002, wheel_rx=lambda t: t + 0.002, drop=(1.0, 1.3)))
    assert g.episodes == 1 and GAP in seen and g.state == OK


def test_bad_parameters():
    for kw in ({'gap_s': 0.0}, {'wheelbase': 0.0}, {'tau_b': 0.0}, {'rec_n': 0}):
        try:
            ImuGap(**kw)
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
