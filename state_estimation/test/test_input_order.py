#!/usr/bin/env python3
"""Tests for input_order.py (stamp-ordered inputs, estimator-stall fix A, 2026-11-07). No ROS.
Run: python3 test_input_order.py

The core test replays what rclpy's SingleThreadedExecutor does in a backlog: per round it takes ONE ready
message from each subscription (IMU first: it is created first in state_pipeline), each callback costs
wall time, the queues are KEEP_LAST (depth 100 / 50: the oldest is dropped). The consumer is ekf_node's
behaviour: meas_cb stores the newest speed, imu_cb predicts with it -> we record the speed used at every
IMU stamp and compare it with the true speed AT THAT STAMP.
"""
import collections
import sys

try:
    from state_estimation.input_order import InputOrder
except ImportError:                                    # run from the staging folder
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from input_order import InputOrder

RESULTS = []


def check(name, cond, info=''):
    RESULTS.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f'   [{info}]' if info else ''))


def raises(f):
    try:
        f()
    except ValueError:
        return True
    return False


def v_true(t):
    """racing-like profile: 6 m/s, brake at 5 m/s^2 to 2.1, accelerate at 4 m/s^2 back, period 3 s"""
    p = t % 3.0
    if p < 0.5:
        return 6.0
    if p < 0.5 + 3.9 / 5.0:
        return 6.0 - 5.0 * (p - 0.5)
    if p < 1.6:
        return 2.1
    return min(6.0, 2.1 + 4.0 * (p - 1.6))


def replay(order, T=6.0, imu_hz=100.0, wheel_hz=50.0, imu_depth=100, wheel_depth=50,
           lag=(1.0, 4.0, 0.012), imu_silent=None, transport=0.001):
    """returns dict: err = |speed used - true speed| at every processed IMU stamp, wheel stamps in processing
    order, per-wheel (arrival wall, release wall), IMU stamps processed"""
    imu = [(k / imu_hz, 'imu') for k in range(int(T * imu_hz))]
    if imu_silent:
        imu = [m for m in imu if not (imu_silent[0] <= m[0] < imu_silent[1])]
    whl = [(k / wheel_hz, 'whl') for k in range(int(T * wheel_hz))]
    srcs = {'imu': imu, 'whl': whl}
    nxt = {'imu': 0, 'whl': 0}
    q = {'imu': collections.deque(), 'whl': collections.deque()}
    depth = {'imu': imu_depth, 'whl': wheel_depth}
    now = 0.0
    v_used = None
    out = {'err': [], 'wheel_order': [], 'hold': [], 'imu_stamps': [], 'dropped': 0}
    arrived = {}

    def deliver(stamp):                        # = ekf_node.meas_cb: stores the newest speed
        nonlocal v_used
        v_used = v_true(stamp)
        out['wheel_order'].append(stamp)
        out['hold'].append(now - arrived[stamp])

    def cost_imu(t_wall):
        if lag and lag[0] <= t_wall < lag[1]:
            return lag[2]
        return 0.0006

    while True:
        for k in ('imu', 'whl'):               # arrivals up to now (KEEP_LAST: the oldest is dropped)
            while nxt[k] < len(srcs[k]) and srcs[k][nxt[k]][0] + transport <= now:
                st = srcs[k][nxt[k]][0]
                q[k].append(st)
                if k == 'whl':
                    arrived[st] = st + transport
                nxt[k] += 1
                if len(q[k]) > depth[k]:
                    q[k].popleft()
                    out['dropped'] += 1
        if not q['imu'] and not q['whl']:
            pend = [srcs[k][nxt[k]][0] + transport for k in srcs if nxt[k] < len(srcs[k])]
            if not pend:
                break
            now = max(now, min(pend))
            continue
        if q['imu']:                           # one IMU callback
            st = q['imu'].popleft()
            if order is not None:
                for w in order.imu(st, now):
                    deliver(w)
            out['imu_stamps'].append(st)
            if v_used is not None:
                out['err'].append(abs(v_used - v_true(st)))
            now += cost_imu(now)
        if q['whl']:                           # one wheel callback
            st = q['whl'].popleft()
            if order is None:
                deliver(st)
            else:
                for w in order.wheel(st, st, now):
                    deliver(w)
            now += 0.0003
    return out


def main():
    check('bad parameters refused', raises(lambda: InputOrder(wait_s=0.0)) and raises(lambda: InputOrder(max_held=0)))

    o = InputOrder()
    check('start-up (no IMU yet): wheel passes at once', o.wheel(1.00, 'a', 0.0) == ['a'])
    o.imu(1.00, 0.0)
    check('wheel older than the IMU done: at once', o.wheel(0.99, 'b', 0.001) == ['b'])
    check('wheel ahead of the IMU: waits', o.wheel(1.02, 'c', 0.002) == [] and len(o.held) == 1)
    check('a later, OLDER wheel that is still ahead: kept in stamp order',
          o.wheel(1.015, 'd', 0.003) == [] and [h[2] for h in o.held] == ['d', 'c'])
    check('IMU 1.01: releases nothing yet', o.imu(1.01, 0.004) == [])
    check('IMU 1.02: releases d then c (stamp order), before the IMU', o.imu(1.02, 0.005) == ['d', 'c'] and not o.held)
    o.wheel(1.05, 'e', 0.006)
    check('IMU silent for > wait_s: the next wheel releases everything (dropout)',
          o.wheel(1.07, 'f', 0.006 + 0.030) == ['e', 'f'] and o.n_timeout == 1)
    o = InputOrder(max_held=3)
    o.imu(0.0, 0.0)
    rel = []
    for k in range(5):
        rel += o.wheel(0.1 + 0.01 * k, k, 0.001)
    check('max_held: the oldest are released first', rel == [0, 1] and len(o.held) == 3 and o.n_overflow == 2)
    check('report resets the counters', 'waited 5' in o.report() and o.n_wheel == 0 and o.n_held == 0)

    # --- the executor replay -------------------------------------------------------------------------------
    clean_off = replay(None, lag=None)
    clean_on = replay(InputOrder(), lag=None)
    e0, e1 = max(clean_off['err']), max(clean_on['err'])
    check('no lag: arrival order is already fine (speed error <= 0.11 m/s = 20 ms of 5 m/s^2 + a step)',
          e0 <= 0.11, f'{e0:.3f}')
    check('no lag: ordering changes nothing worth seeing', e1 <= 0.11 and abs(e1 - e0) < 0.02, f'{e1:.3f}')
    check('no lag: no wheel message waits longer than one IMU period (10 ms)',
          max(clean_on['hold']) <= 0.0105, f"{1000 * max(clean_on['hold']):.1f} ms")

    off = replay(None)
    on = replay(InputOrder())
    eo, en = max(off['err']), max(on['err'])
    mo, mn = sum(off['err']) / len(off['err']), sum(on['err']) / len(on['err'])
    check('F07-like lag, arrival order: the bug is reproduced (speed error > 1.5 m/s)', eo > 1.5,
          f'max {eo:.2f}, mean {mo:.3f} m/s')
    check('F07-like lag, stamp order: speed error <= 0.11 m/s', en <= 0.11, f'max {en:.3f}, mean {mn:.3f} m/s')
    check('stamp order: wheel messages processed in stamp order',
          all(b >= a for a, b in zip(on['wheel_order'], on['wheel_order'][1:])))
    check('stamp order: every IMU stamp still processed (none skipped by the ordering)',
          len(on['imu_stamps']) == len(off['imu_stamps']))
    check('the replay really fell behind (> 0.5 s of wheel waiting)', max(on['hold']) > 0.5,
          f"{max(on['hold']):.2f} s")

    o = InputOrder(wait_s=0.025)
    drop = replay(o, lag=None, imu_silent=(2.0, 3.0))
    wheel_in_gap = [h for s, h in zip(drop['wheel_order'], drop['hold']) if 2.0 <= s < 3.0]
    check('real IMU dropout (1 s): wheel messages still flow (released by IMU silence)',
          len(wheel_in_gap) >= 48 and o.n_timeout > 0, f'{len(wheel_in_gap)} wheel msgs, {o.n_timeout} timeouts')
    check('real IMU dropout: each waits at most wait_s + one wheel period (45 ms)', max(wheel_in_gap) <= 0.0455,
          f'{1000 * max(wheel_in_gap):.1f} ms')

    car_off = replay(None, wheel_hz=100.0)
    car_on = replay(InputOrder(), wheel_hz=100.0)
    co, cn = max(car_off['err']), max(car_on['err'])
    check('car rates (IMU + wheel 100 Hz, wheel depth 50), lag: stamp order <= 0.06 m/s', cn <= 0.06,
          f'arrival order max {co:.2f} (dropped {car_off["dropped"]} msgs) -> stamp order {cn:.3f} m/s')

    print(f'\n{sum(RESULTS)} of {len(RESULTS)} passed')
    return 0 if all(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
