#!/usr/bin/env python3
"""OFFLINE test of the simulated gyro models on recorded simulator data (no ROS needed).

Input: a raw recording folder (eval_logs/raw/<run>/): odom_raw.csv = every simulator
message at full rate (receive time, pose, yaw-rate state), imu_raw.csv = the real
fake_imu reading times (used for realistic tick timing).
Models     v1  pose difference since the previous reading (committed fake_imu)
           v2  the simulator's yaw-rate state (rejected 2026-10-02)
           v3  PauseProofRate from fake_imu.py; v3-norho = the same without rho
Scenarios  recorded  the simulator ran in real time
           pauses    the simulator freezes for 0.05 / 0.15 / 0.40 s in the three
                     sharpest corners (no new states, our sensor keeps ticking)
           slow      the simulator runs at 70 % speed for 3 s around the sharpest corner
Truth = what the simulated car really did in WALL time (frozen during a pause).
Second part: the WHEEL SPEED (fake_vehicle_sensors) the same way -- "rate-state" = the
simulator's speed state (current), v3 = PauseProofRate on the distance travelled.
Run:  python3 imu_model_offline.py <recording folder> [<recording folder> ...]
"""
import csv
import os
import sys

import numpy as np

try:                                     # installed package (tools/ in the workspace)
    from state_estimation.pause_proof import PauseProofRate
except ImportError:                      # next to pause_proof.py (staging folder)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pause_proof import PauseProofRate  # noqa: E402

STEP = 0.01          # simulator physics step [s]
FROZEN = 0.025       # no new state for longer than this = the car is frozen (truth)


def load(folder, speed=False):
    rows = list(csv.DictReader(open(os.path.join(folder, 'odom_raw.csv'))))
    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    t, x, y, yaw, wz, vx, vy = col('t_rx'), col('x'), col('y'), col('yaw'), col('wz'), col('vx'), col('vy')
    new = np.r_[True, (np.diff(x) != 0) | (np.diff(y) != 0) | (np.diff(yaw) != 0)]
    imu = list(csv.DictReader(open(os.path.join(folder, 'imu_raw.csv'))))
    tick_t = np.array([float(r['t_stamp']) for r in imu])
    if speed:
        # value = distance travelled along the path (signed by the driving direction), rate = speed state
        v = np.copysign(np.hypot(vx, vy), vx)[new]
        step = np.r_[0.0, np.hypot(np.diff(x[new]), np.diff(y[new]))] * np.where(v < 0, -1.0, 1.0)
        S = np.cumsum(step)
        return t[new], S, S, v, vx[new], tick_t
    return t[new], yaw[new], np.unwrap(yaw[new]), wz[new], vx[new], tick_t


def scenario(a, wz, kind):
    """Modified arrival times + truth windows [(start, end, rho)]."""
    a = a.copy()
    win = []
    if kind == 'pauses':
        picks = []
        for j in np.argsort(-np.abs(wz)):
            if 0 < j < len(a) - 1 and all(abs(a[j] - a[p]) > 5.0 for p in picks):
                picks.append(int(j))
            if len(picks) == 3:
                break
        for j, dur in zip(sorted(picks), [0.05, 0.15, 0.40]):
            start = a[j]                       # freezes right after state j arrived
            a[j + 1:] += dur
            win.append((start, start + dur + STEP, 0.0))
    elif kind == 'slow':
        j = int(np.argmax(np.abs(wz)))
        s0, length, f = a[j] - 1.5, 3.0, 0.7
        after, inside = a >= s0 + length, (a >= s0) & (a < s0 + length)
        a[after] += length / f - length
        a[inside] = s0 + (a[inside] - s0) / f
        win.append((s0, s0 + length / f, f))
    return a, win


def truth_heading(a, Y, t):
    """Wall-time heading: linear between states; after a long gap the next state's
    rotation happened in its last STEP only (the car was frozen before that)."""
    j = np.clip(np.searchsorted(a, t, side='right') - 1, 0, len(a) - 2)
    g = a[j + 1] - a[j]
    span = np.where(g <= FROZEN, g, STEP)
    w = np.clip((t - (a[j + 1] - span)) / np.maximum(span, 1e-9), 0.0, 1.0)
    return Y[j] + (Y[j + 1] - Y[j]) * w


def evaluate(a, win, ticks, yaw, Y0, wz, vx, models, angle=True):
    """Metrics per model for one timeline: arrival times a, truth windows, reading times."""
    latest = np.searchsorted(a, ticks, side='right') - 1
    Yt = truth_heading(a, Y0, ticks)
    rho_true = np.ones(len(ticks))
    for (s, e, r) in win:
        if r > 0:
            rho_true[(ticks >= s) & (ticks < e)] = r
    frozen = (ticks - a[latest]) > FROZEN
    r_true = wz[latest] * np.where(frozen, 0.0, rho_true)
    moving = np.abs(vx[latest]) > 0.05
    res = {}
    for m in models:
        out = np.zeros(len(ticks))
        if m == 'v1':
            out[1:] = np.diff(Y0[latest]) / np.diff(ticks)
        elif m in ('v2', 'rate-state'):
            out = wz[latest].copy()
        else:
            g = PauseProofRate(window_s=(0.25 if m == 'v3' else 0.0), angle=angle,
                               max_jump=(1.0 if angle else 0.5))
            k = 0
            for i, t in enumerate(ticks):
                while k < len(a) and a[k] <= t:
                    g.on_state(a[k], yaw[k], wz[k])
                    k += 1
                out[i] = g.sample(t)
        A = Yt[0] + np.r_[0.0, np.cumsum(out[1:] * np.diff(ticks))]
        herr = A - Yt
        rerr = (out - r_true)[moving]
        phantom = max((np.abs(herr[(ticks >= s) & (ticks < e + 0.5)]).max() for (s, e, r) in win if r == 0),
                      default=float('nan'))
        conv = np.degrees if angle else (lambda e: 100.0 * e)     # heading deg / distance cm
        res[m] = (np.sqrt(np.mean(rerr ** 2)), np.percentile(np.abs(rerr), 95),
                  conv(np.sqrt(np.mean(herr ** 2))), conv(np.abs(herr).max()),
                  conv(phantom), abs(conv(herr[-1])))
    return res


PHASES = [0.0, 0.002, 0.004, 0.006, 0.008]   # offsets of the reading clock vs the physics clock [s]


def run(folder):
    a0, yaw, Y0, wz, vx, tick_t = load(folder)
    tick_dt = np.diff(tick_t)
    print(f'\n##### {os.path.basename(folder.rstrip("/"))}: {len(a0)} simulator states, '
          f'{len(tick_t)} recorded readings')
    models = ['v1', 'v2', 'v3', 'v3-norho']
    groups = [('recorded, REAL reading times', 'recorded', None)] + \
             [(f'{k}, worst of {len(PHASES)} reading phases', k, PHASES) for k in ['recorded', 'pauses', 'slow']]
    for title, kind, phases in groups:
        a, win = scenario(a0, wz, kind)
        if phases is None:
            runs = [tick_t[(tick_t > a[0] + 0.05) & (tick_t < a[-1] - 0.02)]]
        else:
            # reading times: the recorded intervals, cycled over the (longer) timeline, shifted by a phase
            n = int((a[-1] - a[0] - 0.1) / 0.01)
            base = a[0] + 0.05 + np.cumsum(np.resize(tick_dt, n))
            runs = [(base + ph)[(base + ph) < a[-1] - 0.02] for ph in phases]
        all_res = [evaluate(a, win, ticks, yaw, Y0, wz, vx, models) for ticks in runs]
        res = {m: tuple(max(r[m][i] for r in all_res) for i in range(6)) for m in models}
        print(f'  -- {title}')
        print(f'     {"model":9s} {"rate err rms":>13s} {"rate err p95":>13s} {"heading err rms":>16s} '
              f'{"heading err max":>16s} {"max in/after pause":>19s} {"at end":>8s}')
        print(f'     {"":9s} {"[rad/s]":>13s} {"[rad/s]":>13s} {"[deg]":>16s} {"[deg]":>16s} {"[deg]":>19s} {"[deg]":>8s}')
        for m in models:
            r = res[m]
            print(f'     {m:9s} {r[0]:13.4f} {r[1]:13.4f} {r[2]:16.2f} {r[3]:16.2f} {r[4]:19.2f} {r[5]:8.2f}')


def run_speed(folder):
    a0, raw, S0, v, vx, tick_t = load(folder, speed=True)
    tick_dt = np.diff(tick_t)
    print(f'\n##### WHEEL SPEED, {os.path.basename(folder.rstrip("/"))}')
    models = ['rate-state', 'v3']
    for title, kind, phases in [('recorded, REAL reading times', 'recorded', None),
                                ('pauses, worst of 5 reading phases', 'pauses', PHASES),
                                ('slow, worst of 5 reading phases', 'slow', PHASES)]:
        a, win = scenario(a0, v, kind)
        if phases is None:
            runs = [tick_t[(tick_t > a[0] + 0.05) & (tick_t < a[-1] - 0.02)]]
        else:
            n = int((a[-1] - a[0] - 0.1) / 0.01)
            base = a[0] + 0.05 + np.cumsum(np.resize(tick_dt, n))
            runs = [(base + ph)[(base + ph) < a[-1] - 0.02] for ph in phases]
        all_res = [evaluate(a, win, ticks, raw, S0, v, vx, models, angle=False) for ticks in runs]
        res = {m: tuple(max(r[m][i] for r in all_res) for i in range(6)) for m in models}
        print(f'  -- {title}')
        print(f'     {"model":11s} {"speed err rms":>14s} {"speed err p95":>14s} {"distance err rms":>17s} '
              f'{"distance err max":>17s} {"max in/after pause":>19s} {"at end":>8s}')
        print(f'     {"":11s} {"[m/s]":>14s} {"[m/s]":>14s} {"[cm]":>17s} {"[cm]":>17s} {"[cm]":>19s} {"[cm]":>8s}')
        for m in models:
            r = res[m]
            print(f'     {m:11s} {r[0]:14.4f} {r[1]:14.4f} {r[2]:17.2f} {r[3]:17.2f} {r[4]:19.2f} {r[5]:8.2f}')


if __name__ == '__main__':
    for folder in sys.argv[1:]:
        run(folder)
    for folder in sys.argv[1:]:
        run_speed(folder)
