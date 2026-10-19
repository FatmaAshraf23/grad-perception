#!/usr/bin/env python3
"""wheel_delay_calib.py <recording> [...] -- how late is the wheel speed? (ekf_node wheel_delay_s; 2026-10-19)

WHY
    ekf_node's wheel-speed delay compensation (wheel_delay.py) needs tau_w = how much older the wheel speed
    is than its time stamp.  On the real car it depends on the STM32 code (an encoder speed = counts over a
    window, so it describes the window's MIDDLE) and on where the message gets its stamp.  No ground truth
    is needed: the IMU's forward acceleration a_x tells when the speed REALLY changed.
HOW
    If the wheel value with stamp t is the true speed at t - tau, then over a window of wheel messages
        k * (v_w[j] - v_w[i])  =  integral of (a_x - b) dt  over [t[i] - tau, t[j] - tau]
    (k = speed scale, b = accelerometer bias).  For every tau on a 1 ms grid, k and b are fitted by least
    squares; the tau with the smallest residual is the answer.  The information comes from the moments
    the acceleration CHANGES (start of a brake or of an acceleration) -> drive several hard accelerations
    and brakes (the usual laps do it), >= 1 min.
    The result is RELATIVE TO THE IMU: tau_w = tau_found + the IMU's own delay (its low-pass filter;
    MPU-6050 DLPF 42 Hz ~ 5 ms, 20 Hz ~ 9 ms -- datasheet), then set wheel_delay_s to that.
    SIMULATOR: fake_imu smooths a_x with a ~40 ms low-pass, so here tau_found ~ tau_w - 30..40 ms.
INPUT
    A folder with imu.csv (stamp, ax) and wheel.csv (stamp, speed) as written by bag_to_csv.py (runner
    tools), several folders are pooled.  --truth: the folder's truth.csv is used instead of a_x (simulator
    check: then tau_found is the wheel speed's delay against the TRUE time).
"""
import argparse
import csv
import os
import sys

import numpy as np


def load(path):
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        rows = [[float(v) for v in row] for row in r if row]
    a = np.array(rows)
    return {h: a[:, i] for i, h in enumerate(hdr)}


def segments(folder, use_truth):
    """-> (t_wheel, v_wheel, t_acc, a) of one recording"""
    wh = load(os.path.join(folder, 'wheel.csv'))
    if use_truth:
        tr = load(os.path.join(folder, 'truth.csv'))
        new = np.ones(len(tr['x']), bool)
        new[1:] = (np.diff(tr['x']) != 0) | (np.diff(tr['y']) != 0)
        t, v = tr['stamp'][new], np.hypot(tr['vx'], tr['vy'])[new]
        a = np.gradient(v, t)
        return wh['stamp'], wh['speed'], t, a
    im = load(os.path.join(folder, 'imu.csv'))
    return wh['stamp'], wh['speed'], im['stamp'], im['ax']


def cumint(t_acc, a):
    """cumulative integral of a (trapezoid) at the samples t_acc"""
    return np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(t_acc))])


def fit(recs, tau, span=5):
    """least squares  dv_w = (dI - b * dt) / k  over windows of `span` wheel intervals (the wheel speed's
    noise is in dv_w, so it is the measured side; the integrated a_x is nearly noise-free);
    -> (residual rms [m/s], k, b, number of windows)"""
    A, y = [], []
    for tw, vw, ta, a in recs:
        I = cumint(ta, a)
        lo, hi = tw[:-span] - tau, tw[span:] - tau
        ok = (lo > ta[0]) & (hi < ta[-1]) & (vw[:-span] > 0.3) & (vw[span:] > 0.3)   # driving only
        dI = np.interp(hi[ok], ta, I) - np.interp(lo[ok], ta, I)
        dv = (vw[span:] - vw[:-span])[ok]
        dt = (hi - lo)[ok]
        A.append(np.column_stack([dI, dt]))
        y.append(dv)
    A, y = np.vstack(A), np.concatenate(y)
    if len(y) < 20:
        return np.inf, np.nan, np.nan, len(y)
    (c, d), *_ = np.linalg.lstsq(A, y, rcond=None)          # dv = c * dI + d * dt, c = 1/k, d = -b/k
    res = y - A @ np.array([c, d])
    return float(np.sqrt(np.mean(res ** 2))), float(1.0 / c), float(-d / c), len(y)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folders', nargs='+')
    ap.add_argument('--truth', action='store_true', help='simulator check: true acceleration instead of a_x')
    ap.add_argument('--span', type=int, default=5, help='wheel intervals per window (default 5 = 0.1 s at 50 Hz)')
    ap.add_argument('--imu-delay-ms', type=float, default=0.0,
                    help="the IMU's own delay (low-pass filter), added to the result")
    a = ap.parse_args(argv)
    recs = [segments(f, a.truth) for f in a.folders]
    taus = np.arange(-0.060, 0.0605, 0.001)
    out = [(tau, *fit(recs, tau, a.span)) for tau in taus]
    best = min(out, key=lambda r: r[1])
    tau, rms, k, b, n = best
    print(f'{len(recs)} recording(s), {n} windows of {a.span} wheel intervals while driving')
    for t_, r_, k_, b_, n_ in out[::5]:
        print(f'  tau {1000 * t_:+6.1f} ms: residual rms {1000 * r_:6.2f} mm/s' + ('   <- best' if t_ == tau else ''))
    src = 'the TRUE acceleration' if a.truth else 'the IMU a_x'
    print(f'RESULT  wheel speed is {1000 * tau:+.1f} ms late against {src} (residual {1000 * rms:.2f} mm/s, '
          f'speed scale k {k:.4f}, a_x bias {b:+.3f} m/s^2)')
    if not a.truth:
        print(f'        wheel_delay_s = {tau + a.imu_delay_ms / 1000.0:.3f}  (= result + the IMU\'s own delay '
              f'{a.imu_delay_ms:.1f} ms, --imu-delay-ms)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
