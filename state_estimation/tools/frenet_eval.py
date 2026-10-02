#!/usr/bin/env python3
"""Offline check of frenet.py on logged eval runs: TRUE pose vs EKF pose in
Frenet coordinates -> the errors control will actually see in s, e_y, e_psi.

Usage (from the folder that contains frenet.py):
  python3 frenet_eval.py levine_centerline.csv ~/eval_logs/*.csv [--smooth 0.5]
Needs only numpy.
"""
import argparse
import csv
import math

import numpy as np

from apex_track import FrenetTrack, wrap


def run(path, track_csv, smooth):
    tt = FrenetTrack.from_csv(track_csv, smooth_m=smooth)    # truth
    te = FrenetTrack.from_csv(track_csv, smooth_m=smooth)    # estimate (own lap counter)
    ds, dey, dep, kap = [], [], [], []
    for r in csv.DictReader(open(path)):
        if float(r['v']) <= 0.05 or r['ex'] in ('', 'nan'):
            continue
        T = tt.project(float(r['tx']), float(r['ty']), float(r['tyaw']))
        E = te.project(float(r['ex']), float(r['ey']), float(r['eyaw']))
        ds.append(E[0] - T[0])
        dey.append(E[2] - T[2])
        dep.append(math.degrees(wrap(E[3] - T[3])))
        kap.append(abs(T[4]))
    a = lambda v: np.abs(np.array(v))  # noqa: E731
    straight = np.array(kap) < 0.05
    return dict(laps=tt.lap, s_end=T[0], L=tt.L,
                s_mean=a(ds).mean() * 100, s_max=a(ds).max() * 100,
                ey_mean=a(dey).mean() * 100, ey_p95=np.percentile(a(dey), 95) * 100,
                ey_max=a(dey).max() * 100,
                ep_mean=a(dep).mean(), ep_p95=np.percentile(a(dep), 95), ep_max=a(dep).max(),
                ep_straight_max=a(dep)[straight].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('track_csv')
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--smooth', type=float, default=0.5)
    a = ap.parse_args()
    print(f'{"run":28s} laps  s_abs end | s err mean/max [cm] | e_y err mean/p95/max [cm] |'
          f' e_psi err mean/p95/max [deg] (straights max)')
    for p in a.logs:
        r = run(p, a.track_csv, a.smooth)
        name = p.replace('\\', '/').split('/')[-1][:-4]
        print(f'{name:28s} {r["laps"]:3d} {r["s_end"]:9.1f} | {r["s_mean"]:5.1f} / {r["s_max"]:5.1f}'
              f'       | {r["ey_mean"]:4.1f} / {r["ey_p95"]:4.1f} / {r["ey_max"]:4.1f}'
              f'          | {r["ep_mean"]:4.2f} / {r["ep_p95"]:4.2f} / {r["ep_max"]:5.2f}'
              f' ({r["ep_straight_max"]:4.2f})')
    print(f'smoothed track length {r["L"]:.2f} m (smooth_m = {a.smooth})')


if __name__ == '__main__':
    main()
