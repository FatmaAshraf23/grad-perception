#!/usr/bin/env python3
"""Speed-sweep analysis: localization / StateEstimate errors per run, per TRUE-SPEED band and on
straights vs corners, for several eval_logger CSVs side by side.

  python3 speed_sweep_analysis.py spd15.csv spd30.csv spd45.csv spd60.csv [--labels 1.5,3,4.5,6] [--out sweep.md]

Same definitions as plot_eval.py (so the numbers can be compared with every earlier run):
  driving rows      v > 0.05 m/s
  EKF position err  hypot(ex - tx, ey - ty)       EKF heading err  wrap(eyaw - tyaw)
  s err             se_s - tr_s                   e_y err          se_ey - tr_ey
  e_psi err         wrap(se_epsi - tr_epsi)       vx / vy / r err  se_* - tr_*
  straight          |tr_kappa| < 0.05 1/m         corner           |tr_kappa| >= 0.3 1/m
  inside 2 sigma    |e_y err| <= 2 se_std_ey, |e_psi err| <= 2 se_std_epsi
numpy only.
"""
import argparse
import math
import os
import sys

import numpy as np

BANDS = [(0.05, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.5), (5.5, 99.0)]


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def p95(x):
    x = np.abs(x[np.isfinite(x)])
    return float(np.percentile(x, 95)) if len(x) else float('nan')


def analyse(path):
    d = np.genfromtxt(path, delimiter=',', names=True)
    m = d['v'] > 0.05
    se = m & np.isfinite(d['se_s']) & np.isfinite(d['tr_s'])
    e_pos = np.hypot(d['ex'] - d['tx'], d['ey'] - d['ty'])
    e_yaw = wrap_deg(np.degrees(d['eyaw'] - d['tyaw']))
    e = dict(s=d['se_s'] - d['tr_s'], ey=d['se_ey'] - d['tr_ey'],
             epsi=wrap_deg(np.degrees(d['se_epsi'] - d['tr_epsi'])),
             vx=d['se_vx'] - d['tr_vx'], vy=d['se_vy'] - d['tr_vy'], r=d['se_r'] - d['tr_r'])
    straight = se & (np.abs(d['tr_kappa']) < 0.05)
    corner = se & (np.abs(d['tr_kappa']) >= 0.3)
    v = d['v']
    modes = d['se_mode'][se]
    out = dict(
        file=os.path.basename(path), drive_s=0.1 * m.sum(), dist=float(d['s'][-1]),
        v_mean=float(v[m].mean()) if m.any() else float('nan'), v_top=float(np.percentile(v[m], 99)) if m.any() else float('nan'),
        share_45=float(np.mean(v[m] > 4.5)) if m.any() else float('nan'),
        ekf_pos=p95(e_pos[m]), ekf_yaw=p95(e_yaw[m]), ekf_in=float(np.mean(e_pos[m] <= 2 * d['e_sig'][m])) if m.any() else float('nan'),
        amcl=p95(d['a_err'][m]),
        s=p95(e['s'][se]), ey=p95(e['ey'][se]), epsi=p95(e['epsi'][se]),
        epsi_str=p95(e['epsi'][straight]), epsi_cor=p95(e['epsi'][corner]),
        ey_str=p95(e['ey'][straight]), ey_cor=p95(e['ey'][corner]),
        vx=p95(e['vx'][se]), vy=p95(e['vy'][se]), r=p95(e['r'][se]),
        in_ey=float(np.mean(np.abs(e['ey'][se]) <= 2 * d['se_std_ey'][se])) if se.any() else float('nan'),
        in_epsi=float(np.mean(np.abs(np.radians(e['epsi'][se])) <= 2 * d['se_std_epsi'][se])) if se.any() else float('nan'),
        ok=float(np.mean(modes == 1)) if len(modes) else float('nan'),
        lat=p95(d['se_lat_ms'][se]), age=p95(d['se_age_ms'][se]),
        bands=[])
    for lo, hi in BANDS:
        b = se & (v >= lo) & (v < hi)
        bm = m & (v >= lo) & (v < hi)
        out['bands'].append(dict(lo=lo, hi=hi, n=int(b.sum()), ey=p95(e['ey'][b]), epsi=p95(e['epsi'][b]),
                                 s=p95(e['s'][b]), vy=p95(e['vy'][b]), ekf_pos=p95(e_pos[bm])))
    return out


def table(runs, labels):
    L = []
    hdr = '| quantity | ' + ' | '.join(labels) + ' |'
    L += [hdr, '|' + '---|' * (len(labels) + 1)]
    rows = [
        ('driving time [s] / distance [m]', lambda r: f'{r["drive_s"]:.0f} / {r["dist"]:.0f}'),
        ('true speed mean / top (p99) [m/s]', lambda r: f'{r["v_mean"]:.2f} / {r["v_top"]:.2f}'),
        ('time above 4.5 m/s', lambda r: f'{100 * r["share_45"]:.0f} %'),
        ('EKF position p95 [cm]', lambda r: f'{100 * r["ekf_pos"]:.1f}'),
        ('EKF heading p95 [deg]', lambda r: f'{r["ekf_yaw"]:.2f}'),
        ('EKF inside 2 sigma', lambda r: f'{100 * r["ekf_in"]:.0f} %'),
        ('AMCL position p95 [cm]', lambda r: f'{100 * r["amcl"]:.1f}'),
        ('SE s p95 [cm]', lambda r: f'{100 * r["s"]:.1f}'),
        ('SE e_y p95 [cm] (straight / corner)', lambda r: f'{100 * r["ey"]:.1f} ({100 * r["ey_str"]:.1f} / {100 * r["ey_cor"]:.1f})'),
        ('SE e_psi p95 [deg] (straight / corner)', lambda r: f'{r["epsi"]:.2f} ({r["epsi_str"]:.2f} / {r["epsi_cor"]:.2f})'),
        ('SE vx / vy p95 [m/s]', lambda r: f'{r["vx"]:.3f} / {r["vy"]:.3f}'),
        ('SE r p95 [rad/s]', lambda r: f'{r["r"]:.4f}'),
        ('inside 2 sigma e_y / e_psi', lambda r: f'{100 * r["in_ey"]:.0f} / {100 * r["in_epsi"]:.0f} %'),
        ('mode OK while driving', lambda r: f'{100 * r["ok"]:.2f} %'),
        ('SE latency / age p95 [ms]', lambda r: f'{r["lat"]:.1f} / {r["age"]:.1f}'),
    ]
    for name, f in rows:
        L.append(f'| {name} | ' + ' | '.join(f(r) for r in runs) + ' |')
    L += ['', 'By TRUE speed band (all runs pooled per band; n = rows at 10 Hz):', '',
          '| band [m/s] | n | e_y p95 [cm] | e_psi p95 [deg] | s p95 [cm] | vy p95 [m/s] | EKF pos p95 [cm] |',
          '|---|---|---|---|---|---|---|']
    return L


def pooled_bands(paths):
    """bands pooled over all runs (per-run bands are small)"""
    E = {k: [] for k in ('v', 'ey', 'epsi', 's', 'vy', 'pos', 'vm', 'se')}
    for p in paths:
        d = np.genfromtxt(p, delimiter=',', names=True)
        m = d['v'] > 0.05
        se = m & np.isfinite(d['se_s']) & np.isfinite(d['tr_s'])
        E['v'].append(d['v']); E['se'].append(se); E['vm'].append(m)
        E['ey'].append(d['se_ey'] - d['tr_ey']); E['s'].append(d['se_s'] - d['tr_s'])
        E['epsi'].append(wrap_deg(np.degrees(d['se_epsi'] - d['tr_epsi'])))
        E['vy'].append(d['se_vy'] - d['tr_vy']); E['pos'].append(np.hypot(d['ex'] - d['tx'], d['ey'] - d['ty']))
    A = {k: np.concatenate(v) for k, v in E.items()}
    L = []
    for lo, hi in BANDS:
        b = A['se'] & (A['v'] >= lo) & (A['v'] < hi)
        bm = A['vm'] & (A['v'] >= lo) & (A['v'] < hi)
        hi_txt = f'{hi:.1f}' if hi < 50 else '+'
        L.append(f'| {lo:.2f}-{hi_txt} | {int(b.sum())} | {100 * p95(A["ey"][b]):.1f} | {p95(A["epsi"][b]):.2f} | '
                 f'{100 * p95(A["s"][b]):.1f} | {p95(A["vy"][b]):.3f} | {100 * p95(A["pos"][bm]):.1f} |')
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--labels')
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    labels = a.labels.split(',') if a.labels else [os.path.basename(p).split('_2026')[0] for p in a.csv]
    runs = [analyse(p) for p in a.csv]
    lines = ['Files: ' + ', '.join(r['file'] for r in runs), ''] + table(runs, labels) + pooled_bands(a.csv)
    text = '\n'.join(lines)
    print(text)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
