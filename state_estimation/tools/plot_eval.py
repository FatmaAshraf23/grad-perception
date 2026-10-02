#!/usr/bin/env python3
"""Plot and summarise an evaluation run recorded by eval_logger.

Usage (plain Python, no ROS needed):
    python3 plot_eval.py ~/eval_logs/run_20260929_201500.csv
    python3 plot_eval.py run_a.csv run_b.csv          # compare runs (e.g. before/after tuning)
Optional: --map ~/sim_ws/src/f1tenth_gym_ros/maps/levine.yaml   (draws the walls)

Writes next to the (first) CSV:
    <name>_trajectory.png   truth vs EKF vs wheel odometry
    <name>_position.png     position error vs distance (EKF with +-2 sigma, AMCL, wheel)
    <name>_heading.png      heading error vs distance
    <name>_states.png       speed scale k and gyro bias vs distance, with the true values
    <name>_summary.txt      the numbers table (also printed)
"""
import argparse
import math
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

TRUE_K = 1.0 / 1.03             # fake_vehicle_sensors speed_scale_error = 3 %
C_TRUTH, C_EKF, C_AMCL, C_WHEEL = '#222222', '#1f77b4', '#ff7f0e', '#d62728'


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def load(path):
    """Returns the data and a mask of the rows where the car is driving.
    Statistics use only these rows: while the car stands still AMCL does not
    update, so its last value would be counted over and over."""
    d = np.genfromtxt(path, delimiter=',', names=True)
    return d, d['v'] > 0.05


def errors(d):
    e_pos = np.hypot(d['ex'] - d['tx'], d['ey'] - d['ty'])
    e_yaw = wrap_deg(np.degrees(d['eyaw'] - d['tyaw']))
    w_pos = np.hypot(d['wx'] - d['tx'], d['wy'] - d['ty'])
    w_yaw = wrap_deg(np.degrees(d['wyaw'] - d['tyaw']))
    return e_pos, e_yaw, w_pos, w_yaw


def stats(x):
    x = x[np.isfinite(x)]
    if not len(x):
        return 'n/a'
    return (f'mean {np.mean(x):7.3f}  rms {math.sqrt(np.mean(x ** 2)):7.3f}  '
            f'p95 {np.percentile(x, 95):7.3f}  max {np.max(x):7.3f}')


def summary(name, d, m):
    e_pos, e_yaw, w_pos, w_yaw = errors(d)
    inside = e_pos[m] <= 2 * d['e_sig'][m]
    lines = [
        f'== {name} ==  (statistics over driving time only)',
        f'distance driven {d["s"][-1]:.1f} m, driving time {0.1 * np.sum(m):.0f} s '
        f'(recording {d["t"][-1]:.0f} s)',
        f'EKF   position error [m]   {stats(e_pos[m])}',
        f'EKF   heading error [deg]  {stats(np.abs(e_yaw[m]))}',
        f'AMCL  position error [m]   {stats(d["a_err"][m])}',
        f'AMCL  heading error [deg]  {stats(np.abs(d["a_err_yaw"][m]))}',
        f'Wheel position error [m]   {stats(w_pos[m])}',
        f'Wheel heading error [deg]  {stats(np.abs(w_yaw[m]))}',
        f'EKF error inside its +-2 sigma: {100 * np.mean(inside):.0f} % of samples (ideal ~95 %)',
        f'EKF position first 20 m [m]  {stats(e_pos[m & (d["s"] < 20.0)])}',
        f'EKF position after 20 m [m]  {stats(e_pos[m & (d["s"] >= 20.0)])}',
        f'Speed scale k start {d["k"][np.flatnonzero(np.isfinite(d["k"]))[0]]:.4f}',
        f'Speed scale k final {d["k"][-1]:.4f} (true {TRUE_K:.4f})',
        f'Gyro bias final {math.degrees(d["bias"][-1]):.3f} deg/s '
        f'(true {math.degrees(d["true_bias"][-1]):.3f})',
    ]
    return '\n'.join(lines)


def draw_map(ax, yaml_path):
    import yaml
    from PIL import Image
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    img = np.array(Image.open(os.path.join(os.path.dirname(yaml_path), meta['image'])).convert('L'))
    h = img.shape[0]
    ys, xs = np.nonzero(img < 50)
    ox, oy = meta['origin'][0], meta['origin'][1]
    r = meta['resolution']
    ax.plot(ox + xs * r, oy + (h - 1 - ys) * r, ',', color='#999999')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--map', default=None, help='map .yaml to draw the walls')
    args = ap.parse_args()

    runs = [(os.path.splitext(os.path.basename(p))[0], *load(p)) for p in args.csv]
    base = os.path.splitext(args.csv[0])[0]
    text = '\n\n'.join(summary(n, d, m) for n, d, m in runs)
    print(text)
    with open(base + '_summary.txt', 'w') as f:
        f.write(text + '\n')

    name, d, _ = runs[0]
    e_pos, e_yaw, w_pos, w_yaw = errors(d)
    s = d['s']

    # 1. Trajectory
    fig, ax = plt.subplots(figsize=(9, 6))
    if args.map:
        draw_map(ax, args.map)
    ax.plot(d['wx'], d['wy'], color=C_WHEEL, lw=1, label='wheel odometry')
    ax.plot(d['tx'], d['ty'], color=C_TRUTH, lw=2.5, label='truth')
    ax.plot(d['ex'], d['ey'], color=C_EKF, lw=1.2, label='EKF (global)')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(loc='best')
    ax.set_title('Trajectory')
    lim_x = np.nanpercentile(np.concatenate([d['tx'], d['ex']]), [0, 100])
    lim_y = np.nanpercentile(np.concatenate([d['ty'], d['ey']]), [0, 100])
    ax.set_xlim(lim_x[0] - 3, lim_x[1] + 3)
    ax.set_ylim(lim_y[0] - 3, lim_y[1] + 3)
    fig.savefig(base + '_trajectory.png', dpi=130, bbox_inches='tight')

    # 2. Position error vs distance
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for n, dd, _ in runs:
        ep = errors(dd)[0]
        lbl = 'EKF' if len(runs) == 1 else f'EKF – {n}'
        ax.plot(dd['s'], ep, lw=1.4, label=lbl, color=C_EKF if len(runs) == 1 else None)
    ax.fill_between(s, 0, 2 * d['e_sig'], color=C_EKF, alpha=0.15, label='EKF ±2σ')
    ax.plot(s, d['a_err'], color=C_AMCL, lw=0.8, alpha=0.8, label='AMCL')
    ax.plot(s, w_pos, color=C_WHEEL, lw=1, label='wheel odometry')
    ax.set_yscale('log')
    ax.set_ylim(0.005, max(10.0, np.nanmax(w_pos) * 1.2))
    ax.set_xlabel('distance driven [m]')
    ax.set_ylabel('position error [m] (log)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper left', ncol=2)
    ax.set_title('Position error')
    fig.savefig(base + '_position.png', dpi=130, bbox_inches='tight')

    # 3. Heading error vs distance
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(s, -2 * np.degrees(d['e_sig_yaw']), 2 * np.degrees(d['e_sig_yaw']),
                    color=C_EKF, alpha=0.15, label='EKF ±2σ')
    ax.plot(s, e_yaw, color=C_EKF, lw=1.2, label='EKF')
    ax.plot(s, d['a_err_yaw'], color=C_AMCL, lw=0.8, alpha=0.8, label='AMCL')
    ax.set_ylim(-5, 5)
    ax.set_xlabel('distance driven [m]')
    ax.set_ylabel('heading error [deg]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_title('Heading error (wheel odometry is off the scale)')
    fig.savefig(base + '_heading.png', dpi=130, bbox_inches='tight')

    # 4. Estimated sensor errors
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    a1.plot(s, d['k'], color=C_EKF, label='EKF estimate')
    a1.axhline(TRUE_K, color=C_TRUTH, ls='--', label=f'true ({TRUE_K:.3f})')
    a1.set_ylabel('speed scale k')
    a1.legend(loc='lower right')
    a1.grid(True, alpha=0.3)
    a2.plot(s, np.degrees(d['bias']), color=C_EKF, label='EKF estimate')
    a2.plot(s, np.degrees(d['true_bias']), color=C_TRUTH, ls='--', label='true')
    a2.set_ylabel('gyro bias [deg/s]')
    a2.set_xlabel('distance driven [m]')
    a2.legend(loc='lower right')
    a2.grid(True, alpha=0.3)
    fig.suptitle('Sensor errors learned by the EKF')
    fig.savefig(base + '_states.png', dpi=130, bbox_inches='tight')

    print(f'\nFigures and summary written next to {args.csv[0]}')


if __name__ == '__main__':
    main()
