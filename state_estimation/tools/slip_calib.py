#!/usr/bin/env python3
"""slip_calib -- find the tyre-slip coefficient K for the EKF from a recording, WITHOUT ground truth.

WHY: the EKF's sideslip model (vehicle_model.py, beta_model 'slip') is  vy = r * (lr - K * vx^2).
K = m * lf / (L * C_r) depends on the rear tyres' cornering stiffness C_r, which no datasheet gives
for RC tyres -> measure it.  In the simulator the K found this way (0.016-0.020 from fast runs) agrees
with the truth fit (0.0170) and the physics value (0.0178).

HOW: the global EKF is replayed on the recording (IMU, wheel speed, AMCL poses) for a range of K, with
the SAME classes as the live node (ekf_node.LocalizationEKF, vehicle_model.Sideslip).  For each K we
measure how much the AMCL fixes have to correct the EKF's prediction (mean squared position
innovation).  A wrong K makes the prediction drift sideways in every fast corner and the fixes pull it
back; the right K needs the smallest corrections.  The answer: the minimum (parabola through the best
grid points), and the range of K that fits within +5 % of the best (how sharply the data decide).

WHAT TO RECORD (real car, after wheel_calib; AMCL running on the track map):
    ros2 bag record -o ~/slip_drive /imu/data /vehicle/measured /amcl_pose
    -> 3+ laps with the fastest corners you will race (the tool refuses recordings with too little
       sideways acceleration: at walking speed the tyres barely slip and AMCL's own errors dominate).
RUN:
    python3 slip_calib.py ~/slip_drive [more bags / CSV folders] [--write ~/.ros/vehicle_slip.yaml]
    then launch with beta_model:=slip (slip_k < 0 = read the file).
Inputs: rosbag2 folders (needs rosbag2_py, i.e. ROS sourced) or bag_to_csv.py folders (imu.csv,
wheel.csv, amcl.csv; plain numpy).  The EKF settings default to ekf_node's; the speed scale k is read
from the wheel_calib anchor file (--scale-file).
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np

try:
    from state_estimation.ekf_node import CHI2_3DOF_99, LocalizationEKF, StateHistory, amcl_measurement_noise, wrap
    from state_estimation.vehicle_model import Sideslip
except ImportError:                                   # from the source tree (tests, no install)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, '..', 'state_estimation'))
    from ekf_node import CHI2_3DOF_99, LocalizationEKF, StateHistory, amcl_measurement_noise, wrap
    from vehicle_model import Sideslip

EKF = dict(sigma_pos0=0.1, sigma_yaw0_deg=2.0, sigma_bias0=0.05, sigma_v=0.05, sigma_w=0.005,
           sigma_bw=0.0001, sigma_kw=0.0005, amcl_cov_scale=4.0, amcl_min_sigma_pos=0.10,
           amcl_min_sigma_yaw_deg=1.0, gate=CHI2_3DOF_99, history_s=2.0, stopped_speed=0.01,
           stopped_time=0.3, k0=1.0, sigma_k0=0.01, amcl_offset_m=0.025)


# ------------------------------------------------------------------ reading a recording
def _read_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        a = np.array([[float(v) for v in row] for row in r])
    return {h: a[:, i] for i, h in enumerate(hdr)}


def _read_bag(path):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    st = lambda s: s.sec + s.nanosec * 1e-9  # noqa: E731
    rows = {'imu': [], 'wheel': [], 'amcl': []}
    while reader.has_next():
        topic, data, t_rx = reader.read_next()
        rx = t_rx * 1e-9
        if topic == '/imu/data':
            m = deserialize_message(data, types[topic])
            rows['imu'].append([st(m.header.stamp), rx, m.angular_velocity.z])
        elif topic == '/vehicle/measured':
            m = deserialize_message(data, types[topic])
            rows['wheel'].append([st(m.header.stamp), rx, m.drive.speed, m.drive.steering_angle])
        elif topic == '/amcl_pose':
            m = deserialize_message(data, types[topic])
            p, c, q = m.pose.pose, m.pose.covariance, m.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            rows['amcl'].append([st(m.header.stamp), rx, p.position.x, p.position.y, yaw, c[0], c[1], c[7], c[35]])
    cols = {'imu': ['stamp', 'rx', 'wz'], 'wheel': ['stamp', 'rx', 'speed', 'steer'],
            'amcl': ['stamp', 'rx', 'x', 'y', 'yaw', 'cxx', 'cxy', 'cyy', 'cyaw']}
    out = {}
    for k, v in rows.items():
        if not v:
            raise ValueError(f'{path}: no {k} messages')
        a = np.array(v)
        out[k] = {h: a[:, i] for i, h in enumerate(cols[k])}
    return out


def load_recording(path):
    if os.path.exists(os.path.join(path, 'metadata.yaml')):
        return _read_bag(path)
    return {n: _read_csv(os.path.join(path, n + '.csv')) for n in ('imu', 'wheel', 'amcl')}


# ------------------------------------------------------------------ the replay
def replay(rec, K, lf, lr, prm):
    """ekf_global on the recording with sideslip K; returns the innovations of the accepted AMCL fixes:
    array of (along, across, nis) in the car frame [m]"""
    am, im, wh = rec['amcl'], rec['imu'], rec['wheel']
    off = prm['amcl_offset_m']
    ekf = LocalizationEKF(
        x0=[am['x'][0] + off, am['y'][0] + off, am['yaw'][0], 0.0, prm['k0']],
        P0_diag=[prm['sigma_pos0'], prm['sigma_pos0'], math.radians(prm['sigma_yaw0_deg']),
                 prm['sigma_bias0'], prm['sigma_k0']],
        sigma_v=prm['sigma_v'], sigma_w=prm['sigma_w'], sigma_bw=prm['sigma_bw'], sigma_kw=prm['sigma_kw'])
    slip = Sideslip('slip', lf, lr, K)
    hist = StateHistory(prm['history_s'])
    ev = sorted([(im['rx'][i], 0, i) for i in range(len(im['rx']))]
                + [(wh['rx'][i], 1, i) for i in range(len(wh['rx']))]
                + [(am['rx'][i], 2, i) for i in range(1, len(am['rx']))])
    v = 0.0
    stopped_since = last_t = None
    inn = []
    for _, kind, i in ev:
        if kind == 1:
            v = wh['speed'][i]
            t = wh['stamp'][i]
            if abs(v) < prm['stopped_speed']:
                stopped_since = t if stopped_since is None else stopped_since
            else:
                stopped_since = None
        elif kind == 0:
            t = im['stamp'][i]
            if last_t is None:
                last_t = t
                continue
            dt, last_t = t - last_t, t
            if dt <= 0 or dt > 0.5:
                continue
            w = im['wz'][i]
            if stopped_since is not None and t - stopped_since >= prm['stopped_time']:
                ekf.stationary_predict(dt)
                ekf.zupt(w)
            else:
                ekf.predict(v, w, slip.beta(ekf.x[4] * v, w - ekf.x[3]), dt)
            hist.add(int(round(t * 1e9)), ekf.x)
        else:
            x_then = hist.at(int(round(am['stamp'][i] * 1e9)))
            if x_then is None:
                continue
            z = np.array([am['x'][i] + off, am['y'][i] + off, am['yaw'][i]])
            nu = z - x_then[:3]
            nu[2] = wrap(nu[2])
            cov6 = np.zeros(36)
            cov6[0], cov6[1], cov6[6], cov6[7], cov6[35] = am['cxx'][i], am['cxy'][i], am['cxy'][i], am['cyy'][i], am['cyaw'][i]
            R = amcl_measurement_noise(cov6, prm['amcl_cov_scale'], prm['amcl_min_sigma_pos'],
                                       math.radians(prm['amcl_min_sigma_yaw_deg']))
            nis = ekf.pose_nis(nu, R)
            if nis > prm['gate']:
                continue
            c, s = math.cos(x_then[2]), math.sin(x_then[2])
            inn.append((c * nu[0] + s * nu[1], -s * nu[0] + c * nu[1], nis))
            _, dx = ekf.pose_update(nu, R)
            hist.shift(dx)
    return np.array(inn) if inn else np.zeros((0, 3))


def excitation(rec, k0):
    """share of the driving time with sideways acceleration |v * r| above 2.5 m/s^2 (where slip shows)"""
    im, wh = rec['imu'], rec['wheel']
    v = np.interp(im['stamp'], wh['stamp'], wh['speed']) * k0
    ay = np.abs(v * im['wz'])
    moving = np.abs(v) > 0.3
    return float(np.mean(ay[moving] > 2.5)) if moving.any() else 0.0, float(np.percentile(ay[moving], 99)) if moving.any() else 0.0


def read_scale(path):
    try:
        vals = {}
        with open(os.path.expanduser(path)) as f:
            for line in f:
                if ':' in line and not line.lstrip().startswith('#'):
                    k, v = line.split(':', 1)
                    vals[k.strip()] = v.strip()
        return float(vals['speed_scale'])
    except (OSError, KeyError, ValueError):
        return None


def calibrate(recs, lf, lr, prm, ks):
    """pooled mean squared position innovation for every K; minimum by a parabola; +5 % band"""
    per_run = []
    for rec in recs:
        per_run.append([replay(rec, float(k), lf, lr, prm) for k in ks])
    msq = np.array([[np.mean(i[:, 0] ** 2 + i[:, 1] ** 2) if len(i) else np.nan for i in run] for run in per_run])
    n = np.array([[len(i) for i in run] for run in per_run])
    pooled = np.nansum(msq * n, axis=0) / np.sum(n, axis=0)
    j = int(np.nanargmin(pooled))
    k_best = float(ks[j])
    if 0 < j < len(ks) - 1:                           # parabola through the best point and its neighbours
        y0, y1, y2 = pooled[j - 1], pooled[j], pooled[j + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            k_best = float(ks[j] + 0.5 * (ks[1] - ks[0]) * (y0 - y2) / den)
    band = ks[pooled <= 1.05 * np.nanmin(pooled)]
    run_best = [float(ks[int(np.nanargmin(m))]) for m in msq]
    return dict(k=k_best, band=(float(band.min()), float(band.max())), pooled=pooled, per_run=msq,
                run_best=run_best, gain=float(1.0 - np.nanmin(pooled) / pooled[0]), n=n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recordings', nargs='+', help='rosbag2 folders or bag_to_csv folders')
    ap.add_argument('--lf', type=float, default=0.15875)
    ap.add_argument('--lr', type=float, default=0.17145)
    ap.add_argument('--k-max', type=float, default=0.05)
    ap.add_argument('--k-step', type=float, default=0.002)
    ap.add_argument('--scale-file', default='~/.ros/ekf_speed_scale.yaml', help="wheel_calib anchor ('none' = k 1.0)")
    ap.add_argument('--amcl-offset', type=float, default=0.025, help='half the map resolution (ekf_node amcl_offset_m)')
    ap.add_argument('--min-excitation', type=float, default=0.05,
                    help='minimum share of driving time above 2.5 m/s^2 sideways (default 5 %%)')
    ap.add_argument('--write', help='calibration file to write, e.g. ~/.ros/vehicle_slip.yaml')
    a = ap.parse_args(argv)

    prm = dict(EKF, amcl_offset_m=a.amcl_offset)
    k_scale = None if a.scale_file.lower() == 'none' else read_scale(a.scale_file)
    prm['k0'] = k_scale if k_scale else 1.0
    print(f'speed scale k0 = {prm["k0"]:.4f}' + ('' if k_scale else ' (no wheel_calib file -- run wheel_calib first on the car)'))
    recs, ok = [], []
    for p in a.recordings:
        rec = load_recording(os.path.expanduser(p))
        share, ay99 = excitation(rec, prm['k0'])
        enough = share >= a.min_excitation
        print(f'{p}: {len(rec["imu"]["stamp"])} IMU, {len(rec["amcl"]["stamp"])} AMCL poses; sideways accel > 2.5 m/s^2 '
              f'{100 * share:.0f} % of the driving time (p99 {ay99:.1f} m/s^2) -> {"used" if enough else "NOT USED: too little cornering"}')
        if enough:
            recs.append(rec)
            ok.append(p)
    if not recs:
        print('REFUSED: no recording with enough fast cornering -- drive the corners at racing speed')
        return 1
    ks = np.round(np.arange(0.0, a.k_max + 1e-9, a.k_step), 6)
    t0 = time.time()
    r = calibrate(recs, a.lf, a.lr, prm, ks)
    print(f'{len(ks)} values of K x {len(recs)} recordings replayed in {time.time() - t0:.0f} s')
    print('   K       mean sq. AMCL correction [cm^2]')
    for k, v in zip(ks, r['pooled']):
        print(f'  {k:.4f}  {1e4 * v:8.2f}' + ('   <- best grid point' if v == np.nanmin(r['pooled']) else ''))
    print(f'RESULT  K = {r["k"]:.4f} s^2/m   (within 5 % of the best fit: {r["band"][0]:.4f} .. {r["band"][1]:.4f}; '
          f'per recording: {", ".join(f"{k:.4f}" for k in r["run_best"])})')
    print(f'        AMCL corrections {100 * r["gain"]:.0f} % smaller than with K = 0 (no tyre slip)')
    sigma = max((r['band'][1] - r['band'][0]) / 2.0, a.k_step / 2.0)
    if not (0.002 <= r['k'] <= 0.9 * a.k_max):
        print('WARNING: K at the edge of the search range or implausible -- check the recording (AMCL lost? '
              'wrong amcl_offset? wheel_calib missing?) before using it')
    if a.write:
        path = os.path.expanduser(a.write)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write('# Tyre-slip coefficient for the EKF sideslip model (vehicle_model.py): vy = r * (lr - K * vx^2)\n')
            f.write('# written by slip_calib (AMCL-innovation fit, no ground truth)\n')
            f.write(f'slip_k: {r["k"]:.6f}\n')
            f.write(f'slip_k_sigma: {sigma:.6f}\n')
            f.write(f'lf: {a.lf}\nlr: {a.lr}\nspeed_scale_used: {prm["k0"]:.6f}\n')
            f.write(f'recordings: {"; ".join(ok)}\n')
            f.write(f"saved: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        os.replace(tmp, path)
        print(f'written: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
