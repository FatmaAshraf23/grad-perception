"""Seam, corner and kappa tests (APEX contract: 1 mm / 0.001 rad).
Run from the package folder:  python3 test/test_vectors.py [path/to/track.yaml]

  1. round trip at the chosen seam / corner / straight poses
  2. round trip at EVERY 2 cm of the track, e_y = -0.3 ... +0.3 m
     (catches problems between the chosen poses)
  3. driving ACROSS the start/finish seam, forward then backward, in 1 cm steps:
     s_abs must change by exactly the step -- no 62 m jump, lap counter right
  4. kappa_at(s) gives the same kappa as project(), and works on whole arrays
  5. golden file tracks/<name>/test_vectors.csv reproduced within tolerance
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from apex_track import load_track, wrap                               # noqa: E402
from apex_track.vectors import (TOL_ANG, TOL_POS, pose_from_frenet,   # noqa: E402
                                read_golden, round_trip)

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(HERE), 'tracks', 'levine', 'track.yaml')
tr = load_track(path)
print(f'track {tr.track_id}, hash {tr.track_hash}, L = {tr.L:.3f} m')
ok = True


def check(name, cond, info=''):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name} {info}")


# 1) chosen poses
res = round_trip(tr)
worst = max(res, key=lambda r: max(abs(r[1]), abs(r[2])) / TOL_POS + abs(r[3]) / TOL_ANG)
check(f'1 round trip, {len(res)} chosen poses (seam, 3 sharpest corners, straights)',
      all(abs(ds) <= TOL_POS and abs(de) <= TOL_POS and abs(dp) <= TOL_ANG
          for _, ds, de, dp, _ in res),
      f'(worst {worst[0]}: ds {worst[1]*1e3:.4f} mm, de_y {worst[2]*1e3:.4f} mm, '
      f'de_psi {worst[3]*1e3:.4f} mrad)')

# 2) dense sweep
rng = np.random.default_rng(1)
mx = np.zeros(3)
for e_y in (-0.3, -0.1, 0.1, 0.3):
    for s in np.arange(0.0, tr.L, 0.02):
        e_psi = rng.uniform(-0.3, 0.3)
        x, y, yaw = pose_from_frenet(tr, s, e_y, e_psi)
        tr.reset()
        _, s2, e_y2, e_psi2, _ = tr.project(x, y, yaw)
        err = [abs((s2 - s + tr.L / 2) % tr.L - tr.L / 2), abs(e_y2 - e_y), abs(wrap(e_psi2 - e_psi))]
        mx = np.maximum(mx, err)
check('2 round trip every 2 cm, |e_y| <= 0.3 m',
      mx[0] <= TOL_POS and mx[1] <= TOL_POS and mx[2] <= TOL_ANG,
      f'(max ds {mx[0]*1e3:.4f} mm, de_y {mx[1]*1e3:.4f} mm, de_psi {mx[2]*1e3:.4f} mrad)')

# 3) across the seam, forward then backward, without reset (= how the node uses it)
tr.reset()
steps = np.concatenate([np.arange(tr.L - 1.0, tr.L + 1.0, 0.01),       # forward over the line
                        np.arange(tr.L + 1.0, tr.L - 1.0, -0.01)])     # and back again
prev, max_jump, s_abs_list = None, 0.0, []
for s in steps:
    x, y, yaw = pose_from_frenet(tr, s, 0.15, 0.05)
    s_abs, *_ = tr.project(x, y, yaw)
    s_abs_list.append(s_abs)
    if prev is not None:
        max_jump = max(max_jump, abs(abs(s_abs - prev) - 0.01))
    prev = s_abs
peak = max(s_abs_list)
check('3a s_abs continuous across the seam (each 1 cm step = 1 cm)', max_jump <= TOL_POS,
      f'(worst step error {max_jump*1e3:.4f} mm)')
check('3b s_abs past the line = L + 1 m, lap = 1', abs(peak - (tr.L + 0.99)) <= 0.011,
      f'(peak s_abs {peak:.3f}, L {tr.L:.3f})')
check('3c back behind the line: lap = 0 again', tr.lap == 0 and s_abs_list[-1] < tr.L,
      f'(lap {tr.lap}, s_abs {s_abs_list[-1]:.3f})')

# 4) kappa_at
d = 0.0
for s in np.arange(0.0, tr.L, 0.013):
    x, y, yaw = pose_from_frenet(tr, s, 0.1, 0.0)
    tr.reset()
    _, s2, _, _, k = tr.project(x, y, yaw)
    d = max(d, abs(k - tr.kappa_at(s2)))
check('4a kappa_at(s) == kappa from project()', d < 1e-9, f'(max diff {d:.1e})')
horizon = np.linspace(tr.L - 0.5, tr.L + 1.5, 21)                     # an MPC horizon across the seam
k_vec = tr.kappa_at(horizon)
k_one = np.array([tr.kappa_at(float(s)) for s in horizon])
check('4b kappa_at(array) == one-by-one, also across the seam',
      k_vec.shape == horizon.shape and np.allclose(k_vec, k_one, atol=1e-12))

# 5) golden file
gpath = os.path.join(os.path.dirname(tr.source_file), 'test_vectors.csv')
if not os.path.exists(gpath):
    check('5 golden file exists', False, f'({gpath} missing: python3 -m apex_track.vectors {path})')
else:
    meta, rows = read_golden(gpath)
    if meta.get('track_hash') != tr.track_hash:
        check('5 golden file belongs to this track', False,
              f"(file hash {meta.get('track_hash')} != track {tr.track_hash}: regenerate ONLY if the "
              'geometry change was on purpose)')
    else:
        worst = 0.0
        for r in rows:
            tr.reset()
            _, s2, e_y2, e_psi2, k2 = tr.project(r['x'], r['y'], r['yaw'])
            ds = abs((s2 - r['s_track'] + tr.L / 2) % tr.L - tr.L / 2)
            worst = max(worst, ds / TOL_POS, abs(e_y2 - r['e_y']) / TOL_POS,
                        abs(wrap(e_psi2 - r['e_psi'])) / TOL_ANG, abs(k2 - r['kappa']) / 1e-3)
        check(f'5 golden file reproduced ({len(rows)} poses)', worst <= 1.0,
              f'(worst = {worst:.4f} x tolerance)')

print('ALL PASS' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
