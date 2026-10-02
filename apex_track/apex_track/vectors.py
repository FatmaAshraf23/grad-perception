"""Frenet test vectors: known poses with their (s, e_y, e_psi, kappa).

Two uses (APEX contract section 9, deliverable 1):
  round_trip(track)    start from chosen (s, e_y, e_psi), build the pose,
                       project it back -> must return the same numbers.
                       Checks the code against itself from the other
                       direction, no stored answers needed.
  golden file          tracks/<name>/test_vectors.csv: ~20 poses with the
                       answers of THIS implementation. Any later version (or a
                       teammate's own code) must reproduce them within
                       1 mm / 0.001 rad. The file carries the track_hash, so it
                       is never compared against a different track.

The poses are chosen where Frenet code usually breaks:
  - the START/FINISH SEAM (s jumps from L back to 0)
  - the SHARPEST CORNERS (largest |kappa|), inside and outside
  - a few straight / ordinary points as a sanity reference

Make the golden file (only when the track or geometry changes ON PURPOSE):
    python3 -m apex_track.vectors tracks/levine/track.yaml
"""
import csv
import math
import os
import sys

import numpy as np

from .frenet import wrap

TOL_POS = 0.001     # m   (control's acceptance tolerance)
TOL_ANG = 0.001     # rad
COLUMNS = ['name', 'x', 'y', 'yaw', 's_track', 'e_y', 'e_psi', 'kappa']


def pose_from_frenet(track, s, e_y, e_psi):
    """Map pose (x, y, yaw) that is e_y left of the centre line at s, with heading error e_psi."""
    cx, cy, th = track.point(s)
    return cx - e_y * math.sin(th), cy + e_y * math.cos(th), wrap(th + e_psi)


def _corner_centres(track, count=3, min_gap_m=3.0):
    """s of the `count` sharpest corners (largest |kappa|, at least min_gap_m apart)."""
    s_mid = track.s0 + 0.5 * track.seg_len
    order = np.argsort(-np.abs(track.kappa_seg))
    picked = []
    for j in order:
        s = float(s_mid[j])
        gaps = [min(abs(s - q), track.L - abs(s - q)) for q in picked]
        if all(g >= min_gap_m for g in gaps):
            picked.append(s)
        if len(picked) == count:
            break
    return picked


def frenet_cases(track):
    """List of (name, s, e_y, e_psi) chosen at the seam, the sharpest corners and straights."""
    L = track.L
    cases = [  # start/finish seam: both sides, both directions of offset
        ('seam_before_30cm', L - 0.30, 0.20, math.radians(5)),
        ('seam_before_5cm', L - 0.05, -0.15, math.radians(-8)),
        ('seam_at_line', 0.0, 0.0, 0.0),
        ('seam_after_5cm', 0.05, 0.15, math.radians(3)),
        ('seam_after_30cm', 0.30, -0.20, math.radians(-4)),
        ('seam_after_1m', 1.00, 0.30, math.radians(10)),
    ]
    for c, s_c in enumerate(_corner_centres(track), start=1):
        k = track.kappa_at(s_c)
        inside = 0.25 if k > 0 else -0.25        # left turn -> inside is left (+e_y)
        cases += [
            (f'corner{c}_apex_centre', s_c, 0.0, 0.0),
            (f'corner{c}_apex_inside', s_c, inside, math.radians(8) * np.sign(k)),
            (f'corner{c}_apex_outside', s_c, -inside, math.radians(-6) * np.sign(k)),
            (f'corner{c}_entry', s_c - 0.6, 0.5 * inside, math.radians(4)),
        ]
    for s in (L * 0.25, L * 0.5, L * 0.75):
        cases.append((f'track_{s:.1f}m', s, 0.10, math.radians(2)))
    return cases


def round_trip(track):
    """Errors (ds, de_y, de_psi) for every case; each projected from scratch (global search)."""
    out = []
    for name, s, e_y, e_psi in frenet_cases(track):
        x, y, yaw = pose_from_frenet(track, s, e_y, e_psi)
        track.reset()
        _, s2, e_y2, e_psi2, k2 = track.project(x, y, yaw)
        ds = (s2 - s + track.L / 2) % track.L - track.L / 2        # seam-safe difference
        out.append((name, ds, e_y2 - e_y, wrap(e_psi2 - e_psi), k2))
    return out


def golden_rows(track):
    rows = []
    for name, s, e_y, e_psi in frenet_cases(track):
        x, y, yaw = pose_from_frenet(track, s, e_y, e_psi)
        track.reset()
        _, s2, e_y2, e_psi2, k2 = track.project(x, y, yaw)
        rows.append([name, x, y, yaw, s2, e_y2, e_psi2, k2])
    return rows


def write_golden(track, path):
    with open(path, 'w', newline='') as f:
        f.write(f'# track_id={track.track_id} track_hash={track.track_hash}\n')
        f.write('# poses in the map frame (m, rad); answers from apex_track project(). '
                'Tolerance 1 mm / 0.001 rad.\n')
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in golden_rows(track):
            w.writerow([r[0]] + [f'{v:.6f}' for v in r[1:]])


def read_golden(path):
    meta, rows = {}, []
    with open(path, newline='') as f:
        lines = f.read().splitlines()
    for line in lines:
        if line.startswith('#'):
            for part in line[1:].split():
                if '=' in part:
                    k, v = part.split('=', 1)
                    meta[k] = v
    data = [ln for ln in lines if ln and not ln.startswith('#')]
    for r in csv.DictReader(data):
        rows.append({k: (r[k] if k == 'name' else float(r[k])) for k in COLUMNS})
    return meta, rows


if __name__ == '__main__':
    from .track import load_track
    if len(sys.argv) != 2:
        sys.exit('usage: python3 -m apex_track.vectors tracks/<name>/track.yaml')
    tr = load_track(sys.argv[1])
    out = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), 'test_vectors.csv')
    write_golden(tr, out)
    print(f'wrote {out} ({len(frenet_cases(tr))} poses, track_hash {tr.track_hash})')
