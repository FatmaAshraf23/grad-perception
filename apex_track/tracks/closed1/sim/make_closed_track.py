#!/usr/bin/env python3
"""make_closed_track.py <out_dir> -- the closed1 SIMULATOR test track (v2 2026-10-06: thin walls): 1.6 m lane,
walls all around (no openings, no side corridors), corner radii from 1.1 m (our design minimum) to 2.2 m,
one right-hand turn, one long featureless straight, start/finish on a straight at (0, 0) heading +x
(same start pose as Levine). Made to test whether Levine's OPEN corners (side corridors) caused its
corner errors -- they did not; nav2 AMCL's half-cell map offset did (ekf_node parameter amcl_offset_m).

Built from straights and circular arcs; the two lengths marked 'solve' are computed so the loop
closes exactly. Writes:
  <name>.png / <name>.yaml        ROS map (5 cm, white = free, black = wall)
  <name>_centerline.csv           x,y every 0.2 m (test_driver waypoints, apex_track centre line)
and prints the geometry + checks (lane clearance between neighbouring parts of the loop).
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

NAME = 'closed1'
HALF_W = 0.8            # m, half lane width (1.6 m lane)
RES = 0.05              # m per pixel
MARGIN = 1.0            # m around the outside of the lane in the image
WALL_T = 0.15           # m: walls are a THIN band (3 px) with UNKNOWN (grey 205) behind them, like a SLAM map.
                        # NOT solid black: AMCL's likelihood-field model scores a laser point by its distance to
                        # the nearest occupied pixel, so inside a solid black block EVERY point scores as a perfect
                        # hit and AMCL's estimate drifted/jumped into the walls (jobs 068/069, 2026-10-06).
START_OFFSET = 4.0      # m: the start/finish line (and the car's start pose (0, 0, 0)) lies this far into
                        # the long straight -- low curvature at the line (control, 2026-10-03)
# (kind, value, radius): 'S' straight length (None = solve), 'L'/'R' arc of <value> degrees
SEGMENTS = [
    ('S', None, 0),     # S1: long featureless straight (start/finish), solved
    ('L', 90, 1.1),     # C1: tight (design minimum)
    ('S', 6.0, 0),
    ('L', 90, 2.2),     # C2: gentle
    ('S', 7.0, 0),
    ('L', 90, 1.5),     # C3: medium
    ('S', 1.0, 0),
    ('R', 90, 1.5),     # C4: medium, RIGHT-hand
    ('S', 3.3, 0),
    ('L', 90, 1.1),     # C5: tight
    ('S', None, 0),     # S6: solved
    ('L', 90, 1.5),     # C6: medium, back onto the start straight
]


def build(lengths, step=0.01):
    """Dense centre line (x, y, heading) for the segment list with the solved lengths filled in."""
    x, y, th = 0.0, 0.0, 0.0
    pts = [(x, y, th)]
    li = iter(lengths)
    for kind, val, r in SEGMENTS:
        if kind == 'S':
            L = val if val is not None else next(li)
            n = max(int(round(L / step)), 1)
            for _ in range(n):
                x += L / n * math.cos(th)
                y += L / n * math.sin(th)
                pts.append((x, y, th))
        else:
            sgn = 1.0 if kind == 'L' else -1.0
            ang = math.radians(val)
            n = max(int(round(r * ang / step)), 1)
            cx, cy = x - sgn * r * math.sin(th), y + sgn * r * math.cos(th)
            for i in range(1, n + 1):
                t = th + sgn * ang * i / n
                pts.append((cx + sgn * r * math.sin(t), cy - sgn * r * math.cos(t), t))
            x, y, th = pts[-1]
    return np.array(pts)


def solve():
    """Two unknown straight lengths that close the loop (end point = start point)."""
    base = build([0.0, 0.0])[-1, :2]
    d1 = build([1.0, 0.0])[-1, :2] - base
    d2 = build([0.0, 1.0])[-1, :2] - base
    A = np.column_stack([d1, d2])
    return np.linalg.solve(A, -base)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    out.mkdir(parents=True, exist_ok=True)
    L1, L6 = solve()
    print(f'solved straights: S1 = {L1:.3f} m, S6 = {L6:.3f} m')
    if min(L1, L6) < 0.5:
        raise SystemExit('a solved straight is too short -- change the segment list')
    pts = build([L1, L6])
    end_err = math.hypot(*pts[-1, :2])
    xy = pts[:-1, :2]
    seg = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])[:-1]
    Ltot = s[-1] + seg[-1]
    # move the start/finish line START_OFFSET into the first straight, and put it at (0, 0)
    i0 = int(np.argmin(np.abs(s - START_OFFSET)))
    xy = np.roll(xy - xy[i0], -i0, axis=0)
    seg = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])[:-1]
    print(f'loop length {Ltot:.2f} m, closure error {end_err * 1000:.2f} mm, '
          f'heading change {math.degrees(pts[-1, 2] - pts[0, 2]):.1f} deg')

    # clearance: centre-line points that are far apart ALONG the loop must be far apart in space
    tree = cKDTree(xy)
    worst, where = 1e9, None
    for i, j in tree.query_pairs(r=2 * HALF_W + 1.0):
        ds = abs(s[i] - s[j])
        ds = min(ds, Ltot - ds)
        if ds > 4.0:
            d = math.hypot(*(xy[i] - xy[j]))
            if d < worst:
                worst, where = d, (s[i], s[j])
    if where:
        print(f'closest approach of two different parts of the loop: {worst:.2f} m (s {where[0]:.1f} / {where[1]:.1f}); '
              f'lane 1.6 m -> wall between them {worst - 2 * HALF_W:.2f} m')
    else:
        print(f'no two different parts of the loop closer than {2 * HALF_W + 1.0:.1f} m')

    # map: free = within HALF_W of the centre line, everything else wall
    x0, y0 = xy.min(0) - HALF_W - MARGIN
    x1, y1 = xy.max(0) + HALF_W + MARGIN
    w, h = int(math.ceil((x1 - x0) / RES)), int(math.ceil((y1 - y0) / RES))
    cx = x0 + (np.arange(w) + 0.5) * RES
    cy = y0 + (h - 1 - np.arange(h) + 0.5) * RES          # image row 0 = top
    X, Y = np.meshgrid(cx, cy)
    d, _ = tree.query(np.column_stack([X.ravel(), Y.ravel()]))
    d = d.reshape(h, w)
    img = np.full((h, w), 205, dtype=np.uint8)            # unknown (map_server: neither free nor occupied)
    img[d < HALF_W + WALL_T] = 0                           # wall band
    img[d < HALF_W] = 254                                  # lane (free)
    Image.fromarray(img).save(out / f'{NAME}.png')
    print(f'map pixels: free {np.mean(img == 254) * 100:.1f} %, wall {np.mean(img == 0) * 100:.1f} %, unknown {np.mean(img == 205) * 100:.1f} %')
    (out / f'{NAME}.yaml').write_text(
        f'image: {NAME}.png\nresolution: {RES:.6f}\norigin: [{x0:.6f}, {y0:.6f}, 0.000000]\n'
        'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    print(f'map {w} x {h} px @ {RES} m, origin ({x0:.3f}, {y0:.3f})')

    # waypoints every 0.2 m (same spacing as levine_centerline.csv)
    n = int(round(Ltot / 0.2))
    t = np.linspace(0, Ltot, n, endpoint=False)
    sx = np.concatenate([s, [Ltot]])
    closed = np.vstack([xy, xy[:1]])
    wp = np.column_stack([np.interp(t, sx, closed[:, 0]), np.interp(t, sx, closed[:, 1])])
    np.savetxt(out / f'{NAME}_centerline.csv', wp, delimiter=',', fmt='%.4f', header='x,y', comments='')
    print(f'{len(wp)} waypoints -> {NAME}_centerline.csv')
    # where the corners are (for the report)
    acc = -START_OFFSET
    li = iter([L1, L6])
    for kind, val, r in SEGMENTS:
        if kind == 'S':
            L = val if val is not None else next(li)
            print(f'  s {acc:6.2f} - {acc + L:6.2f}  straight {L:5.2f} m'
                  + ('  (start/finish line at s = 0; s < 0 = end of the lap)' if acc < 0 else ''))
            acc += L
        else:
            L = r * math.radians(val)
            print(f'  s {acc:6.2f} - {acc + L:6.2f}  {"left " if kind == "L" else "RIGHT"} {val} deg, R {r} m (kappa {1 / r:.2f})')
            acc += L


if __name__ == '__main__':
    main()
