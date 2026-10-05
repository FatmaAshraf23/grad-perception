#!/usr/bin/env python3
"""Track BOUNDARIES: how far the walls are, left and right of the centre line.

WHAT IT PRODUCES
    For every point of the smoothed centre line (every ds = 5 cm of s):
        w_left  [m]  distance from the centre line to the LEFT wall,
                     measured along the left normal (+ e_y direction)
        w_right [m]  distance to the RIGHT wall along the right normal (- e_y)
    So the drivable corridor at s is   -w_right(s) <= e_y <= w_left(s),
    in exactly the Frenet coordinates of the StateEstimate message.
    Planning's room at the car's position:
        room_left  = w_left(s)  - e_y
        room_right = w_right(s) + e_y
    (raw walls: each team subtracts its own margin -- car half-width + buffer)

WHY MEASURED ALONG THE NORMAL (not "nearest wall")
    e_y is measured along the normal, so the boundary has to be too; then
    "e_y <= w_left(s)" means "the point is inside the wall" with no extra
    geometry. Planning/MPC constraints become simple bounds on e_y.

HOW (ray casting on the occupancy map)
    The map (.yaml + image, the same one AMCL uses -- from slam_toolbox on
    the real car) is read with map_server's rules: a pixel is FREE when its
    occupancy p < free_thresh; occupied AND unknown pixels both stop a ray
    (unknown = we have never seen it, so we must not drive there).
    From each centre-line point a ray is marched in steps of `step_m`
    (default res/10) along the normal until it meets a non-free pixel.
    Accuracy: about half a pixel diagonal (map resolution 5 cm -> ~3.5 cm).

OPENINGS (doors, side corridors, junctions)
    Where a ray runs further than `max_w` (default 2.0 m) or leaves the map,
    there is no wall on that side (rays that graze a door post and jump
    > 10 cm wider than their neighbour count as opening too). These
    stretches are closed by a straight
    VIRTUAL wall from the last wall point before the opening to the first
    one after it (width = where the ray meets that line; rays that miss it
    fall back to linear interpolation along s) -- and flagged in the columns
    left_virtual / right_virtual so nobody mistakes them for real walls.
    Our own track is closed by walls, so it should have none; Levine (an
    office corridor) has several.

FILE  tracks/<name>/boundaries.csv
    '#' header lines: the track_hash the widths were made for, map file +
    its SHA-256, settings.  Then:  s,w_left,w_right,left_virtual,right_virtual
    load_track() checks the header track_hash and the s grid, so widths can
    never be used with a different centre line or smoothing.

MAKE / REMAKE IT (after the map or the centre line changes):
    python3 -m apex_track.boundaries tracks/levine/track.yaml \\
        --map ~/sim_ws/src/f1tenth_gym_ros/maps/levine.yaml
    then make sure track.yaml has the line   boundaries: boundaries.csv
    (Needs Pillow to read the map image -- only this tool, not the runtime.)

Real car: identical; --map = our slam_toolbox map (.yaml + .pgm).
"""
import argparse
import datetime
import hashlib
import math
import os
import sys

import numpy as np
import yaml

from .boundary_io import COLUMNS


# --- map ---------------------------------------------------------------------

class OccupancyMap:
    """ROS map (.yaml + image) -> free-space test for world points."""

    def __init__(self, map_yaml):
        from PIL import Image          # tool-only dependency (not needed at runtime)
        map_yaml = os.path.abspath(os.path.expanduser(map_yaml))
        with open(map_yaml, encoding='utf-8') as f:
            meta = yaml.safe_load(f)
        img_path = os.path.join(os.path.dirname(map_yaml), meta['image'])
        v = np.asarray(Image.open(img_path).convert('L'), dtype=float)
        # map_server: p = occupancy probability from the pixel value
        p = v / 255.0 if int(meta.get('negate', 0)) else (255.0 - v) / 255.0
        self.free = p < float(meta['free_thresh'])          # occupied + unknown -> not free
        self.h, self.w = self.free.shape
        self.res = float(meta['resolution'])
        self.ox, self.oy = float(meta['origin'][0]), float(meta['origin'][1])
        yaw = float(meta['origin'][2]) if len(meta['origin']) > 2 else 0.0
        self.c, self.s = math.cos(yaw), math.sin(yaw)
        self.yaml_path, self.image_path = map_yaml, img_path
        with open(img_path, 'rb') as f:
            self.image_sha256 = hashlib.sha256(f.read()).hexdigest()

    def is_free(self, x, y):
        """Bool array: world points (x, y) in free space. Outside the map -> False."""
        dx, dy = np.asarray(x) - self.ox, np.asarray(y) - self.oy
        u = (self.c * dx + self.s * dy) / self.res          # map-image column (float)
        v = (-self.s * dx + self.c * dy) / self.res         # rows counted from the BOTTOM
        col = np.floor(u).astype(int)
        row = self.h - 1 - np.floor(v).astype(int)          # image row 0 = top
        inside = (col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)
        out = np.zeros(np.shape(col), dtype=bool)
        out[inside] = self.free[row[inside], col[inside]]
        return out


def cast_rays(omap, x0, y0, ux, uy, max_range, step):
    """Distance from each (x0, y0) along unit vector (ux, uy) to the first
    non-free sample. NaN where none was met within max_range (an opening)."""
    n = len(x0)
    dist = np.full(n, np.nan)
    alive = np.ones(n, dtype=bool)
    for k in range(1, int(math.ceil(max_range / step)) + 1):
        d = k * step
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break
        hit = ~omap.is_free(x0[idx] + d * ux[idx], y0[idx] + d * uy[idx])
        dist[idx[hit]] = d - 0.5 * step          # wall edge lies inside the last step
        alive[idx[hit]] = False
    return dist


def _fill_periodic(s, w, L):
    """Linear interpolation (along s) over the NaN stretches of a closed loop."""
    ok = ~np.isnan(w)
    if ok.sum() < 2:
        raise ValueError('fewer than 2 measured widths on one side -- check the map / max_w')
    ss, ww = s[ok], w[ok]
    ss = np.concatenate([ss[-1:] - L, ss, ss[:1] + L])     # wrap around the seam
    ww = np.concatenate([ww[-1:], ww, ww[:1]])
    return np.interp(s, ss, ww)


def _grow_openings(w, jump=0.10):
    """Rays right next to an opening often GRAZE the door post (they pass
    through the post's edge pixels and hit something further away). Such a
    point is wider than its neighbour on the wall side by more than `jump`;
    count it as part of the opening too, so the virtual wall starts at a
    clean wall point. Repeats until nothing changes."""
    w = w.copy()
    n = len(w)
    changed = True
    while changed:
        changed = False
        miss = np.isnan(w)
        for j in np.nonzero(miss)[0]:
            for e, o in (((j - 1) % n, (j - 2) % n), ((j + 1) % n, (j + 2) % n)):
                if not miss[e] and not np.isnan(w[o]) and w[e] - w[o] > jump:
                    w[e] = np.nan
                    changed = True
    return w


def _close_openings(x0, y0, ux, uy, w, max_w):
    """Fill each NaN stretch (an opening) with a STRAIGHT virtual wall: the segment
    from the last wall point before the opening to the first one after it.
    Width = where the ray meets that segment. Rays that miss it (very bent
    stretches) keep the along-s interpolation done by the caller."""
    n = len(w)
    miss = np.isnan(w)
    out = w.copy()
    if not miss.any() or miss.all():
        return out
    hx, hy = x0 + np.nan_to_num(w) * ux, y0 + np.nan_to_num(w) * uy      # wall hit points
    i = 0
    start = int(np.argmin(miss))                 # a measured point: walk the loop from here
    while i < n:
        j = (start + i) % n
        if not miss[j]:
            i += 1
            continue
        a = (j - 1) % n                          # last measured point before the opening
        k = i
        while k < n and miss[(start + k) % n]:
            k += 1
        b = (start + k) % n                      # first measured point after it
        ax, ay, bx, by = hx[a], hy[a], hx[b], hy[b]
        for m in range(i, k):
            q = (start + m) % n
            # x0 + t*ux = ax + u*(bx-ax) ; y0 + t*uy = ay + u*(by-ay)
            det = ux[q] * -(by - ay) + (bx - ax) * uy[q]
            if abs(det) < 1e-9:
                continue
            rx, ry = ax - x0[q], ay - y0[q]
            t = (rx * -(by - ay) + (bx - ax) * ry) / det
            u = (ux[q] * ry - uy[q] * rx) / det
            if 0.0 < t <= max_w and -1e-6 <= u <= 1 + 1e-6:
                out[q] = t
        i = k
    return out


def compute_boundaries(track, map_yaml, max_w=2.0, step_m=None):
    """Widths for every centre-line point of `track` (a FrenetTrack)."""
    omap = OccupancyMap(map_yaml)
    step = step_m or omap.res / 10.0
    s = track.s0.copy()
    x0, y0 = track.p[:, 0].copy(), track.p[:, 1].copy()
    th = np.array([track.point(si)[2] for si in s])          # the same smooth tangent as project()
    nx, ny = -np.sin(th), np.cos(th)                          # left normal (+ e_y)

    bad = ~omap.is_free(x0, y0)
    if bad.any():
        raise ValueError(f'{bad.sum()} centre-line points are not in free space '
                         f'(first at s = {s[bad][0]:.2f} m) -- wrong map or map origin?')
    wl = cast_rays(omap, x0, y0, nx, ny, max_w, step)
    wr = cast_rays(omap, x0, y0, -nx, -ny, max_w, step)
    wl, wr = _grow_openings(wl), _grow_openings(wr)
    lv, rv = np.isnan(wl), np.isnan(wr)
    wl = _close_openings(x0, y0, nx, ny, wl, max_w)
    wr = _close_openings(x0, y0, -nx, -ny, wr, max_w)
    return {
        's': s,
        'w_left': _fill_periodic(s, wl, track.L),
        'w_right': _fill_periodic(s, wr, track.L),
        'left_virtual': lv, 'right_virtual': rv,
        'info': {'map': omap.yaml_path, 'map_image_sha256': omap.image_sha256,
                 'map_resolution': omap.res, 'max_w': max_w, 'step_m': step},
    }


# --- file ----------------------------------------------------------------------

def write_boundaries_csv(path, track, b):
    i = b['info']
    with open(path, 'w', newline='\n') as f:
        f.write(f'# apex_track boundaries for track_id={track.track_id} track_hash={track.track_hash}\n')
        f.write(f'# map={os.path.basename(i["map"])} map_image_sha256={i["map_image_sha256"]} '
                f'resolution={i["map_resolution"]}\n')
        f.write(f'# max_w={i["max_w"]} step_m={i["step_m"]:.4f} '
                f'made={datetime.date.today().isoformat()} (python3 -m apex_track.boundaries)\n')
        f.write('# w_left / w_right = metres from the centre line to the wall along the normal '
                '(corridor: -w_right <= e_y <= w_left); *_virtual = 1: no wall there, closed by a straight virtual wall\n')
        f.write(','.join(COLUMNS) + '\n')
        for row in zip(b['s'], b['w_left'], b['w_right'], b['left_virtual'], b['right_virtual']):
            f.write(f'{row[0]:.4f},{row[1]:.4f},{row[2]:.4f},{int(row[3])},{int(row[4])}\n')


# --- summary -------------------------------------------------------------------

def _stretches(s, mask, L):
    """[(s_start, s_end), ...] of the True runs of a closed-loop mask."""
    if not mask.any():
        return []
    if mask.all():
        return [(0.0, L)]
    k = int(np.argmin(mask))                     # start the scan at a False point
    m, ss = np.roll(mask, -k), np.roll(s, -k)
    out, start = [], None
    for j in range(len(m)):
        if m[j] and start is None:
            start = ss[j]
        if not m[j] and start is not None:
            out.append((start, ss[j - 1]))
            start = None
    if start is not None:
        out.append((start, ss[-1]))
    return out


def summary(track, b):
    s, wl, wr = b['s'], b['w_left'], b['w_right']
    k = track.kappa_at(s)
    # Frenet singularity margin at the walls: D = 1 - kappa*e_y must stay > 0
    d_left, d_right = 1.0 - k * wl, 1.0 + k * wr
    lines = [
        f'track {track.track_id}  hash {track.track_hash}  L = {track.L:.2f} m  points {len(s)}',
        f'w_left  min {wl.min():.3f} / median {np.median(wl):.3f} / max {wl.max():.3f} m',
        f'w_right min {wr.min():.3f} / median {np.median(wr):.3f} / max {wr.max():.3f} m',
        f'corridor width min {np.min(wl + wr):.3f} m at s = {s[np.argmin(wl + wr)]:.2f}',
        f'Frenet margin D = 1 - kappa*e_y at the walls: min {min(d_left.min(), d_right.min()):.3f}'
        f' (left wall {d_left.min():.3f} at s = {s[np.argmin(d_left)]:.2f}, '
        f'right wall {d_right.min():.3f} at s = {s[np.argmin(d_right)]:.2f})'
        + ('   !! <= 0: normals cross before the wall -- Frenet is NOT valid there'
           if min(d_left.min(), d_right.min()) <= 0 else ''),
    ]
    for side in ('left', 'right'):
        st = _stretches(s, b[f'{side}_virtual'], track.L)
        lines.append(f'{side} virtual (no wall, closed by a straight line): '
                     + (', '.join(f's {a:.2f}-{e:.2f}' for a, e in st) if st else 'none'))
    return '\n'.join(lines)


def main(argv=None):
    from .track import load_track
    ap = argparse.ArgumentParser(description='Compute tracks/<name>/boundaries.csv from a ROS map')
    ap.add_argument('track_yaml')
    ap.add_argument('--map', required=True, help='ROS map .yaml (same map AMCL uses)')
    ap.add_argument('--max_w', type=float, default=2.0,
                    help='rays longer than this = opening, filled as a virtual wall [m]')
    ap.add_argument('--out', default=None, help='default: boundaries.csv next to track.yaml')
    a = ap.parse_args(argv)

    track = load_track(a.track_yaml, with_boundaries=False)
    b = compute_boundaries(track, a.map, max_w=a.max_w)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.track_yaml)), 'boundaries.csv')
    write_boundaries_csv(out, track, b)
    print(summary(track, b))
    print(f'wrote {out}')
    with open(a.track_yaml, encoding='utf-8') as f:
        if 'boundaries' not in yaml.safe_load(f):
            print(f'NEXT: add the line   boundaries: {os.path.basename(out)}   to {a.track_yaml}')


if __name__ == '__main__':
    sys.exit(main())
