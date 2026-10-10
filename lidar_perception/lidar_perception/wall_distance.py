#!/usr/bin/env python3
"""wall_distance.py -- obstacle detection step 2.1: how far is every map cell from the nearest wall?

Approach B (map difference) asks ONE question per LiDAR point: "is there a wall of the car's map
close to this point?" The car asks it ~2500 times per second (360 beams x 7 scans), so the hard
part is done ONCE, when the map arrives: a grid with the same cells as the map that holds, for
every cell, the distance from its centre to the nearest wall surface. A point is then answered
with one array lookup: find its cell, read the number.

How the grid is made (plain numpy, no scipy -> also runs on the Pi):
  A wall cell (di, dj) cells away from a cell is a 5 x 5 cm square. The gap between this cell's
  centre and that square is  res * hypot(max(|dj| - 0.5, 0), max(|di| - 0.5, 0))
  (0.025 m to the face of a side neighbour, 0.035 m to the corner of a diagonal one).
  For every offset (di, dj) closer than max_dist: shift the whole wall mask by (di, dj); every
  cell that finds a wall there could be that close to a wall -> keep the smallest value.
  That is exact (the true nearest wall square) up to max_dist; farther cells just get max_dist.

Map convention (ROS: map_server, RViz, the simulator, apex_track; NOT nav2 AMCL's, see ekf_node):
  OccupancyGrid order (row 0 = bottom); cell (row, col) covers x in origin_x + [col, col+1) * res,
  y in origin_y + [row, row+1) * res. Walls = occupied cells (>= 65). UNKNOWN cells are not walls.

Usage (offline check):  python3 wall_distance.py <map.yaml> [cones_truth.csv]
"""
import csv
import math
import os
import sys
import time

import numpy as np


def load_map(yaml_path):
    """Map yaml + image -> (grid, res, ox, oy): the OccupancyGrid map_server would publish
    (100 wall, 0 free, -1 unknown; row 0 = bottom). Offline only -- the node gets /map instead, so yaml and
    PIL are imported here and the node needs nothing but numpy."""
    import yaml
    from PIL import Image
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    img = np.array(Image.open(os.path.join(os.path.dirname(os.path.abspath(yaml_path)), cfg['image'])).convert('L'))
    shade = img.astype(np.float64) / 255.0
    occ = shade if cfg.get('negate', 0) else 1.0 - shade      # map_server, trinary mode
    grid = np.full(img.shape, -1, dtype=np.int8)
    grid[occ > cfg['occupied_thresh']] = 100
    grid[occ < cfg['free_thresh']] = 0
    return grid[::-1].copy(), float(cfg['resolution']), float(cfg['origin'][0]), float(cfg['origin'][1])


class WallDistance:
    """Distance from every map cell to the nearest wall surface [m], plus a lookup for points."""

    def __init__(self, grid, res, ox, oy, max_dist=1.0, wall_from=65):
        self.res, self.ox, self.oy, self.max_dist = float(res), float(ox), float(oy), float(max_dist)
        grid = np.asarray(grid)
        self.wall = grid >= wall_from            # the only thing the car's map "knows"
        self.free = grid == 0                    # the track surface (where an obstacle can stand)
        self.h, self.w = grid.shape
        self.dist = self._compute()

    def _compute(self):
        h, w, k = self.h, self.w, int(math.ceil(self.max_dist / self.res))
        dist = np.full((h, w), self.max_dist, dtype=np.float64)
        dist[self.wall] = 0.0
        pad = np.zeros((h + 2 * k, w + 2 * k), dtype=bool)       # walls with a border of k empty cells
        pad[k:k + h, k:k + w] = self.wall
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                d = self.res * math.hypot(max(abs(dj) - 0.5, 0.0), max(abs(di) - 0.5, 0.0))
                if (di == 0 and dj == 0) or d >= self.max_dist:
                    continue
                wall_there = pad[k + di:k + di + h, k + dj:k + dj + w]   # [r, c] = wall at (r + di, c + dj)?
                dist[wall_there & (dist > d)] = d
        # Rounded to 1 um so a value is exactly its decimal number (0.05 * 1.5 is 0.07500000000000001 in
        # floating point). Still: choose tolerances BETWEEN the grid's values (on straight walls 2.5, 7.5,
        # 12.5 ... cm), e.g. 10 cm -- a tolerance equal to a grid value decides by rounding luck.
        return np.round(dist, 6)

    def cell(self, x, y):
        """(row, col) of the cell each point (x, y) [m, map frame] falls in."""
        col = np.floor((np.asarray(x, dtype=float) - self.ox) / self.res).astype(np.int64)
        row = np.floor((np.asarray(y, dtype=float) - self.oy) / self.res).astype(np.int64)
        return row, col

    def inside(self, row, col):
        return (row >= 0) & (row < self.h) & (col >= 0) & (col < self.w)

    def at(self, x, y):
        """Distance to the nearest wall [m] for each point. Outside the map = 0 (never an obstacle)."""
        row, col = self.cell(x, y)
        ok = self.inside(row, col)
        out = np.zeros(row.shape, dtype=np.float64)
        out[ok] = self.dist[row[ok], col[ok]]
        return out

    def on_free(self, x, y):
        """True where the point lies on the track surface (free cell) -- not in unknown space."""
        row, col = self.cell(x, y)
        ok = self.inside(row, col)
        out = np.zeros(row.shape, dtype=bool)
        out[ok] = self.free[row[ok], col[ok]]
        return out


def brute_force(wd, rows, cols):
    """Slow but obviously right: compare each cell with EVERY wall cell, keep the smallest gap."""
    wr, wc = np.nonzero(wd.wall)
    out = []
    for r, c in zip(rows, cols):
        gx = np.maximum(np.abs(wc - c) - 0.5, 0.0)
        gy = np.maximum(np.abs(wr - r) - 0.5, 0.0)
        out.append(min(wd.res * float(np.min(np.hypot(gx, gy))), wd.max_dist))
    return np.array(out)


def main():
    yaml_path = sys.argv[1]
    grid, res, ox, oy = load_map(yaml_path)
    t0 = time.perf_counter()
    wd = WallDistance(grid, res, ox, oy)
    ms = (time.perf_counter() - t0) * 1000
    print(f'map {os.path.basename(yaml_path)}: {wd.w} x {wd.h} cells @ {res} m, origin ({ox}, {oy}): '
          f'wall {int(wd.wall.sum())}, free {int(wd.free.sum())}, unknown {int((grid == -1).sum())}')
    print(f'distance grid built in {ms:.0f} ms (exact up to {wd.max_dist:.2f} m)')

    # 1. Is it right? Compare with the slow, obvious calculation on random free cells.
    rng = np.random.default_rng(0)
    fr, fc = np.nonzero(wd.free)
    pick = rng.choice(len(fr), 300, replace=False)
    diff = np.abs(brute_force(wd, fr[pick], fc[pick]) - wd.dist[fr[pick], fc[pick]])
    print(f'check: 300 random free cells vs brute force over all {int(wd.wall.sum())} wall cells: '
          f'largest difference {diff.max() * 100:.4f} cm -> {"OK" if diff.max() < 1e-5 else "WRONG"}')

    # 2. What does it look like on the track surface?
    d_free = wd.dist[wd.free]
    print(f'track surface (free cells): distance to the nearest wall from {d_free.min() * 100:.1f} to '
          f'{d_free.max() * 100:.1f} cm (lane 1.6 m wide -> the middle is ~0.8 m from both walls)')
    print('   tolerance T   part of the track surface closer than T to a wall ("blind strip")')
    for T in (0.05, 0.10, 0.15, 0.20, 0.25):
        print(f'   {T * 100:5.0f} cm     {np.mean(d_free < T) * 100:5.1f} %')

    # 3. The cones of step 1: how far are their cells from the walls of the CAR's map?
    if len(sys.argv) > 2:
        print('cones (from cones_truth.csv) -- distance of their 3 x 3 cells to the nearest wall of the car map:')
        with open(sys.argv[2]) as f:
            for c in csv.DictReader(f):
                x, y, half = float(c['x']), float(c['y']), float(c['size_m']) / 2
                xs = np.arange(x - half + res / 2, x + half, res)
                ys = np.arange(y - half + res / 2, y + half, res)
                X, Y = np.meshgrid(xs, ys)
                d = wd.at(X.ravel(), Y.ravel())
                print(f'   {c["label"]:10s} ({x:.3f}, {y:.3f}): {d.min() * 100:4.1f} .. {d.max() * 100:4.1f} cm '
                      f'({len(d)} cells)')
    return wd


if __name__ == '__main__':
    main()
