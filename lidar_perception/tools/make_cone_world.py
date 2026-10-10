#!/usr/bin/env python3
"""make_cone_world.py -- a simulator WORLD map = the car's map + cones (obstacle detection tests).

Usage:
  python3 make_cone_world.py <car_map.yaml> <track.yaml> <sim_config.yaml> <out_dir> <name> label:s:e_y [...]
    label:s:e_y   one cone: s along the apex_track centre line [m], e_y left of it [m] (+ = left)

Why: the F1TENTH simulator has no obstacles, but it draws its LiDAR scans from a map image.
A cone painted into that image is something the simulated LiDAR hits = an obstacle in the
WORLD. The car must NOT know about it, so the car keeps the original map
(lidar_perception sim_perception.launch.py car_map:=..., see there).

Each cone = 3 x 3 map cells (15 x 15 cm at 5 cm/cell), centred on the cell nearest to (s, e_y),
so the true cone centre is a cell centre and exactly known. Refused: a cone not on free cells,
or one the test driver could hit (it follows the centre line; car 0.31 m wide).

Writes into <out_dir>:
  <name>.png / <name>.yaml   world map (same size, resolution and origin as the car map)
  sim_<name>.yaml            simulator config = <sim_config.yaml> with map_path -> the world map
  cones_truth.csv            label, x, y, s, e_y, size_m, gap_to_wall_m, clearance_m
"""
import csv
import math
import os
import sys

import numpy as np
import yaml
from PIL import Image

CONE_CELLS = 3            # cone = CONE_CELLS x CONE_CELLS map cells
CAR_HALF_WIDTH = 0.155    # m (f1tenth car 0.31 m wide)
MIN_CLEARANCE = 0.10      # m between the car's side and the cone when the car is on the centre line


def load_map(yaml_path):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    img = np.array(Image.open(os.path.join(os.path.dirname(os.path.abspath(yaml_path)), cfg['image'])).convert('L'))
    return cfg, img


def occupied(img, cfg):
    """map_server's rule (trinary, negate 0): occupied if (255 - v) / 255 > occupied_thresh. Image row order."""
    occ = 1.0 - img.astype(np.float64) / 255.0
    return occ > cfg['occupied_thresh']


def main(argv):
    if len(argv) < 7:
        raise SystemExit(__doc__)
    car_yaml, track_yaml, sim_cfg_path, out_dir, name = argv[1:6]
    cones = [c.split(':') for c in argv[6:]]
    from apex_track.track import load_track                 # ROS-free package (needs PYTHONPATH or install)
    from apex_track.vectors import pose_from_frenet

    cfg, img = load_map(car_yaml)
    res, (ox, oy) = float(cfg['resolution']), cfg['origin'][:2]
    h, w = img.shape
    occ_img = occupied(img, cfg)
    free_img = (1.0 - img / 255.0) < cfg['free_thresh']
    track = load_track(track_yaml, with_boundaries=False)
    print(f'car map {car_yaml}: {w} x {h} cells @ {res} m, origin ({ox}, {oy}); track {track.track_id} '
          f'(hash {track.track_hash}, {track.L:.2f} m)')

    # wall cell centres (map frame) for the gap check
    wr, wc = np.nonzero(occ_img)
    wall_x = ox + (wc + 0.5) * res
    wall_y = oy + (h - 1 - wr + 0.5) * res

    world = img.copy()
    half = CONE_CELLS // 2
    size = CONE_CELLS * res
    rows = []
    for label, s_want, ey_want in cones:
        x, y, th = pose_from_frenet(track, float(s_want), float(ey_want), 0.0)
        col = int(math.floor((x - ox) / res))
        row_b = int(math.floor((y - oy) / res))              # OccupancyGrid row (0 = bottom)
        r_img = h - 1 - row_b                                 # image row (0 = top)
        cx, cy = ox + (col + 0.5) * res, oy + (row_b + 0.5) * res
        block = np.s_[r_img - half:r_img + half + 1, col - half:col + half + 1]
        if not free_img[block].all():
            raise SystemExit(f'cone {label}: not all {CONE_CELLS}x{CONE_CELLS} cells at ({cx:.3f}, {cy:.3f}) are free')
        track.reset()
        _, s, e_y, _, kappa = track.project(cx, cy, th)
        clearance = abs(e_y) - size / 2 * math.sqrt(2) - CAR_HALF_WIDTH
        if clearance < MIN_CLEARANCE:
            raise SystemExit(f'cone {label}: only {clearance:.3f} m from a car on the centre line '
                             f'(need {MIN_CLEARANCE} m -- the test driver would hit it)')
        dx = np.maximum(np.abs(wall_x - cx) - size / 2 - res / 2, 0.0)
        dy = np.maximum(np.abs(wall_y - cy) - size / 2 - res / 2, 0.0)
        gap = float(np.min(np.hypot(dx, dy)))
        world[block] = 0
        rows.append([label, f'{cx:.4f}', f'{cy:.4f}', f'{s:.3f}', f'{e_y:+.3f}', f'{size:.3f}', f'{gap:.3f}',
                     f'{clearance:.3f}'])
        print(f'  cone {label:10s} centre ({cx:.3f}, {cy:.3f})  s {s:6.2f} m  e_y {e_y:+.3f} m  kappa {kappa:+.2f}  '
              f'{size * 100:.0f} x {size * 100:.0f} cm  gap to the wall {gap * 100:.0f} cm  '
              f'clearance to a centred car {clearance * 100:.0f} cm')

    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(world).save(os.path.join(out_dir, f'{name}.png'))
    world_cfg = dict(cfg, image=f'{name}.png')
    with open(os.path.join(out_dir, f'{name}.yaml'), 'w') as f:
        yaml.safe_dump(world_cfg, f, sort_keys=False, default_flow_style=None)
    with open(sim_cfg_path) as f:
        sim_cfg = yaml.safe_load(f)
    sim_cfg['bridge']['ros__parameters']['map_path'] = os.path.join(os.path.abspath(out_dir), name)
    sim_cfg['bridge']['ros__parameters']['map_img_ext'] = '.png'
    with open(os.path.join(out_dir, f'sim_{name}.yaml'), 'w') as f:
        yaml.safe_dump(sim_cfg, f, sort_keys=False)
    with open(os.path.join(out_dir, 'cones_truth.csv'), 'w', newline='') as f:
        wr_ = csv.writer(f)
        wr_.writerow(['label', 'x', 'y', 's', 'e_y', 'size_m', 'gap_to_wall_m', 'clearance_m'])
        wr_.writerows(rows)
    added = int(occupied(world, cfg).sum() - occ_img.sum())
    print(f'world map {out_dir}/{name}.png: {added} more occupied cells than the car map '
          f'({len(rows)} cones x {CONE_CELLS * CONE_CELLS}); simulator config sim_{name}.yaml; cones_truth.csv')
    if added != len(rows) * CONE_CELLS * CONE_CELLS:
        raise SystemExit('unexpected number of new occupied cells')


if __name__ == '__main__':
    main(sys.argv)
