#!/usr/bin/env python3
"""Compute the centre line of a closed corridor loop from a ROS map (.yaml + image).

The centre line is where the distance to the INNER island's walls equals the
distance to the OUTER walls (capped, so side corridors don't pull it outward).
Output: CSV of x,y waypoints every 0.2 m, starting next to the start pose and
running in the start heading's direction -- the input for test_driver.

Usage (needs:  ~/yolo_env/bin/pip install scikit-image scipy):
  python3 make_centerline.py ~/sim_ws/src/f1tenth_gym_ros/maps/levine.yaml \
          --start 0 0 --inside 0 4 --out levine_centerline.csv
  --start  x y : a point on the loop (the car's start position), heading +x
  --inside x y : any point INSIDE the island that the loop goes around
"""
import argparse
import os

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage
from skimage import measure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('map_yaml')
    ap.add_argument('--start', nargs=2, type=float, default=[0.0, 0.0])
    ap.add_argument('--start_yaw', type=float, default=0.0, help='rad, driving direction at start')
    ap.add_argument('--inside', nargs=2, type=float, required=True)
    ap.add_argument('--cap', type=float, default=1.0, help='max outer-wall distance used (m)')
    ap.add_argument('--spacing', type=float, default=0.2)
    ap.add_argument('--out', default='centerline.csv')
    a = ap.parse_args()

    with open(a.map_yaml) as f:
        meta = yaml.safe_load(f)
    img = np.array(Image.open(os.path.join(os.path.dirname(a.map_yaml), meta['image'])).convert('L'))
    h = img.shape[0]
    res, ox, oy = meta['resolution'], meta['origin'][0], meta['origin'][1]

    def w2p(x, y):
        return int(round(h - 1 - (y - oy) / res)), int(round((x - ox) / res))

    free = img >= 250
    lab, _ = ndimage.label(free)
    r0, c0 = w2p(*a.start)
    if not free[r0, c0]:
        raise SystemExit('Start point is not in free space -- check --start and the map origin')
    corr = lab == lab[r0, c0]
    lab2, _ = ndimage.label(~corr)
    island = lab2 == lab2[w2p(*a.inside)]
    d_in = ndimage.distance_transform_edt(~island) * res
    d_out = ndimage.distance_transform_edt(~(~corr & ~island)) * res
    field = np.where(corr, np.minimum(d_out, a.cap) - d_in, -5.0)
    c = max(measure.find_contours(field, 0.0), key=len)
    xy = np.column_stack([ox + c[:, 1] * res, oy + (h - 1 - c[:, 0]) * res])[:-1]

    i0 = np.argmin(np.hypot(xy[:, 0] - a.start[0], xy[:, 1] - a.start[1]))
    xy = np.roll(xy, -i0, axis=0)
    fwd = np.array([np.cos(a.start_yaw), np.sin(a.start_yaw)])
    if np.dot(xy[5] - xy[0], fwd) < 0:
        xy = np.vstack([xy[:1], xy[:0:-1]])
    n, w = len(xy), 6
    sm = np.array([xy[[(i + j) % n for j in range(-w, w + 1)]].mean(0) for i in range(n)])
    cl = np.vstack([sm, sm[:1]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))])
    ss = np.arange(0, s[-1], a.spacing)
    wp = np.column_stack([np.interp(ss, s, cl[:, 0]), np.interp(ss, s, cl[:, 1])])
    dist = ndimage.distance_transform_edt(corr) * res
    clr = np.array([dist[w2p(x, y)] for x, y in wp])
    np.savetxt(a.out, wp, delimiter=',', fmt='%.3f', header='x,y', comments='')
    print(f'{a.out}: loop {s[-1]:.1f} m, {len(wp)} waypoints, '
          f'wall clearance min {clr.min():.2f} m / median {np.median(clr):.2f} m')


if __name__ == '__main__':
    main()
