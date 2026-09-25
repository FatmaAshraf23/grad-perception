#!/usr/bin/env python3
"""Generate a simple 1:10 test track (walls + cones) for the F1TENTH simulator.

Creates grad_track.png + grad_track.yaml in the folder given as argument
(default: current folder). Map convention follows ROS map_server:
white = free, black = occupied, resolution in m/pixel, origin = bottom-left.

Layout (metres):
  - Oval corridor ~1.6 m wide: outer wall 14 x 9 m, inner island 10.8 x 5.8 m
  - Cones: small round obstacles, ~15 cm across (3 px at 0.05 m/px)
  - Car start pose: x=3.0, y=0.8, yaw=0 (bottom straight, driving +x)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RES = 0.05                 # m per pixel
X_MIN, X_MAX = -1.0, 15.0  # map extent (m), 1 m margin around the track
Y_MIN, Y_MAX = -1.0, 10.0

# Outer wall (rounded rectangle) and inner island
OUTER = dict(x0=0.0, y0=0.0, x1=14.0, y1=9.0, r=3.0)
INNER = dict(x0=1.6, y0=1.6, x1=12.4, y1=7.4, r=1.4)

CONE_RADIUS = 0.075        # m  (15 cm diameter)
CONES = [                  # (x, y) in metres, all inside the 1.6 m lane
    (7.0, 0.5),            # bottom straight, right side of lane
    (9.5, 1.1),            # bottom straight, left side of lane
    (13.2, 4.5),           # right straight, middle of lane
    (8.0, 8.2),            # top straight
    (5.0, 7.8),            # top straight
    (0.8, 4.5),            # left straight
]


def rounded_rect_sdf(px, py, x0, y0, x1, y1, r):
    """Signed distance to a rounded rectangle (negative inside)."""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx, by = (x1 - x0) / 2 - r, (y1 - y0) / 2 - r
    qx = np.abs(px - cx) - bx
    qy = np.abs(py - cy) - by
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside - r


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    out_dir.mkdir(parents=True, exist_ok=True)

    w = int(round((X_MAX - X_MIN) / RES))
    h = int(round((Y_MAX - Y_MIN) / RES))
    # Pixel centres in world coordinates. Image row 0 is the TOP (max y).
    xs = X_MIN + (np.arange(w) + 0.5) * RES
    ys = Y_MAX - (np.arange(h) + 0.5) * RES
    px, py = np.meshgrid(xs, ys)

    free = (rounded_rect_sdf(px, py, **OUTER) < 0) & (rounded_rect_sdf(px, py, **INNER) > 0)
    for cx, cy in CONES:
        free &= np.hypot(px - cx, py - cy) > CONE_RADIUS

    img = np.where(free, 254, 0).astype(np.uint8)
    Image.fromarray(img).save(out_dir / 'grad_track.png')

    (out_dir / 'grad_track.yaml').write_text(
        'image: grad_track.png\n'
        f'resolution: {RES:.6f}\n'
        f'origin: [{X_MIN:.6f}, {Y_MIN:.6f}, 0.000000]\n'
        'negate: 0\n'
        'occupied_thresh: 0.65\n'
        'free_thresh: 0.196\n'
    )
    print(f'Wrote {out_dir / "grad_track.png"} ({w} x {h} px) and grad_track.yaml')
    print(f'{len(CONES)} cones. Start pose: sx=3.0, sy=0.8, stheta=0.0')


if __name__ == '__main__':
    main()
