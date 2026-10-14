"""projection_demo -- what the camera would see: the LiDAR points + the true cones projected into a virtual camera.

Uses a RECORDED simulator run (job 090: scans.npz + csv/truth.csv, the cone world's cones_truth.csv). For each cone it
picks the scan where that cone is ~3 m ahead of the camera and fully in view, then projects
  - every LiDAR return: laser -> base_link -> camera_optical_frame -> pixel (the chain of projection.py),
  - every cone's 3-D surface -> its box in the picture (cone_box),
with the camera where sensor_mounts_sim.yaml puts it and the lens of the given camera_info YAML.
FUSION IDEA it shows: the LiDAR points inside a cone's box tell WHERE the cone is -- but the box also covers the
ground behind the cone, so points of a wall further back can fall into it too: the association must keep only the
NEAREST group of points in the box (range gating).
Left panel: top view (car, LiDAR points, cones, camera view wedge). Right panel: the camera picture.

  ros2 run sensor_fusion projection_demo --run <dir with scans.npz, csv/truth.csv> --cones <cones_truth.csv>
      --camera <camera_info yaml> [--mounts sim] [--cone-height 0.30] [--out <dir>]
"""
import argparse
import csv
import math
import os

import numpy as np

from sensor_fusion import projection as pj


def load_truth(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r['stamp']) for r in rows])
    return t, np.array([float(r['x']) for r in rows]), np.array([float(r['y']) for r in rows]), \
        np.unwrap(np.array([float(r['yaw']) for r in rows]))


def load_cones(path):
    with open(path) as f:
        return [dict(label=r['label'], x=float(r['x']), y=float(r['y']), size=float(r.get('size_m') or 0.2))
                for r in csv.DictReader(f)]


def nearest_group(pts_base, gap=0.30):
    """The points within `gap` (along the range) of the closest point -- the object in front."""
    d = np.hypot(pts_base[:, 0], pts_base[:, 1])
    return d <= d.min() + gap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--cones', required=True)
    ap.add_argument('--camera', required=True, help='camera_info YAML (e.g. iphone_iriun_640x480.yaml)')
    ap.add_argument('--mounts', default='sim')
    ap.add_argument('--cone-height', type=float, default=0.30, help='m (the 2-D simulator has no cone height)')
    ap.add_argument('--distance', type=float, default=3.0, help='m from the camera to show each cone at')
    ap.add_argument('--out', default='.')
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from apex_description.mounts import load_mounts

    cam = pj.CameraModel.from_yaml(a.camera)
    m = load_mounts(a.mounts)
    Ts = pj.mount_transforms(m)
    T_bl, T_bo = Ts['laser'], Ts['camera_optical_frame']
    T_ob = pj.invert(T_bo)
    cam_xy = T_bo[:3, 3]
    z = np.load(os.path.join(a.run, 'scans.npz'))
    stamps, ranges, meta = z['scan_a1_stamp'], z['scan_a1_ranges'], z['scan_a1_meta']
    tt, tx, ty, tyaw = load_truth(os.path.join(a.run, 'csv', 'truth.csv'))
    cones = load_cones(a.cones)
    print(f'camera {a.camera}: {cam.width} x {cam.height}, fx {cam.fx:.1f} fy {cam.fy:.1f}; mount '
          f"x {m['camera']['x']} z {m['camera']['z']} pitch {m['camera']['pitch']} deg; {len(stamps)} scans; "
          f'{len(cones)} cones, assumed {a.cone_height:.2f} m tall')
    os.makedirs(a.out, exist_ok=True)

    def cones_in_base(k):
        x, y, yaw = (np.interp(stamps[k], tt, v) for v in (tx, ty, tyaw))
        c, s = math.cos(yaw), math.sin(yaw)
        out = []
        for cn in cones:
            dx, dy = cn['x'] - x, cn['y'] - y
            out.append((cn, np.array([c * dx + s * dy, -s * dx + c * dy])))
        return out

    # the frame per cone: ~a.distance ahead of the camera, fully in view
    picks = {}
    for k in range(len(stamps)):
        for cn, p in cones_in_base(k):
            d = math.hypot(p[0] - cam_xy[0], p[1] - cam_xy[1])
            b = pj.cone_box(cam, T_ob, p, cn['size'] / 2, a.cone_height)
            if b and b['all_visible']:
                score = abs(d - a.distance)
                if cn['label'] not in picks or score < picks[cn['label']][0]:
                    picks[cn['label']] = (score, k)
    print(f'{"cone":>10} {"scan":>5} {"dist":>6} {"LiDAR pts":>9} {"in box":>6} {"nearest":>7} {"pos error":>10} '
          f'{"cam-only dist err/px":>20}')

    # ground grid + horizon (fixed for every frame)
    grid = [np.column_stack([np.full(50, x), np.linspace(-3, 3, 50), np.zeros(50)]) for x in range(1, 11)] + \
           [np.column_stack([np.linspace(0.5, 10, 50), np.full(50, y), np.zeros(50)]) for y in (-2, -1, 0, 1, 2)]
    hz = np.column_stack([np.full(60, 1000.0), np.linspace(-800, 800, 60), np.full(60, cam_xy[2])])
    for label, (_, k) in sorted(picks.items()):
        pts_l, _ = pj.scan_to_points(ranges[k], meta[0], meta[1], meta[2], meta[3], z=m['laser']['z'])
        pts_b = pj.apply(T_bl, pts_l)
        uv, ok = cam.project(pj.apply(T_ob, pts_b))
        vis = ok & cam.in_image(uv)
        rng_b = np.hypot(pts_b[:, 0], pts_b[:, 1])
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={'width_ratios': [1, 1.25]})
        # --- top view: x forward = up, y left = left
        ax0.scatter(-pts_b[:, 1], pts_b[:, 0], s=3, c='0.55', label='LiDAR returns')
        ax0.scatter(-pts_b[vis, 1], pts_b[vis, 0], s=5, c=rng_b[vis], cmap='viridis', vmin=0, vmax=10,
                    label='… seen by the camera')
        half = math.atan2(cam.width / 2, cam.fx)
        pitch_free = 12.0
        for sgn in (-1, 1):
            ax0.plot([-cam_xy[1], -(cam_xy[1] + sgn * pitch_free * math.tan(half))],
                     [cam_xy[0], cam_xy[0] + pitch_free], color='tab:orange', lw=0.8)
        ax0.add_patch(plt.Rectangle((-0.15, -0.25), 0.30, 0.55, fc='tab:blue', alpha=0.4))
        rows = []
        for cn, p in cones_in_base(k):
            b = pj.cone_box(cam, T_ob, p, cn['size'] / 2, a.cone_height)
            ax0.add_patch(plt.Circle((-p[1], p[0]), max(cn['size'] / 2, 0.08), fc='tab:red', ec='k'))
            ax0.annotate(cn['label'], (-p[1], p[0]), xytext=(6, 4), textcoords='offset points', fontsize=8)
            if b is None:
                continue
            u0, v0, u1, v1 = b['box']
            ax1.add_patch(plt.Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False, ec='tab:red', lw=1.6))
            inb = vis & pj.in_box(uv, b['box'], margin=2.0)
            d_cam = math.hypot(p[0] - cam_xy[0], p[1] - cam_xy[1])
            txt = f"{cn['label']} {d_cam:.1f} m"
            if inb.any():
                near = np.zeros_like(inb)
                near[np.nonzero(inb)[0][nearest_group(pts_b[inb])]] = True
                ax1.scatter(uv[near, 0], uv[near, 1], s=26, facecolors='none', edgecolors='tab:red', lw=1.2)
                est = pts_b[near].mean(0)[:2]
                err = math.hypot(*(est - p)) * 100
                foot, _ = cam.project(pj.apply(T_ob, [p[0], p[1], 0.0]))
                sens = pj.ground_sensitivity(cam, T_bo, foot)[0]
                rows.append((cn['label'], int(np.count_nonzero(inb)), int(np.count_nonzero(near)), err, d_cam, sens))
                txt += f'\n{np.count_nonzero(inb)} pts in box, {np.count_nonzero(near)} nearest'
            ax1.text(u0, v0 - 4, txt, color='tab:red', fontsize=8, va='bottom')
        for g in grid:
            guv, gok = cam.project(pj.apply(T_ob, g))
            ax1.plot(guv[gok, 0], guv[gok, 1], color='0.82', lw=0.7, zorder=0)
        huv, hok = cam.project(pj.apply(T_ob, hz))
        ax1.plot(huv[hok, 0], huv[hok, 1], '--', color='tab:blue', lw=0.8, zorder=0)
        sc = ax1.scatter(uv[vis, 0], uv[vis, 1], s=7, c=rng_b[vis], cmap='viridis', vmin=0, vmax=10, zorder=2)
        fig.colorbar(sc, ax=ax1, fraction=0.035, pad=0.02, label='LiDAR range [m]')
        ax1.set_xlim(-0.5, cam.width - 0.5)
        ax1.set_ylim(cam.height - 0.5, -0.5)
        ax1.set_aspect('equal')
        ax1.set_facecolor('#f7f7f4')
        ax1.set_title(f'camera picture ({cam.width} x {cam.height}): LiDAR returns + cone boxes (dashed: horizon)',
                      fontsize=9)
        ax0.set_xlim(-6, 6)
        ax0.set_ylim(-1.5, 10.5)
        ax0.set_aspect('equal')
        ax0.set_xlabel('left  <-  y [m]  ->  right')
        ax0.set_ylabel('x [m] (forward)')
        ax0.set_title(f'top view, scan {k} (orange: camera view)', fontsize=9)
        ax0.legend(loc='lower left', fontsize=7)
        fn = os.path.join(a.out, f'projection_demo_{label}.png')
        fig.tight_layout()
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        for r in rows:
            if r[0] == label:
                print(f'{r[0]:>10} {k:5d} {r[4]:5.2f}m {np.count_nonzero(vis):9d} {r[1]:6d} {r[2]:7d} {r[3]:8.1f}cm '
                      f'{r[5] * 100:16.1f} cm/px')
        print(f'           -> {fn}')


if __name__ == '__main__':
    main()
