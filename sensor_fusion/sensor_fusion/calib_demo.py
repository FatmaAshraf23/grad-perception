"""calib_demo -- a FAKE camera-LiDAR calibration session, solved and drawn: what you will do on the real car.

16 board poses (1.1-2.0 m ahead, left / centre / right of the picture, turned AND tilted), the corner pixels the
camera detects (+0.3 px noise), 10 LiDAR scans per pose (+1 cm noise) in a 6 x 5 m room, a first guess off by a few
cm / degrees -> lidar_camera_calib.calibrate() -> the camera's mount vs the truth (sensor_mounts_car.yaml).
Figure: top view (boards, LiDAR returns, the ones used) + the camera picture of 2 poses: the LiDAR points on the
board drawn with the FIRST GUESS (red) and with the RESULT (blue) -- blue must lie on the board, on the line where
the LiDAR's scan plane crosses it (green dashed = the truth).
  ros2 run sensor_fusion calib_demo [--vertical-only] [--out DIR] [--camera <camera_info yaml>]
"""
import argparse
import os

import numpy as np

from sensor_fusion import lidar_camera_calib as lc
from sensor_fusion import projection as pj
from sensor_fusion import synthetic as sy

IPHONE_640 = ([[748.472839, 0, 335.172741], [0, 1000.794742, 235.273704], [0, 0, 1]],
              [0.138469, 0.185298, -0.007312, 0.008586, 0.0], 640, 480)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mounts', default='car', help='the TRUTH (car / sim / path)')
    ap.add_argument('--camera', default='', help='camera_info YAML (default: the iPhone 640x480)')
    ap.add_argument('--vertical-only', action='store_true', help='boards never tilted (the degenerate case)')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from apex_description.mounts import load_mounts

    cam = pj.CameraModel.from_yaml(a.camera) if a.camera else pj.CameraModel(*IPHONE_640)
    m = load_mounts(a.mounts)
    Ts = pj.mount_transforms(m)
    T_bl, T_bo = Ts['laser'], Ts['camera_optical_frame']
    objp = lc.board_object_points()
    truth = lc.camera_mount(T_bo)
    poses = sy.make_session(cam, T_bo, T_bl, sy.default_specs(T_bl[2, 3], tilted=not a.vertical_only), objp,
                            sigma_px=0.3, sigma_r=0.01, n_scans=10, seed=a.seed)
    # the first guess: the true camera mount off by 4 / 3 / -3 cm and 3 / -2 / 2 deg
    T_link = pj.T_from_xyz_rpy(truth[:3] + [0.04, 0.03, -0.03], np.radians(truth[3:] + [3.0, -2.0, 2.0]))
    T0 = pj.invert(T_link @ pj._T(lc.R_LINK_OPT, (0, 0, 0))) @ T_bl
    res = lc.calibrate(poses, cam, T0, T_base_laser=T_bl, n_boot=200)
    tag = 'vertical_only' if a.vertical_only else 'tilted'
    print(f'{len(poses)} poses ({tag}), camera {cam.width} x {cam.height}; truth = {m["_path"]}')
    for line in res['log'] + lc.report(res, truth):
        print(line)

    T_ol_true = pj.invert(T_bo) @ T_bl
    fig = plt.figure(figsize=(15, 5.4))
    ax0 = fig.add_subplot(1, 3, 1)
    used = {o['pose']: o['pts'] for o in res['obs']}
    cmap = plt.get_cmap('tab20')
    for k, p in enumerate(poses):
        first = pj.apply(T_bl, p['scan_points'][: len(p['scan_points']) // 10])     # the first of the 10 scans
        ax0.scatter(-first[:, 1], first[:, 0], s=2, c='0.75')
        lo, hi = objp[:, :2].min(0) - lc.BOARD_MARGIN, objp[:, :2].max(0) + lc.BOARD_MARGIN
        rect = np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0], [hi[0], hi[1], 0], [lo[0], hi[1], 0], [lo[0], lo[1], 0]])
        rb = pj.apply(p['T_base_board'], rect)
        ax0.plot(-rb[:, 1], rb[:, 0], color=cmap(k % 20), lw=1.0)
        if k in used:
            u = pj.apply(T_bl, used[k])
            ax0.scatter(-u[:, 1], u[:, 0], s=3, color=cmap(k % 20))
    ax0.plot([0], [T_bl[0, 3]], 'k^', ms=7)
    ax0.set_xlim(-1.6, 1.6)
    ax0.set_ylim(-0.2, 2.6)
    ax0.set_aspect('equal')
    ax0.set_xlabel('left  <-  y [m]  ->  right')
    ax0.set_ylabel('x [m] (forward)')
    ax0.set_title('top view: boards (outlines), LiDAR (grey), on the boards (colour)', fontsize=9)
    used_poses = [o['pose'] for o in res['obs']]               # the leftmost and the rightmost pose in the picture
    mean_u = [float(poses[k]['corners'][:, 0].mean()) for k in used_poses]
    show = [used_poses[int(np.argmin(mean_u))], used_poses[int(np.argmax(mean_u))]]
    for j, k in enumerate(show):
        ax = fig.add_subplot(1, 3, 2 + j)
        p = poses[k]
        ax.scatter(p['corners'][:, 0], p['corners'][:, 1], s=6, c='tab:green', label='board corners (detected)')
        pts = used[k]
        for T, col, mk, lab in ((T0, 'tab:red', 'x', 'LiDAR with the FIRST GUESS'),
                                (res['T'], 'tab:blue', 'o', 'LiDAR with the RESULT')):
            uv, ok = cam.project(pj.apply(T, pts))
            ax.scatter(uv[ok, 0], uv[ok, 1], s=10, c=col, marker=mk, label=lab)
        line = np.linspace(pts[:, :2].min(0), pts[:, :2].max(0), 30)
        uvt, okt = cam.project(pj.apply(T_ol_true, np.column_stack([line, np.zeros(30)])))
        ax.plot(uvt[okt, 0], uvt[okt, 1], '--', color='tab:green', lw=1, label='truth: where the scan crosses')
        ax.set_xlim(-0.5, cam.width - 0.5)
        ax.set_ylim(cam.height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_facecolor('#f7f7f4')
        d, lat, zc, turn, tilt = p['spec']
        ax.set_title(f'camera, pose {k}: {d:.1f} m, turned {turn} deg, tilted {tilt} deg', fontsize=9)
        if j == 0:
            ax.legend(loc='lower left', fontsize=7)
    e = res['mount'] - truth
    fig.suptitle(f'camera-LiDAR calibration on a fake session ({tag}): camera position error '
                 f'{np.abs(e[:3]).max() * 1000:.1f} mm, angle error {np.abs(e[3:]).max():.2f} deg'
                 + (f'   WEAK: {", ".join(res["weak"])}' if res['weak'] else ''), fontsize=10)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f'calib_demo_{tag}.png')
    fig.savefig(fn, dpi=110)
    print(f'-> {fn}')


if __name__ == '__main__':
    main()
