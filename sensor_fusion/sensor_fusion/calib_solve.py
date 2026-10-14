"""calib_solve -- solve a recorded camera-LiDAR calibration session (a calib_capture folder).

  ros2 run sensor_fusion calib_solve ~/calib_sessions/<session> --mounts car

First guess = the camera + LiDAR numbers of the mount YAML (tape measure / CAD); the LiDAR's own mount (base_link ->
laser) is taken as measured. Prints the per-pose table, the camera's mount relative to base_link +- its spread
(WEAK = not determined by these poses: record more, more spread-out and more TILTED poses) and the lines for
apex_description/config/sensor_mounts_car.yaml. Writes into the session folder:
  result.yaml       the camera numbers, spreads, WEAK flags, residual, the poses used
  overlay_XX.png    each pose's picture with its LiDAR points drawn with the FIRST GUESS (red) and the RESULT (blue);
                    blue must lie ON the board, along the straight line where the LiDAR's scan crosses it
  overlays.png      all of them on one sheet
Exit code 0 = ok, 3 = result with WEAK numbers, 1 = refused (the message says why).
"""
import argparse
import os
import sys
import time

import numpy as np

from sensor_fusion import lidar_camera_calib as lc
from sensor_fusion import projection as pj
from sensor_fusion.calib_session import Session

NAMES = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')


def overlay(img, cam, corners, pts_all, pts_board, T0, T, pattern, label):
    """The check picture: board corners (small green dots), every LiDAR return drawn with the RESULT (grey), the
    board's LiDAR points with the FIRST GUESS (red crosses) and with the RESULT (blue dots, white rim). Blue must lie
    ON the board, along one straight line where the LiDAR's scan crosses it."""
    import cv2
    out = img.copy()
    for u, v in np.asarray(corners, float).reshape(-1, 2):
        cv2.circle(out, (int(round(u)), int(round(v))), 2, (0, 200, 0), -1)
    uv, ok = cam.project(pj.apply(T, pts_all))
    for u, v in uv[ok & cam.in_image(uv)]:
        cv2.circle(out, (int(round(u)), int(round(v))), 1, (170, 170, 170), -1)
    uv, ok = cam.project(pj.apply(T0, pts_board))
    for u, v in uv[ok & cam.in_image(uv)]:
        cv2.drawMarker(out, (int(round(u)), int(round(v))), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 9, 2)
    uv, ok = cam.project(pj.apply(T, pts_board))
    for u, v in uv[ok & cam.in_image(uv)]:
        cv2.circle(out, (int(round(u)), int(round(v))), 4, (255, 255, 255), -1)
        cv2.circle(out, (int(round(u)), int(round(v))), 3, (255, 90, 0), -1)
    cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, 'green = board corners, red x = LiDAR with the first guess, blue = with the result',
                (6, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('session', help='the calib_capture session folder')
    ap.add_argument('--mounts', default='car', help="first guess: 'car', 'sim' or a mount YAML path")
    ap.add_argument('--truth', default='', help='a mount YAML with the TRUE numbers (fake sessions): prints errors')
    ap.add_argument('--n-boot', type=int, default=200)
    ap.add_argument('--no-images', action='store_true')
    a = ap.parse_args()
    from apex_description.mounts import load_mounts

    folder = os.path.expanduser(a.session)
    s = Session.load(folder)
    cam = pj.CameraModel(s.K, s.D, s.width, s.height)
    m = load_mounts(a.mounts)
    Ts = pj.mount_transforms(m)
    T_bl, T_bo = Ts['laser'], Ts['camera_optical_frame']
    T0 = pj.invert(T_bo) @ T_bl
    poses = s.calib_poses()
    print(f'session {folder}: {len(poses)} poses, {sum(p["n_scans"] for p in poses)} LiDAR scans; camera '
          f'{s.width} x {s.height}; board {s.pattern[0]} x {s.pattern[1]}, squares {s.square[0] * 1000:.2f} x '
          f'{s.square[1] * 1000:.2f} mm; first guess {m["_path"]} (status {m["status"]})')
    if np.isfinite(s.T_tf).all():
        d = pj.invert(T0) @ s.T_tf
        print(f'  TF at capture time vs the first guess: {np.linalg.norm(d[:3, 3]) * 100:.1f} cm, '
              f'{lc.rot_angle_deg(d[:3, :3]):.1f} deg apart' + ('  (different YAML used during capture?)'
                                                                  if np.linalg.norm(d[:3, 3]) > 0.005 else ''))
    objp = lc.board_object_points(s.pattern, s.square)
    try:
        res = lc.calibrate(poses, cam, T0, T_base_laser=T_bl, objp=objp, n_boot=a.n_boot)
    except ValueError as e:
        print(f'REFUSED: {e}')
        sys.exit(1)
    truth = None
    if a.truth:
        truth = lc.camera_mount(pj.mount_transforms(load_mounts(a.truth))['camera_optical_frame'])
    for line in res['log'] + lc.report(res, truth):
        print(line)
    if res['weak']:
        print(f"WARNING: {', '.join(res['weak'])} not determined by these poses -- record more, more spread-out and "
              'more TILTED poses before using the result')
    mt, sd = res['mount'], res['std']
    with open(os.path.join(folder, 'result.yaml'), 'w') as f:
        f.write(f'# calib_solve {time.strftime("%Y-%m-%d %H:%M")}: session {folder}, first guess {m["_path"]}\n')
        f.write('camera: {' + ', '.join(f'{nm}: {v:.4f}' for nm, v in zip(NAMES, mt)) + '}\n')
        f.write('spread: {' + ', '.join(f'{nm}: {v:.4f}' for nm, v in zip(NAMES, sd)) + '}\n')
        f.write(f"weak: [{', '.join(res['weak'])}]\n")
        f.write(f"residual_rms_m: {res['info']['rms']:.4f}\n")
        f.write(f"poses_used: [{', '.join(str(o['pose']) for o in res['obs'])}]\n")
        f.write(f'poses_recorded: {len(poses)}\n')
        if truth is not None:
            f.write('error_vs_truth: {' + ', '.join(f'{nm}: {v:+.4f}' for nm, v in zip(NAMES, mt - truth)) + '}\n')
    print(f"-> {os.path.join(folder, 'result.yaml')}")
    if a.no_images:
        sys.exit(3 if res['weak'] else 0)
    import cv2
    used = {o['pose']: o['pts'] for o in res['obs']}
    tiles = []
    for k, p in enumerate(poses):
        fn = os.path.join(folder, f'pose_{k:02d}.png')
        if not os.path.isfile(fn):
            continue
        img = cv2.imread(fn)
        label = f'pose {k}' + ('' if k in used else ' (NOT used)')
        ov = overlay(img, cam, p['corners'], p['scan_points'], used.get(k, np.zeros((0, 3))), T0, res['T'],
                     s.pattern, label)
        cv2.imwrite(os.path.join(folder, f'overlay_{k:02d}.png'), ov)
        tiles.append(cv2.resize(ov, (ov.shape[1] // 2, ov.shape[0] // 2), interpolation=cv2.INTER_AREA))
    if tiles:
        cols = 4
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(os.path.join(folder, 'overlays.png'), sheet)
        print(f"-> {len(poses)} overlay pictures + {os.path.join(folder, 'overlays.png')}")
    sys.exit(3 if res['weak'] else 0)


if __name__ == '__main__':
    main()
