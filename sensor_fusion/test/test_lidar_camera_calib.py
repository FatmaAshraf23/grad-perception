"""test_lidar_camera_calib.py -- the camera-LiDAR extrinsic solver on FAKE sessions with known answers.

Run:  python3 test/test_lidar_camera_calib.py   (plain runner)   or   python3 -m pytest test/test_lidar_camera_calib.py
Needs cv2 (solvePnP) and apex_description (the car mount YAML = the "truth"); else SKIP.
"""
import math
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
REPO = os.path.dirname(os.path.dirname(HERE))
from sensor_fusion import lidar_camera_calib as lc      # noqa: E402
from sensor_fusion import projection as pj              # noqa: E402
from sensor_fusion import synthetic as sy               # noqa: E402

try:
    import cv2  # noqa: F401
except ImportError:
    cv2 = None


class Skip(Exception):
    pass


CAM = pj.CameraModel([[748.472839, 0, 335.172741], [0, 1000.794742, 235.273704], [0, 0, 1]],
                     [0.138469, 0.185298, -0.007312, 0.008586, 0.0], 640, 480, 'iPhone 640x480')
OBJP = lc.board_object_points()
NAMES = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')


def setup():
    """The truth = the car mount YAML (LiDAR 0.096/0/0.122, camera 0.05/0/0.18 pitch 8)."""
    if cv2 is None:
        raise Skip('no cv2 (solvePnP)')
    path = os.environ.get('MOUNTS_CAR', os.path.join(REPO, 'apex_description', 'config', 'sensor_mounts_car.yaml'))
    try:
        from apex_description.mounts import load_mounts
    except ImportError:
        raise Skip('apex_description not importable')
    if not os.path.isfile(path):
        raise Skip(f'no {path}')
    m = load_mounts(path)
    Ts = pj.mount_transforms(m)
    T_bl, T_bo = Ts['laser'], Ts['camera_optical_frame']
    return m, T_bl, T_bo, pj.invert(T_bo) @ T_bl          # truth T_cam_from_laser


def guess(T_bl, T_bo, dx=0.04, dy=0.03, dz=-0.03, droll=3.0, dpitch=-2.0, dyaw=2.0):
    """A first guess = the true camera mount off by a few cm / degrees (like a CAD or tape measure)."""
    mt = lc.camera_mount(T_bo)
    T_bl_link = pj.T_from_xyz_rpy(mt[:3] + [dx, dy, dz], np.radians(mt[3:] + [droll, dpitch, dyaw]))
    T_bo_guess = T_bl_link @ pj._T(lc.R_LINK_OPT, (0, 0, 0))
    return pj.invert(T_bo_guess) @ T_bl


def err(res, T_bo):
    return res['mount'] - lc.camera_mount(T_bo)


def test_plane_from_pnp_noise_free():
    _, T_bl, T_bo, _ = setup()
    T_bb = sy.board_pose((1.5, 0.1, 0.12), 20, 15, OBJP)
    T_cb = pj.invert(T_bo) @ T_bb
    uv, ok = CAM.project(pj.apply(T_cb, OBJP))
    R, t, rms = lc.board_pose_pnp(uv, CAM, OBJP)
    n, d = lc.plane_from_pose(R, t)
    n0, d0 = lc.plane_from_pose(T_cb[:3, :3], T_cb[:3, 3])
    print(f'    noise-free corners -> PnP rms {rms:.1e} px, plane normal error {np.degrees(np.arccos(min(1, n @ n0))):.1e} '
          f'deg, distance error {abs(d - d0) * 1000:.1e} mm')
    assert rms < 1e-6 and abs(d - d0) < 1e-6 and n @ n0 > 1 - 1e-12


def test_noise_free_session_exact():
    _, T_bl, T_bo, T_true = setup()
    poses = sy.make_session(CAM, T_bo, T_bl, sy.default_specs(T_bl[2, 3]), OBJP, sigma_px=0.0, sigma_r=0.0, n_scans=1)
    res = lc.calibrate(poses, CAM, guess(T_bl, T_bo), T_base_laser=T_bl, n_boot=0)
    e = err(res, T_bo)
    print(f'    {len(res["obs"])} poses, no noise, guess off 4/3/3 cm 3/2/2 deg -> errors '
          + ', '.join(f'{nm} {v:+.1e}' for nm, v in zip(NAMES, e)) + f'; residual rms {res["info"]["rms"]:.1e} m')
    assert np.abs(e[:3]).max() < 1e-6 and np.abs(e[3:]).max() < 1e-4


def test_linear_init_noise_free():
    _, T_bl, T_bo, T_true = setup()
    poses = sy.make_session(CAM, T_bo, T_bl, sy.default_specs(T_bl[2, 3]), OBJP, sigma_px=0.0, sigma_r=0.0, n_scans=1)
    res = lc.calibrate(poses, CAM, T_true, T_base_laser=T_bl, n_boot=0)         # associations with the truth
    T_lin, cond = lc.linear_init(res['obs'])
    e_t = np.linalg.norm(T_lin[:3, 3] - T_true[:3, 3])
    e_r = lc.rot_angle_deg(T_lin[:3, :3].T @ T_true[:3, :3])
    print(f'    Zhang-Pless closed form, no noise: error {e_t * 1000:.1e} mm, {e_r:.1e} deg (condition {cond:.0f})')
    assert e_t < 1e-6 and e_r < 1e-4


def test_realistic_session():
    _, T_bl, T_bo, _ = setup()
    poses = sy.make_session(CAM, T_bo, T_bl, sy.default_specs(T_bl[2, 3]), OBJP, sigma_px=0.3, sigma_r=0.01,
                            n_scans=10, seed=3)
    res = lc.calibrate(poses, CAM, guess(T_bl, T_bo), T_base_laser=T_bl, n_boot=200)
    e = err(res, T_bo)
    for line in res['log'] + lc.report(res, lc.camera_mount(T_bo)):
        print('    ' + line)
    assert not res['weak'], f'weak: {res["weak"]}'
    assert np.abs(e[:3]).max() < 0.01 and np.abs(e[3:]).max() < 0.5, 'more than 1 cm / 0.5 deg off'
    z = np.abs(e) / res['std']
    print('    error / spread: ' + ', '.join(f'{nm} {v:.1f}' for nm, v in zip(NAMES, z)) + '  (honest spread: < ~3)')
    assert z.max() < 4.0, 'the reported spread is not honest'


def test_vertical_boards_only_are_flagged():
    _, T_bl, T_bo, _ = setup()
    poses = sy.make_session(CAM, T_bo, T_bl, sy.default_specs(T_bl[2, 3], tilted=False), OBJP, sigma_px=0.3,
                            sigma_r=0.01, n_scans=10, seed=3)
    res = lc.calibrate(poses, CAM, guess(T_bl, T_bo), T_base_laser=T_bl, n_boot=200)
    e = err(res, T_bo)
    print('    ONLY VERTICAL boards (turned, never tilted): spread ' +
          ', '.join(f'{nm} {s:.4f}' for nm, s in zip(NAMES, res['std'])) + f'\n    -> flagged WEAK: {res["weak"]}'
          + '; errors ' + ', '.join(f'{nm} {v:+.4f}' for nm, v in zip(NAMES, e)))
    assert res['weak'], 'vertical-only boards must be flagged'


def test_no_wall_points_on_the_boards():
    _, T_bl, T_bo, T_true = setup()
    poses = sy.make_session(CAM, T_bo, T_bl, sy.default_specs(T_bl[2, 3]), OBJP, sigma_px=0.3, sigma_r=0.01,
                            n_scans=10, seed=3)
    res = lc.calibrate(poses, CAM, guess(T_bl, T_bo), T_base_laser=T_bl, n_boot=0)
    worst = 0.0
    for o in res['obs']:
        T_bb = poses[o['pose']]['T_base_board']
        Xb = pj.apply(pj.invert(T_bb) @ T_bl, o['pts'])
        worst = max(worst, float(np.abs(Xb[:, 2]).max()))
    total = sum(len(p['scan_points']) for p in poses)
    used = sum(len(o['pts']) for o in res['obs'])
    print(f'    {used} of {total} LiDAR returns used; the farthest of them from its TRUE board: {worst * 1000:.0f} mm '
          f'(range noise 10 mm: no wall points)')
    assert worst < 0.05


def test_far_first_guess_and_backwards_lidar():
    _, T_bl, T_bo, _ = setup()
    poses = sy.make_session(CAM, T_bo, T_bl, sy.default_specs(T_bl[2, 3]), OBJP, sigma_px=0.3, sigma_r=0.01,
                            n_scans=10, seed=3)
    res = lc.calibrate(poses, CAM, guess(T_bl, T_bo, 0.12, -0.10, 0.08, 8.0, -8.0, 10.0), T_base_laser=T_bl, n_boot=0)
    e = err(res, T_bo)
    print('    first guess 12/10/8 cm and 8/8/10 deg off -> errors ' + ', '.join(f'{nm} {v:+.4f}' for nm, v in zip(NAMES, e)))
    assert np.abs(e[:3]).max() < 0.01 and np.abs(e[3:]).max() < 0.5
    for deg in (180.0, 90.0, -90.0, 45.0):                         # the LiDAR's 0 degrees turned the wrong way
        flip = pj.T_from_xyz_rpy((0, 0, 0), (0, 0, math.radians(deg)))
        try:
            r = lc.calibrate(poses, CAM, guess(T_bl, T_bo) @ flip, T_base_laser=T_bl, n_boot=0)
            raise AssertionError(f'a LiDAR yaw {deg:.0f} deg off must not give a result (got {np.round(r["mount"], 3)})')
        except ValueError as ex:
            print(f'    LiDAR yaw {deg:+.0f} deg off -> refused: {str(ex)[:110]}...')


if __name__ == '__main__':
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
    n_fail = n_skip = 0
    for name, t in tests:
        try:
            t()
            print(f'PASS {name}')
        except Skip as ex:
            n_skip += 1
            print(f'SKIP {name}: {ex}')
        except Exception:
            n_fail += 1
            print(f'FAIL {name}')
            traceback.print_exc()
    print(f'RESULT {len(tests) - n_fail - n_skip} passed, {n_fail} failed, {n_skip} skipped')
    sys.exit(1 if n_fail else 0)
