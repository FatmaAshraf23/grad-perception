"""test_render_session.py -- the synthetic camera pictures (render.py) and the session file (calib_session.py).
Run: python3 test/test_render_session.py   (or pytest). Needs cv2."""
import os
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sensor_fusion import lidar_camera_calib as lc   # noqa: E402
from sensor_fusion import projection as pj           # noqa: E402
from sensor_fusion import synthetic as sy            # noqa: E402
from sensor_fusion.calib_session import Session      # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None


class Skip(Exception):
    pass


CAM = pj.CameraModel([[748.472839, 0, 335.172741], [0, 1000.794742, 235.273704], [0, 0, 1]],
                     [0.138469, 0.185298, -0.007312, 0.008586, 0.0], 640, 480, 'iPhone 640x480')
OBJP = lc.board_object_points()


def test_rendered_board_is_found_where_it_should_be():
    if cv2 is None:
        raise Skip('no cv2')
    from sensor_fusion.render import Renderer
    rend = Renderer(CAM, seed=1)
    T_cam_base = pj.T_from_xyz_rpy((0, 0, 0), (0, 0, 0))
    # camera looking along +x of a 'base' frame: build T_cam_board directly in the optical frame
    worst = []
    for dist, turn, tilt in ((1.1, -30, -20), (1.4, 20, 15), (1.7, 0, 25), (2.0, -35, -10)):
        T_ob = pj.invert(pj._T(lc.R_LINK_OPT, (0, 0, 0)))                 # optical <- link (car-like axes)
        T_bb = sy.board_pose((dist, 0.05, 0.0), turn, tilt, OBJP)          # board in the link frame
        T_cb = T_ob @ T_bb @ T_cam_base
        img = rend.render(T_cb)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])     # like the camera stream
        gray = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        ok, c = cv2.findChessboardCornersSB(gray, (9, 6), flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY)
        assert ok, f'board not found at {dist} m, turned {turn}, tilted {tilt}'
        truth, _ = CAM.project(pj.apply(T_cb, OBJP))
        c = c.reshape(-1, 2)
        e = np.linalg.norm(c - truth, axis=1)
        if np.median(e) > 2.0:                                             # found the other way round (180 deg)?
            e = np.linalg.norm(c[::-1] - truth, axis=1)
        R, t, rms = lc.board_pose_pnp(c if np.median(np.linalg.norm(c - truth, axis=1)) < 2 else c[::-1], CAM, OBJP)
        n, d = lc.plane_from_pose(R, t)
        n0, d0 = lc.plane_from_pose(T_cb[:3, :3], T_cb[:3, 3])
        ang = np.degrees(np.arccos(min(1.0, abs(n @ n0))))
        print(f'    {dist} m, turned {turn:+d}, tilted {tilt:+d}: corner error median {np.median(e):.2f} px, max '
              f'{e.max():.2f} px; board plane from PnP: {ang:.2f} deg, {abs(d - d0) * 1000:.1f} mm off')
        worst.append((np.median(e), e.max(), ang, abs(d - d0)))
    w = np.max(worst, 0)
    # limits from the supersampled renderer (2026-10-14: median 0.14-0.19 px, plane <= 0.2 deg / 2 mm; without
    # supersampling the far boards' corners were ~0.3 px and the plane up to 9 mm off)
    assert w[0] < 0.25 and w[1] < 0.6 and w[2] < 0.5 and w[3] < 0.005


def test_session_round_trip():
    s = Session(CAM.K, CAM.D, 640, 480, camera_frame='camera_optical_frame', laser_frame='laser')
    rng = np.random.default_rng(0)
    meta = (-np.pi, 2 * np.pi / 360, 0.15, 12.0)
    for k in range(3):
        scans = [(10.0 + k + 0.1 * j, rng.uniform(0.5, 5, 360).astype(np.float32), meta) for j in range(4 + k)]
        s.add_pose(rng.uniform(0, 600, (54, 2)), 10.0 + k, scans)
    with tempfile.TemporaryDirectory() as d:
        s.save(d)
        t = Session.load(d)
    assert np.allclose(t.K, s.K) and t.camera_frame == 'camera_optical_frame' and t.laser_frame == 'laser'
    assert len(t.corners) == 3 and all(np.allclose(a, b) for a, b in zip(t.corners, s.corners))
    a, b = s.calib_poses(), t.calib_poses()
    assert [p['n_scans'] for p in b] == [4, 5, 6]
    assert all(np.allclose(p['scan_points'], q['scan_points']) for p, q in zip(a, b))
    print(f"    3 poses, {sum(p['n_scans'] for p in b)} scans saved + loaded identically; "
          f"{sum(len(p['scan_points']) for p in b)} LiDAR points")


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
