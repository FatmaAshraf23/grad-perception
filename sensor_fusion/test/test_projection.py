"""test_projection.py -- sensor_fusion.projection against OpenCV, hand-computed answers and the classic traps.

Run:  python3 test/test_projection.py        (plain runner, prints PASS / FAIL / SKIP per test)
      python3 -m pytest test/test_projection.py
Optional files (else those tests SKIP): env CALIB_2560 / CALIB_640 (camera_info YAMLs), MOUNTS_SIM (sensor mount
YAML); defaults = the repo paths next to this package.
"""
import math
import os
import sys
import traceback
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # this package's source
REPO = os.path.dirname(os.path.dirname(HERE))                # ros2_ws/src
from sensor_fusion import projection as pj                    # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None


class Skip(Exception):
    pass


# the iPhone dev camera as the ROS stream delivers it (camera_publisher/config/iphone_iriun_640x480.yaml)
IPHONE = pj.CameraModel([[748.472839, 0, 335.172741], [0, 1000.794742, 235.273704], [0, 0, 1]],
                        [0.138469, 0.185298, -0.007312, 0.008586, 0.0], 640, 480, 'iPhone 640x480')
# a wide-angle lens with strong barrel distortion: its polynomial turns back (folds) just outside the picture
WIDE = pj.CameraModel([[500, 0, 320], [0, 500, 240], [0, 0, 1]], [-0.2, 0, 0, 0, 0], 640, 480, 'wide k1=-0.2')
P_OPT = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return {'x': np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), 'y': np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            'z': np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def sim_mount():
    path = os.environ.get('MOUNTS_SIM', os.path.join(REPO, 'apex_description', 'config', 'sensor_mounts_sim.yaml'))
    try:
        from apex_description.mounts import load_mounts
    except ImportError:
        raise Skip('apex_description not importable')
    if not os.path.isfile(path):
        raise Skip(f'no {path}')
    m = load_mounts(path)
    return m, pj.mount_transforms(m)


def test_project_matches_opencv():
    if cv2 is None:
        raise Skip('no cv2')
    rng = np.random.default_rng(1)
    for cam in (IPHONE, WIDE):
        n = 5000
        r = cam.r_max * np.sqrt(rng.random(n))
        a = 2 * math.pi * rng.random(n)
        z = rng.uniform(0.3, 30.0, n)
        P = np.column_stack([r * np.cos(a) * z, r * np.sin(a) * z, z])
        uv, ok = cam.project(P)
        assert ok.all(), 'points inside the guard must project'
        ref, _ = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam.K, cam.D)
        err = np.abs(uv - ref.reshape(-1, 2)).max()
        print(f'    {cam.name}: {n} random points, largest difference to cv2.projectPoints {err:.1e} px')
        assert err < 1e-6


def test_pixel_ray_round_trip():
    for cam in (IPHONE, WIDE):
        us, vs = np.meshgrid(np.arange(0.0, cam.width, 8.0), np.arange(0.0, cam.height, 8.0))
        uv = np.column_stack([us.ravel(), vs.ravel()])
        rays, ok = cam.pixel_to_ray(uv)
        assert ok.all(), f'{cam.name}: {np.count_nonzero(~ok)} pixels without a ray'
        back, ok2 = cam.project(rays)
        assert ok2.all()
        err = np.abs(back - uv).max()
        msg = f'    {cam.name}: {len(uv)} pixels -> ray -> pixel, largest error {err:.1e} px'
        if cv2 is not None:
            mine = rays[:, :2] / rays[:, 2:3]
            crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 300, 1e-15)
            try:
                ref = cv2.undistortPointsIter(uv.reshape(-1, 1, 2), cam.K, cam.D, None, None, crit).reshape(-1, 2)
                e2 = np.abs(mine - ref).max()
                msg += f'; vs cv2.undistortPointsIter {e2:.1e} (normalized)'
                assert e2 < 1e-6
            except AttributeError:
                msg += '; (no cv2.undistortPointsIter)'
        print(msg)
        assert err < 1e-6


def test_behind_camera_and_too_close_rejected():
    P = np.array([[0.5, 0.2, -2.0], [0.0, 0.0, -1.0], [0.1, 0.0, 0.01], [0.0, 0.0, 2.0]])
    uv, ok = IPHONE.project(P)
    assert list(ok) == [False, False, False, True]
    assert np.isnan(uv[:3]).all() and np.isfinite(uv[3]).all()
    nu = IPHONE.fx * P[0, 0] / P[0, 2] + IPHONE.cx
    nv = IPHONE.fy * P[0, 1] / P[0, 2] + IPHONE.cy
    print(f'    a point 2 m BEHIND the camera: plain pinhole -> pixel ({nu:.0f}, {nv:.0f}) = INSIDE the picture '
          f'(a ghost); projection.py -> rejected')
    assert 0 <= nu < 640 and 0 <= nv < 480


def test_fov_guard_stops_fold_back():
    P = np.array([[2.0, 0.0, 1.0]])                          # 63.4 deg off-axis; WIDE's picture edge is ~35.5 deg
    uv, ok = WIDE.project(P)
    assert not ok[0]
    if cv2 is not None:
        ref, _ = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), WIDE.K, WIDE.D)
        u, v = ref.ravel()
        print(f'    wide lens, a point 63 deg off-axis (outside the view): cv2.projectPoints -> ({u:.0f}, {v:.0f}) '
              f'INSIDE the picture (folded back); projection.py -> rejected (guard radius {WIDE.r_max:.3f})')
        assert 0 <= u < 640 and 0 <= v < 480
    uv, ok = WIDE.project(np.array([[0.5, 0.2, 1.0]]))      # 28 deg: in view -> kept
    assert ok[0] and WIDE.in_image(uv)[0]
    bad = pj.CameraModel([[500, 0, 320], [0, 500, 240], [0, 0, 1]], [-0.35, 0, 0, 0, 0], 640, 480)
    print(f'    a lens that folds INSIDE its own picture (k1 = -0.35): fov_guard_ok = {bad.fov_guard_ok} (want False)')
    assert not bad.fov_guard_ok and IPHONE.fov_guard_ok and WIDE.fov_guard_ok


def test_scaled_2560_matches_published_640x480():
    p2560 = os.environ.get('CALIB_2560', os.path.join(REPO, 'tools', 'camera_calibration', 'results',
                                                      'calibration_2560x1440_k3fixed_sq.yaml'))
    p640 = os.environ.get('CALIB_640', os.path.join(REPO, 'camera_publisher', 'config', 'iphone_iriun_640x480.yaml'))
    if not (os.path.isfile(p2560) and os.path.isfile(p640)):
        raise Skip('calibration YAMLs not found')
    big, small = pj.CameraModel.from_yaml(p2560), pj.CameraModel.from_yaml(p640)
    s = big.scaled(640, 480)
    e = np.abs(s.K - small.K).max()
    print(f'    2560x1440 calibration scaled to 640x480 vs the published camera_info: largest K difference {e:.1e}')
    assert e < 1e-4 and np.abs(s.D - small.D).max() < 1e-9


def test_sim_mount_semantics():
    m, Ts = sim_mount()
    c = m['camera']
    T_bo = Ts['camera_optical_frame']
    T_ob = pj.invert(T_bo)
    R_want = P_OPT @ rot('x', -c['roll']) @ rot('y', -c['pitch']) @ rot('z', -c['yaw'])
    assert np.abs(T_ob[:3, :3] - R_want).max() < 1e-12, 'mount rotation != elementary matrices'
    assert np.abs(T_bo[:3, 3] - [c['x'], c['y'], c['z']]).max() < 1e-12
    pin = pj.CameraModel(IPHONE.K, [0] * 5, 640, 480)
    pitch = math.radians(c['pitch'])
    uv, ok = pin.project(pj.apply(T_ob, [1000.0, 0.0, c['z']]))           # far ahead at the camera's height
    v_h = pin.cy - pin.fy * math.tan(pitch)
    print(f"    camera pitched {c['pitch']} deg down: far straight ahead -> pixel ({uv[0, 0]:.2f}, {uv[0, 1]:.2f}), "
          f'hand-computed ({pin.cx:.2f}, {v_h:.2f}) = cy - fy*tan(pitch)')
    assert ok[0] and abs(uv[0, 0] - pin.cx) < 1e-6 and abs(uv[0, 1] - v_h) < 1e-6
    x0 = c['x'] + c['z'] / math.tan(pitch)                                  # where the optical axis meets the floor
    uv, ok = IPHONE.project(pj.apply(T_ob, [x0, 0.0, 0.0]))
    g, okg = pj.ground_point(IPHONE, T_bo, [[IPHONE.cx, IPHONE.cy]])
    print(f'    the optical axis meets the floor {x0:.4f} m ahead of base_link: projects to ({uv[0, 0]:.4f}, '
          f'{uv[0, 1]:.4f}) = (cx, cy); the centre pixel back on the floor: x {g[0, 0]:.4f} y {g[0, 1]:.4f}')
    assert np.abs(uv[0] - [IPHONE.cx, IPHONE.cy]).max() < 1e-9 and np.abs(g[0] - [x0, 0, 0]).max() < 1e-9
    uvl, _ = IPHONE.project(pj.apply(T_ob, [3.0, 0.5, 0.0]))
    uvr, _ = IPHONE.project(pj.apply(T_ob, [3.0, -0.5, 0.0]))
    uvt, _ = IPHONE.project(pj.apply(T_ob, [3.0, 0.0, 0.4]))
    assert uvl[0, 0] < IPHONE.cx < uvr[0, 0], 'left of the car must be LEFT in the picture'
    assert uvt[0, 1] < IPHONE.project(pj.apply(T_ob, [3.0, 0.0, 0.0]))[0][0, 1], 'higher must be UP in the picture'


def test_ground_distance_sensitivity():
    m, Ts = sim_mount()
    c = m['camera']
    T_bo = Ts['camera_optical_frame']
    pin = pj.CameraModel(IPHONE.K, [0] * 5, 640, 480)
    pitch, h = math.radians(c['pitch']), c['z']
    v_h = pin.cy - pin.fy * math.tan(pitch)
    _, ok = pj.ground_point(pin, T_bo, [[pin.cx, v_h - 5.0]])
    assert not ok[0], 'a pixel above the horizon has no ground point'
    for d in (2.0, 5.0, 10.0):                       # metres from the camera along the floor, straight ahead
        uv, _ = pin.project(pj.apply(pj.invert(T_bo), [c['x'] + d, 0.0, 0.0]))
        s = pj.ground_sensitivity(pin, T_bo, uv)[0]
        alpha = math.atan(h / d)
        expect = (d * d + h * h) / (h * pin.fy) * math.cos(alpha - pitch) ** 2
        print(f'    {d:4.1f} m ahead: 1 pixel up/down in the picture = {s * 100:5.1f} cm of distance '
              f'(hand formula (d^2+h^2)/(h*fy)*cos^2: {expect * 100:5.1f} cm)')
        assert abs(s - expect) / expect < 0.02


def test_cone_box():
    m, Ts = sim_mount()
    T_ob = pj.invert(Ts['camera_optical_frame'])
    cx = m['camera']['x'] + 3.0                      # 3 m ahead of the camera
    b = pj.cone_box(IPHONE, T_ob, (cx, 0.0), radius=0.10, height=0.30)
    u0, v0, u1, v1 = b['box']
    tip, _ = IPHONE.project(pj.apply(T_ob, [cx, 0.0, 0.30]))
    foot, _ = IPHONE.project(pj.apply(T_ob, [cx, 0.0, 0.0]))
    print(f'    cone r 0.10 m, h 0.30 m, 3 m ahead: box u {u0:.1f}-{u1:.1f} ({u1 - u0:.1f} px wide), v {v0:.1f}-{v1:.1f} '
          f'({v1 - v0:.1f} px high); tip at v {tip[0, 1]:.1f}')
    assert b['all_visible'] and abs((u0 + u1) / 2 - IPHONE.cx) < 2.0
    assert 45.0 < u1 - u0 < 56.0 and abs(tip[0, 1] - v0) < 1.0 and v0 < foot[0, 1] < v1
    assert pj.cone_box(IPHONE, T_ob, (-3.0, 0.0), 0.1, 0.3) is None, 'a cone behind the car has no box'
    edge = pj.cone_box(IPHONE, T_ob, (1.5, 0.55), 0.1, 0.3)
    print(f"    cone at the picture's left edge: visible fraction {edge['visible_fraction']:.2f}, "
          f"all_visible {edge['all_visible']}")
    assert edge is not None and not edge['all_visible'] and 0.0 < edge['visible_fraction'] < 1.0


def test_scan_to_points():
    ranges = [2.0, float('inf'), 0.05, 13.0, float('nan'), 1.0]
    inc = math.pi / 2
    pts, idx = pj.scan_to_points(ranges, -math.pi, inc, 0.15, 12.0, z=0.12)
    assert list(idx) == [0, 5]
    assert np.allclose(pts[0], [-2.0, 0.0, 0.12], atol=1e-12)               # beam 0 at -180 deg
    a5 = -math.pi + 5 * inc                                                  # beam 5 at +270 deg = -90 deg
    assert np.allclose(pts[1], [math.cos(a5), math.sin(a5), 0.12], atol=1e-12)


def test_camera_info_message_and_transforms():
    msg = SimpleNamespace(k=list(IPHONE.K.ravel()), d=list(IPHONE.D), width=640, height=480,
                          distortion_model='plumb_bob', header=SimpleNamespace(frame_id='camera_optical_frame'))
    cam = pj.CameraModel.from_camera_info(msg)
    assert np.allclose(cam.K, IPHONE.K) and cam.name == 'camera_optical_frame'
    try:
        pj.CameraModel.from_camera_info(SimpleNamespace(**{**msg.__dict__, 'distortion_model': 'equidistant'}))
        raise AssertionError('a fisheye model must be refused')
    except ValueError:
        pass
    q = (0.1, -0.2, 0.3, 0.9)
    tf = SimpleNamespace(transform=SimpleNamespace(translation=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                                                   rotation=SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3])))
    T = pj.T_from_tf(tf)
    assert np.allclose(pj.invert(T) @ T, np.eye(4), atol=1e-12)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12)
    T2 = pj.T_from_xyz_rpy((0, 0, 0), (0.3, -0.2, 1.1))
    R_want = rot('z', math.degrees(1.1)) @ rot('y', math.degrees(-0.2)) @ rot('x', math.degrees(0.3))
    assert np.allclose(T2[:3, :3], R_want, atol=1e-12)


if __name__ == '__main__':
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
    n_fail = n_skip = 0
    for name, t in tests:
        try:
            t()
            print(f'PASS {name}')
        except Skip as e:
            n_skip += 1
            print(f'SKIP {name}: {e}')
        except Exception:
            n_fail += 1
            print(f'FAIL {name}')
            traceback.print_exc()
    print(f'RESULT {len(tests) - n_fail - n_skip} passed, {n_fail} failed, {n_skip} skipped')
    sys.exit(1 if n_fail else 0)
