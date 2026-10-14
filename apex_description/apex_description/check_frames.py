"""check_frames -- do the sensor frames in TF match the mount YAML, and do the camera axes mean what we think?

  ros2 run apex_description check_frames --mounts car                       (the real car)
  ros2 run apex_description check_frames --mounts sim --frame-prefix est/ --truth-prefix ego_racecar/ --tree
                                                                             (simulator: estimated car vs true car)
Checks (PASS / FAIL; exit code 1 if any fails):
  1. the camera's position in base_link = the YAML's x, y, z
  2. the optical turn: camera_link forward -> optical +z, left -> optical -x, up -> optical -y
  3. the whole rotation base_link -> optical = the YAML's roll / pitch / yaw, recomputed independently with
     elementary rotation matrices (not with the node's quaternion code)
  4. the LiDAR frame (if published): x, y, z, yaw = the YAML's
Also says in plain words where "far straight ahead of the car" appears in the picture and how a vertical pole leans.
"""
import argparse
import math
import sys
import time

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from tf2_ros import Buffer, TransformListener

from apex_description.mounts import load_mounts

# camera_link -> optical: x_opt = -y_link (right), y_opt = -z_link (down), z_opt = x_link (forward = depth)
P_OPT = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == 'x':
        return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def as_matrix(tf):
    """TransformStamped (from lookup_transform(target, source)) -> R, t with p_target = R @ p_source + t."""
    q = tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    t = tf.transform.translation
    return R, np.array([t.x, t.y, t.z])


class Checker(Node):

    def __init__(self):
        super().__init__('check_frames')
        self.buf = Buffer()
        self.listener = TransformListener(self.buf, self)

    def spin_for(self, seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for(self, target, source, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.buf.can_transform(target, source, Time()):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def lookup(self, target, source):
        return as_matrix(self.buf.lookup_transform(target, source, Time()))

    def print_tree(self):
        frames = yaml.safe_load(self.buf.all_frames_as_yaml()) or {}
        children = {}
        for f, info in frames.items():
            children.setdefault(info['parent'], []).append(f)
        roots = sorted(p for p in children if p not in frames)
        print('TF tree (static = fixed, published once; dynamic = moving, with its rate):')

        def show(f, depth):
            info = frames.get(f)
            if info is None:
                kind = ''
            elif float(info.get('most_recent_transform', 0.0)) == 0.0:
                kind = '  (static)'
            else:
                kind = f"  (dynamic, ~{float(info.get('rate', 0.0)):.0f} Hz)"
            print('  ' + '    ' * depth + f + kind)
            for ch in sorted(children.get(f, [])):
                show(ch, depth + 1)
        for r in roots:
            show(r, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mounts', required=True, help="'car', 'sim' or a path to a sensor-mount YAML")
    ap.add_argument('--frame-prefix', default='', help="frame prefix: '' on the car, 'est/' or 'ego_racecar/' in the sim")
    ap.add_argument('--truth-prefix', default='', help="simulator: compare with this car's camera (ego_racecar/)")
    ap.add_argument('--tree', action='store_true', help='print the whole TF tree')
    ap.add_argument('--wait', type=float, default=10.0, help='seconds to wait for the frames')
    a = ap.parse_args(remove_ros_args(sys.argv)[1:])
    m = load_mounts(a.mounts)
    rclpy.init(args=sys.argv)
    n = Checker()
    P = a.frame_prefix
    base, link, opt, laser = P + 'base_link', P + 'camera_link', P + 'camera_optical_frame', P + 'laser'
    results = []

    def check(name, ok, detail):
        results.append(bool(ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    print(f"mounts {m['_path']} (status {m['status']}); frames {base} -> {link} -> {opt}")
    if not n.wait_for(opt, base, a.wait):
        print(f'  FAIL  no TF {base} -> {opt} within {a.wait:.0f} s (sensor_frames running with prefix {P!r}?)')
        if a.tree:
            n.spin_for(2.0)
            n.print_tree()
        sys.exit(1)
    c = m['camera']

    # 1. position
    _, t_bo = n.lookup(base, opt)                  # p_base = R p_opt + t  ->  t = the camera's origin in base_link
    want = np.array([c['x'], c['y'], c['z']], float)
    check('camera position', np.abs(t_bo - want).max() < 1e-3,
          f'TF ({t_bo[0]:.3f}, {t_bo[1]:.3f}, {t_bo[2]:.3f}) m, YAML ({want[0]:.3f}, {want[1]:.3f}, {want[2]:.3f}) m')

    # 2. the optical turn (the same on every camera)
    R_ol, _ = n.lookup(opt, link)                  # p_opt = R_ol p_link
    words = []
    for name, v in (('forward', (1, 0, 0)), ('left', (0, 1, 0)), ('up', (0, 0, 1))):
        o = R_ol @ np.array(v, float)
        words.append(f'{name} -> ({o[0]:+.0f}, {o[1]:+.0f}, {o[2]:+.0f})')
    check('optical turn', np.abs(R_ol - P_OPT).max() < 1e-6,
          '; '.join(words) + '   [must be: forward (+0, +0, +1) = depth, left (-1, +0, +0) = LEFT in the picture, '
          'up (+0, -1, +0) = UP in the picture]')

    # 3. the whole rotation, recomputed from the YAML angles with elementary matrices
    R_ob, _ = n.lookup(opt, base)                  # p_opt = R_ob p_base + t
    R_want = P_OPT @ rot('x', -c['roll']) @ rot('y', -c['pitch']) @ rot('z', -c['yaw'])
    err = np.abs(R_ob - R_want).max()
    check('mount rotation', err < 1e-6, f"roll {c['roll']} / pitch {c['pitch']} / yaw {c['yaw']} deg, "
          f'largest difference {err:.1e}')
    d = R_ob @ np.array([1.0, 0.0, 0.0])           # far straight ahead of the car, seen from the camera
    right = math.degrees(math.atan2(d[0], d[2]))
    above = math.degrees(math.atan2(-d[1], d[2]))
    u = R_ob @ np.array([0.0, 0.0, 1.0])           # the world's "up" in the picture
    lean = math.degrees(math.atan2(u[0], -u[1]))  # + = the top of a vertical pole leans RIGHT
    print(f"  far straight ahead of the car appears {abs(right):.2f} deg {'right' if right >= 0 else 'left'} of and "
          f"{abs(above):.2f} deg {'above' if above >= 0 else 'below'} the picture centre")
    print(f"  a vertical pole leans {abs(lean):.2f} deg to the {'right' if lean >= 0 else 'left'} at the top")
    print(f'SUMMARY ahead_right_deg={right:.2f} ahead_above_deg={above:.2f} pole_lean_right_deg={lean:.2f}')

    # 4. the LiDAR
    if n.wait_for(base, laser, min(a.wait, 5.0)):
        R_bl, t_bl = n.lookup(base, laser)
        L = m['laser']
        yaw = math.degrees(math.atan2(R_bl[1, 0], R_bl[0, 0]))
        e_pos = max(abs(t_bl[0] - L['x']), abs(t_bl[1] - L['y']), abs(t_bl[2] - L['z']))
        e_yaw = abs((yaw - L['yaw'] + 180.0) % 360.0 - 180.0)
        check('LiDAR position + yaw', e_pos < 1e-3 and e_yaw < 0.1,
              f"TF ({t_bl[0]:.3f}, {t_bl[1]:.3f}, {t_bl[2]:.3f}) m yaw {yaw:.2f} deg, YAML ({L['x']:.3f}, "
              f"{L['y']:.3f}, {L['z']:.3f}) m yaw {L['yaw']:.2f} deg")
    else:
        print(f'  ----  no TF {base} -> {laser} (no LiDAR frame published: fine for a test without a LiDAR)')

    # 5. simulator: the estimated camera vs the true camera
    if a.truth_prefix:
        T = a.truth_prefix + 'camera_optical_frame'
        if n.wait_for('map', opt, a.wait) and n.wait_for('map', T, a.wait):
            R1, t1 = n.lookup('map', opt)
            R2, t2 = n.lookup('map', T)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R1.T @ R2) - 1.0) / 2.0))))
            print(f'  estimated camera vs true camera (map frame): {np.linalg.norm(t1 - t2) * 100:.1f} cm, '
                  f'{ang:.2f} deg  (= the localization error, seen at the camera)')
        else:
            print(f'  ----  map -> {opt} / {T} not available (localization running?)')

    if a.tree:
        n.spin_for(2.0)
        n.print_tree()
    ok = all(results)
    print(f"RESULT {'PASS' if ok else 'FAIL'} ({sum(results)} of {len(results)} checks passed)")
    n.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
