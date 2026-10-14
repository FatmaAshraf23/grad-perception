"""projection.py -- camera <-> 3-D geometry for camera-LiDAR fusion. numpy only (no ROS, no OpenCV needed).

THE CHAIN: a LiDAR point -> a pixel
  1. LiDAR scan -> 3-D points in the laser frame                      scan_to_points()
  2. laser -> base_link -> camera_optical_frame                        4x4 transforms (mount_transforms(), T_from_tf())
  3. 3-D point in camera_optical_frame -> pixel, with the lens's distortion        CameraModel.project()
THE WAY BACK: a pixel -> a direction in the world
  4. pixel -> viewing ray (lens distortion removed)                    CameraModel.pixel_to_ray()
  5. ray -> where it hits the ground (monocular distance -- weak!)     ground_point(), ground_sensitivity()
FOR FUSION
  6. where a cone of known size must appear in the picture             cone_box()

Conventions
  camera_optical_frame: z = depth (where the camera looks), x = right, y = down in the picture (OpenCV, camera_info).
  Transforms are 4x4 matrices named T_<to>_from_<from>:  p_to = T_to_from_from @ p_from.
  A TF lookup_transform(target, source) gives T_target_from_source (T_from_tf).
  Lens model: ROS 'plumb_bob' = OpenCV (k1, k2, p1, p2, k3) -- what camcal_calibrate.py writes.

SAFETY RULES BUILT IN (each one is a classic bug)
  - a point BEHIND the camera (depth <= min_depth) is never projected: the pinhole formula would mirror it INTO the
    picture;
  - a point far outside the camera's view is never projected: the distortion polynomial can fold such points back
    INTO the picture (cv2.projectPoints does too). FOV guard: only viewing angles up to 5 % beyond what the picture's
    own border covers, and never past the radius where the lens polynomial turns back;
  - a pixel whose undistortion does not converge gives no ray (ok = False).
Rejected entries come back as NaN, so an unchecked use shows up instead of silently giving a wrong pixel.
"""
import math

import numpy as np


# ---------------------------------------------------------------- rigid transforms (4 x 4)
def _T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float).ravel()
    return T


def T_from_xyz_rpy(xyz, rpy):
    """T_parent_from_child of a child frame at xyz [m], turned by roll, pitch, yaw [rad] (ROS: R = Rz Ry Rx)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                  [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                  [-sp, cp * sr, cp * cr]])
    return _T(R, xyz)


def T_from_xyz_quat(xyz, q):
    """T_parent_from_child from a translation and a quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return _T(R, xyz)


def T_from_tf(tf):
    """geometry_msgs TransformStamped (or Transform) -> 4x4. From buffer.lookup_transform(target, source, ...)
    this is T_target_from_source."""
    tr = getattr(tf, 'transform', tf)
    t, q = tr.translation, tr.rotation
    return T_from_xyz_quat((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


def invert(T):
    T = np.asarray(T, float)
    R, t = T[:3, :3], T[:3, 3]
    return _T(R.T, -R.T @ t)


def apply(T, pts):
    """Transform points [N, 3] (or one point [3]) -> [N, 3]."""
    p = np.asarray(pts, float).reshape(-1, 3)
    return p @ np.asarray(T, float)[:3, :3].T + np.asarray(T, float)[:3, 3]


def mount_transforms(mounts):
    """{frame: T_base_from_frame} for laser, camera_link, camera_optical_frame from an apex_description mount dict --
    the SAME numbers and conventions sensor_frames publishes as TF (it uses apex_description.mounts.frame_list)."""
    from apex_description.mounts import frame_list
    Ts = {'base_link': np.eye(4)}
    for parent, child, xyz, _rpy_deg, q in frame_list(mounts, '', True):
        Ts[child] = Ts[parent] @ T_from_xyz_quat(xyz, q)
    return Ts


# ---------------------------------------------------------------- the camera
class CameraModel:
    """A calibrated camera: K (3x3), D (plumb_bob k1, k2, p1, p2, k3), picture size in pixels.

    Pixel coordinates: (u, v) = (column, row); pixel (i, j) covers u in [i - 0.5, i + 0.5) -- OpenCV's convention.
    """

    def __init__(self, K, D, width, height, name=''):
        self.K = np.asarray(K, float).reshape(3, 3)
        D = np.asarray(D, float).ravel()
        if D.size > 5 and np.any(D[5:] != 0):
            raise ValueError('only the 5-number plumb_bob lens model (k1, k2, p1, p2, k3) is supported')
        self.D = np.zeros(5)
        self.D[:min(D.size, 5)] = D[:5]
        self.width, self.height, self.name = int(width), int(height), name
        self.fx, self.fy = self.K[0, 0], self.K[1, 1]
        self.cx, self.cy, self.skew = self.K[0, 2], self.K[1, 2], self.K[0, 1]
        self.r_max = math.inf          # set by _fov_guard (undistorted normalized radius)
        self.fov_guard_ok = True
        self._fov_guard()

    # ---- construction
    @classmethod
    def from_camera_info(cls, msg):
        """From a sensor_msgs/CameraInfo message (fields k, d, width, height, distortion_model)."""
        model = getattr(msg, 'distortion_model', 'plumb_bob') or 'plumb_bob'
        if model not in ('plumb_bob', 'rational_polynomial'):
            raise ValueError(f'distortion model {model!r} not supported (plumb_bob only)')
        return cls(np.asarray(msg.k, float).reshape(3, 3), list(msg.d), msg.width, msg.height,
                   getattr(getattr(msg, 'header', None), 'frame_id', ''))

    @classmethod
    def from_yaml(cls, path):
        """From a ROS camera_info YAML (camera_calibration / camcal_calibrate.py format)."""
        import yaml
        with open(path) as f:
            c = yaml.safe_load(f)
        model = c.get('distortion_model', 'plumb_bob')
        if model not in ('plumb_bob', 'rational_polynomial'):
            raise ValueError(f'{path}: distortion model {model!r} not supported (plumb_bob only)')
        return cls(c['camera_matrix']['data'], c['distortion_coefficients']['data'], c['image_width'],
                   c['image_height'], c.get('camera_name', ''))

    def scaled(self, width, height):
        """The same camera for a RESIZED picture (e.g. 2560 x 1440 -> 640 x 480, also a squash): focal lengths and
        centre scale with each axis, the lens numbers stay. NOT valid for a crop."""
        sx, sy = width / self.width, height / self.height
        K = self.K.copy()
        K[0, 0] *= sx
        K[0, 1] *= sx
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K[1, 1] *= sy
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        return CameraModel(K, self.D, width, height, self.name)

    # ---- the lens
    def _distort(self, x, y):
        k1, k2, p1, p2, k3 = self.D
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        return (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

    def _undistort(self, uv, iters=200):
        """pixels -> undistorted normalized (x, y) [N, 2] + ok (converged: re-distorted within 0.001 px).
        Fixed-point iteration as OpenCV's undistortPoints, run to convergence."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        yd = (uv[:, 1] - self.cy) / self.fy
        xd = (uv[:, 0] - self.cx - self.skew * yd) / self.fx
        k1, k2, p1, p2, k3 = self.D
        x, y = xd.copy(), yd.copy()
        with np.errstate(all='ignore'):
            for _ in range(iters):
                r2 = x * x + y * y
                radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
                xn = (xd - 2 * p1 * x * y - p2 * (r2 + 2 * x * x)) / radial
                yn = (yd - p1 * (r2 + 2 * y * y) - 2 * p2 * x * y) / radial
                d = np.abs(np.concatenate([xn - x, yn - y]))
                d = d[np.isfinite(d)]
                x, y = xn, yn
                if d.size == 0 or d.max() < 1e-15:
                    break
            xr, yr = self._distort(x, y)
            res = np.hypot((xr - xd) * self.fx, (yr - yd) * self.fy)
            ok = np.isfinite(x) & np.isfinite(y) & (res < 1e-3)
        return np.stack([x, y], 1), ok

    def _fov_guard(self):
        """r_max: the largest undistorted viewing radius we accept = 5 % beyond what the picture's border reaches,
        and never beyond the radius where the lens polynomial turns back (there, far-off points fold INTO the
        picture). fov_guard_ok = False if the picture's own border needs the impossible (a bad calibration)."""
        n = 64
        us, vs = np.linspace(0.0, self.width - 1.0, n), np.linspace(0.0, self.height - 1.0, n)
        border = np.concatenate([np.column_stack([us, np.zeros(n)]), np.column_stack([us, np.full(n, self.height - 1.0)]),
                                 np.column_stack([np.zeros(n), vs]), np.column_stack([np.full(n, self.width - 1.0), vs])])
        xy, ok = self._undistort(border)
        if not ok.all():
            self.fov_guard_ok = False
        r_border = float(np.sqrt((xy[ok] ** 2).sum(1)).max()) if ok.any() else 1.0
        r_max = 1.05 * r_border
        k1, k2, _, _, k3 = self.D
        rs = np.linspace(0.0, r_max, 4001)
        r2 = rs * rs
        f = rs * (1 + r2 * (k1 + r2 * (k2 + r2 * k3)))        # distorted radius along a ray (radial part)
        turn = np.nonzero(np.diff(f) <= 0)[0]
        if turn.size:
            r_fold = float(rs[turn[0]])
            if r_fold <= r_border:
                self.fov_guard_ok = False
            r_max = min(r_max, r_fold)
        self.r_max = r_max

    # ---- forward / backward
    def project(self, pts, min_depth=0.05):
        """3-D points in camera_optical_frame [N, 3] -> pixels [N, 2] and ok [N] = in front of the camera (depth >
        min_depth) and inside the camera's view (FOV guard). ok pixels can still be a little outside the picture:
        use in_image() when that matters. Not-ok entries are NaN."""
        p = np.asarray(pts, float).reshape(-1, 3)
        z = p[:, 2]
        ok = np.isfinite(p).all(1) & (z > min_depth)
        zs = np.where(ok, z, 1.0)
        x, y = p[:, 0] / zs, p[:, 1] / zs
        ok &= x * x + y * y <= self.r_max ** 2
        xd, yd = self._distort(x, y)
        uv = np.column_stack([self.fx * xd + self.skew * yd + self.cx, self.fy * yd + self.cy])
        uv[~ok] = np.nan
        return uv, ok

    def pixel_to_ray(self, uv):
        """pixels [N, 2] -> unit viewing rays [N, 3] in camera_optical_frame and ok [N]. Not-ok entries are NaN."""
        xy, ok = self._undistort(uv)
        ok &= (xy ** 2).sum(1) <= self.r_max ** 2 * (1 + 1e-9)
        rays = np.column_stack([xy, np.ones(len(xy))])
        with np.errstate(all='ignore'):
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        rays[~ok] = np.nan
        return rays, ok

    def in_image(self, uv, margin=0.0):
        """True where a pixel lies inside the picture (at least `margin` pixels from its edge)."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        with np.errstate(invalid='ignore'):
            return ((uv[:, 0] >= -0.5 + margin) & (uv[:, 0] < self.width - 0.5 - margin) &
                    (uv[:, 1] >= -0.5 + margin) & (uv[:, 1] < self.height - 0.5 - margin))


# ---------------------------------------------------------------- helpers for fusion
def scan_to_points(ranges, angle_min, angle_increment, range_min, range_max, z=0.0):
    """LaserScan ranges -> 3-D points [M, 3] in the laser frame (only real returns) + the beam index of each."""
    r = np.asarray(ranges, float).ravel()
    a = angle_min + angle_increment * np.arange(r.size)
    with np.errstate(invalid='ignore'):
        ok = np.isfinite(r) & (r >= range_min) & (r < range_max)
    idx = np.nonzero(ok)[0]
    return np.column_stack([r[idx] * np.cos(a[idx]), r[idx] * np.sin(a[idx]), np.full(idx.size, float(z))]), idx


def ground_point(cam, T_src_from_opt, uv, ground_z=0.0, max_range=50.0):
    """Where each pixel's viewing ray hits the ground plane z = ground_z of frame src (e.g. base_link: the floor).
    -> points [N, 3] in src, ok [N]. Pixels at / above the horizon have no ground point. This is the camera-only
    distance estimate: see ground_sensitivity() for how poor it is far away on a low car."""
    rays, ok = cam.pixel_to_ray(uv)
    T = np.asarray(T_src_from_opt, float)
    d = rays @ T[:3, :3].T                                   # ray directions in src
    c = T[:3, 3]                                             # the camera's centre in src
    with np.errstate(all='ignore'):
        s = (ground_z - c[2]) / d[:, 2]
        ok &= np.isfinite(s) & (s > 0) & (s <= max_range)
    pts = c + s[:, None] * d
    pts[~ok] = np.nan
    return pts, ok


def ground_sensitivity(cam, T_src_from_opt, uv, ground_z=0.0):
    """Metres of ground distance per pixel of picture height at these pixels (central difference, +-0.5 px):
    how far off a camera-only distance is when the box bottom is 1 pixel off. NaN where there is no ground."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    a, oka = ground_point(cam, T_src_from_opt, uv - [0.0, 0.5], ground_z, max_range=1e6)
    b, okb = ground_point(cam, T_src_from_opt, uv + [0.0, 0.5], ground_z, max_range=1e6)
    out = np.linalg.norm(a - b, axis=1)
    out[~(oka & okb)] = np.nan
    return out


def cone_box(cam, T_opt_from_src, center_xy, radius, height, ground_z=0.0, rings=5, n=32):
    """Where a cone standing on the ground at center_xy (frame src) must appear: the bounding box of its projected
    surface (rings from the base up + the tip). -> None if no part is in view, else dict(box=(u0, v0, u1, v1),
    visible_fraction, all_visible = every surface point projects inside the picture)."""
    cx, cy = (float(v) for v in center_xy)
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    pts = [np.column_stack([cx + radius * (1 - k / rings) * np.cos(a), cy + radius * (1 - k / rings) * np.sin(a),
                            np.full(n, ground_z + height * k / rings)]) for k in range(rings)]
    pts.append(np.array([[cx, cy, ground_z + height]]))
    uv, ok = cam.project(apply(T_opt_from_src, np.vstack(pts)))
    inside = ok & cam.in_image(uv)
    if not ok.any():
        return None
    u, v = uv[ok, 0], uv[ok, 1]
    return {'box': (float(u.min()), float(v.min()), float(u.max()), float(v.max())),
            'visible_fraction': float(inside.mean()), 'all_visible': bool(inside.all())}


def in_box(uv, box, margin=0.0):
    """True where pixels lie inside box (u0, v0, u1, v1), grown by margin pixels."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    u0, v0, u1, v1 = box
    with np.errstate(invalid='ignore'):
        return ((uv[:, 0] >= u0 - margin) & (uv[:, 0] <= u1 + margin) &
                (uv[:, 1] >= v0 - margin) & (uv[:, 1] <= v1 + margin))
