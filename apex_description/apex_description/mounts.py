"""mounts.py -- the sensor-mount YAML (where the LiDAR and the camera sit on the car) -> static transforms.

No ROS imports at module level: launch files, tools and tests can use it too.

Frames (ROS REP 103 / 105):
  base_link             the point StateEstimate reports (the centre of gravity), ON THE GROUND;
                        x forward, y left, z up
  laser                 the 2-D LiDAR's scan centre; x = the scan's 0-degree beam, z up
  camera_link           the camera's lens centre with car-like axes: x = where the camera looks, y left, z up
  camera_optical_frame  the SAME point with the camera convention: z = where it looks (depth), x = RIGHT in the
                        picture, y = DOWN in the picture -- what OpenCV, camera_info and image messages use
"""
import math
import os

import yaml

STATUSES = ('simulator', 'placeholder', 'measured')
KEYS = {'laser': ('x', 'y', 'z', 'yaw'), 'camera': ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}
MAX_OFFSET_M = 1.0          # a 1:10 car: no sensor is a metre from its centre (millimetres typed as metres?)
MAX_TILT_DEG = 60.0         # |roll|, |pitch|: more is a typo
# camera_link -> camera_optical_frame is always the same turn: roll -90, pitch 0, yaw -90 (degrees)
OPTICAL_RPY_DEG = (-90.0, 0.0, -90.0)


def resolve_mounts(name):
    """'car' / 'sim' -> the installed config/sensor_mounts_<car|sim>.yaml; anything else is a path."""
    if name in ('car', 'sim'):
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('apex_description'), 'config', f'sensor_mounts_{name}.yaml')
    return os.path.expanduser(name)


def load_mounts(name):
    """Read + check a sensor-mount YAML ('car', 'sim' or a path). Raises ValueError listing EVERY problem:
    a wrong mount is worse than none -- every fused detection would be placed wrong."""
    path = resolve_mounts(name)
    with open(path) as f:
        m = yaml.safe_load(f)
    if not isinstance(m, dict):
        raise ValueError(f'{path}: not a YAML mapping')
    problems = []
    if m.get('status') not in STATUSES:
        problems.append(f"status must be one of {', '.join(STATUSES)} (got {m.get('status')!r})")
    for sensor, keys in KEYS.items():
        s = m.get(sensor)
        if not isinstance(s, dict):
            problems.append(f'{sensor}: missing')
            continue
        for k in keys:
            v = s.get(k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                problems.append(f'{sensor}.{k}: must be a number (got {v!r})')
            elif k in ('x', 'y', 'z') and abs(v) > MAX_OFFSET_M:
                problems.append(f'{sensor}.{k} = {v} m: more than {MAX_OFFSET_M} m from base_link (metres, not mm)')
            elif k in ('roll', 'pitch') and abs(v) > MAX_TILT_DEG:
                problems.append(f'{sensor}.{k} = {v} deg: more than {MAX_TILT_DEG} deg (angles are in DEGREES)')
            elif k == 'yaw' and abs(v) > 180.0:
                problems.append(f'{sensor}.yaw = {v} deg: outside -180..180')
        extra = sorted(set(s) - set(keys))
        if extra:
            problems.append(f'{sensor}: unknown keys {extra} (typo?)')
    if problems:
        raise ValueError(f'{path}: ' + '; '.join(problems))
    m['_path'] = path
    return m


def quat_from_rpy(roll, pitch, yaw):
    """(x, y, z, w) of R = Rz(yaw) Ry(pitch) Rx(roll) -- the ROS / URDF / tf2 roll-pitch-yaw convention. Radians."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def frame_list(m, prefix='', laser=True):
    """The static transforms: [(parent, child, (x, y, z) [m], (roll, pitch, yaw) [deg], (qx, qy, qz, qw)), ...]."""
    out = []

    def add(parent, child, xyz, rpy_deg):
        xyz = tuple(float(v) for v in xyz)
        rpy_deg = tuple(float(v) for v in rpy_deg)
        out.append((prefix + parent, prefix + child, xyz, rpy_deg, quat_from_rpy(*(math.radians(a) for a in rpy_deg))))

    if laser:
        s = m['laser']
        add('base_link', 'laser', (s['x'], s['y'], s['z']), (0.0, 0.0, s['yaw']))
    c = m['camera']
    add('base_link', 'camera_link', (c['x'], c['y'], c['z']), (c['roll'], c['pitch'], c['yaw']))
    add('camera_link', 'camera_optical_frame', (0.0, 0.0, 0.0), OPTICAL_RPY_DEG)
    return out
