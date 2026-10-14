"""synthetic.py -- a FAKE camera-LiDAR calibration session with KNOWN answers (for the tests and calib_demo):
checkerboard poses in front of the car, the corner pixels the camera would detect (+ pixel noise) and the scans a
2-D LiDAR would give (rays cast against the boards and a room's walls, + range noise)."""
import math

import numpy as np

from sensor_fusion import projection as pj
from sensor_fusion.lidar_camera_calib import BOARD_MARGIN

# a board facing the car: x_b = to the right (-y), y_b = down (-z), z_b = away from the car (+x)
R_FACING = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
WALLS = [((4.0, 0.0, 0.0), (1.0, 0.0, 0.0)), ((-2.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
         ((0.0, 2.5, 0.0), (0.0, 1.0, 0.0)), ((0.0, -2.5, 0.0), (0.0, -1.0, 0.0))]   # a 6 x 5 m room


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return {'x': np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), 'z': np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def board_pose(center, turn_deg, tilt_deg, objp):
    """T_base_from_board for a board whose pattern centre is at `center` (base_link), turned about the vertical by
    turn_deg and tilted about its own horizontal axis by tilt_deg (forward / back)."""
    R = _rot('z', turn_deg) @ R_FACING @ _rot('x', tilt_deg)
    return pj._T(R, np.asarray(center, float) - R @ objp.mean(0))


def default_specs(laser_z, tilted=True):
    """20 poses = the recommended session: 1.0-1.9 m ahead, left / centre / right of the picture, turned up to 35 deg
    and TILTED 15-35 deg both ways (tilted=False: all vertical -- the degenerate case). The pattern centre near the
    LiDAR's height, so the scan crosses it. (2026-10-14: 16 poses with tilts of 10-25 deg left the height / pitch /
    roll WEAK in the end-to-end test.)"""
    combos = [(-30, -25), (30, 25), (0, 30), (0, -30), (-20, 25), (20, -25), (35, -15), (-35, 15), (-15, -35), (15, 35)]
    specs = []
    for k, (turn, tilt) in enumerate(combos * 2):
        half = k // len(combos)                     # the second round: other distances / sides / heights
        dist = (1.0, 1.3, 1.6, 1.9)[(k + half) % 4]
        lat = (-0.40, 0.0, 0.40, 0.20)[(k // 2 + 2 * half) % 4] * dist / 1.5   # spread over the whole picture
        zc = laser_z + (0.0, 0.03, -0.03, 0.015)[(k + 1 + half) % 4]
        specs.append((dist, lat, zc, turn, tilt if tilted else 0.0))
    return specs


def raycast_scan(T_base_laser, T_boards, objp, margin=BOARD_MARGIN, n_beams=360, sigma=0.01, rng=None,
                 range_max=12.0):
    """One 2-D scan: the nearest hit of each beam on the boards (pattern + margin) or the room's walls, + noise."""
    rng = rng if rng is not None else np.random.default_rng(0)
    o, Rl = T_base_laser[:3, 3], T_base_laser[:3, :3]
    a = -math.pi + 2 * math.pi * np.arange(n_beams) / n_beams
    U = np.column_stack([np.cos(a), np.sin(a), np.zeros(n_beams)]) @ Rl.T
    best = np.full(n_beams, np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        for p, nrm in WALLS:
            nrm = np.asarray(nrm, float)
            s = ((np.asarray(p, float) - o) @ nrm) / (U @ nrm)
            best = np.where((s > 0) & (s < best), s, best)
        lo, hi = objp[:, :2].min(0) - margin, objp[:, :2].max(0) + margin
        for T in T_boards:
            Rb, ob = T[:3, :3], T[:3, 3]
            s = ((ob - o) @ Rb[:, 2]) / (U @ Rb[:, 2])
            hb = (o + s[:, None] * U - ob) @ Rb
            hit = (s > 0) & np.all(hb[:, :2] >= lo, 1) & np.all(hb[:, :2] <= hi, 1)
            best = np.where(hit & (s < best), s, best)
    r = best + rng.normal(0.0, sigma, n_beams)
    r[~np.isfinite(best) | (r > range_max)] = np.inf
    return r, -math.pi, 2 * math.pi / n_beams


def make_session(cam, T_base_opt, T_base_laser, specs, objp, sigma_px=0.3, sigma_r=0.01, n_scans=10, seed=0,
                 margin=BOARD_MARGIN):
    """-> poses [{'corners', 'scan_points', 'spec', 'T_base_board'}] (only boards the camera sees completely)."""
    rng = np.random.default_rng(seed)
    T_opt_base = pj.invert(T_base_opt)
    poses = []
    for spec in specs:
        dist, lat, zc, turn, tilt = spec
        T_bb = board_pose((T_base_laser[0, 3] + dist, lat, zc), turn, tilt, objp)
        uv, ok = cam.project(pj.apply(T_opt_base @ T_bb, objp))
        if not (ok.all() and cam.in_image(uv, margin=5.0).all()):
            continue
        corners = uv + rng.normal(0.0, sigma_px, uv.shape)
        pts = []
        for _ in range(n_scans):
            r, a0, inc = raycast_scan(T_base_laser, [T_bb], objp, margin, sigma=sigma_r, rng=rng)
            pts.append(pj.scan_to_points(r, a0, inc, 0.15, 12.0)[0])
        poses.append(dict(corners=corners, scan_points=np.vstack(pts), spec=spec, T_base_board=T_bb))
    return poses
