#!/usr/bin/env python3
"""scan_align.py -- obstacle detection step 3.2: put every scan on the WALLS of the car's map before computing
obstacle positions (scan-to-map alignment).

Why (step 3.1, step3_2_align.py): an obstacle's position = the car's pose + what the LiDAR measured. With the
TRUE pose a drive-by's average is 0.2-1.6 cm from the cone; with StateEstimate's pose the error that stays is the
POSE error (after the EKF's jump back to AMCL ~7 cm for seconds) -- the same in every scan of the drive-by, so
no averaging removes it. But the same scan also sees the walls, and the car's map says exactly where they are.

How (per scan, ~350 wall points, a few Gauss-Newton steps):
  - signed distance field of the car's map: how far a point is from the nearest wall surface, + in free space,
    - inside a wall; bilinear between cell centres (so it has a slope)
  - minimise  sum_i w_i (d(p_i) / (SIGMA_R * SCAN_CORR))^2  +  (pose - StateEstimate)^T P^-1 (pose - StateEstimate)
      p_i = scan point i with the corrected pose; w_i = robust weight (a point 3 cm off a wall counts half, cones
      and noise far from every wall hardly count, > GATE not at all); P = StateEstimate's own std_xy / std_psi.
    Only walls that FACE the LiDAR are used (the field must rise towards the sensor): a point pushed into a wall by
    the pose error would otherwise be pulled out of the wall's far side. Without this rule the first live test
    (job 085) flipped 9 of 4016 scans to a pose 11-13 cm off (one false obstacle for 0.4 s); with it: 0, the
    largest aligned error 4.5 cm.
    The prior matters where the walls say nothing: along a straight corridor the scan cannot tell how far along
    you are, so there the pose stays StateEstimate's.
  - uncertainty of the result = the inverse of the information matrix above -> std_xy (major axis) / std_psi.
    SCAN_CORR: the ~350 points are not independent (the same wall seen by neighbouring beams, the map's 5 cm cells),
    so their information is divided by SCAN_CORR^2 -- without it the alignment claimed 3-4x too little error.
Refused (StateEstimate's pose is used, the reason is counted): fewer than MIN_POINTS wall points, or a correction
larger than MAX_SHIFT / MAX_TURN (the method is local: a pose that far off is a job for the guard and StateEstimate).
numpy only, no ROS.
"""
import math

import numpy as np

try:                                    # inside the ROS package lidar_perception
    from lidar_perception.wall_distance import WallDistance
except ImportError:                     # offline, next to wall_distance.py
    from wall_distance import WallDistance

SIGMA_R = 0.02        # m: how well one wall point fits the map (scan noise + the map's cells)
SOLVE_CORR = 2.0      # weighing the scan against StateEstimate: its information divided by SOLVE_CORR^2
SCAN_CORR = 4.0       # the uncertainty it REPORTS: its information divided by SCAN_CORR^2 (points are correlated);
                      #   both calibrated in step 3.2 on the job 078 + 080 recordings, checked on job 083's
                      #   (scan_align_check.py)
ROBUST_C = 0.03       # m: Cauchy weight 1 / (1 + (d / ROBUST_C)^2)
GATE = 0.15           # m: points farther than this from every wall are not used at all
MAX_RANGE = 8.0       # m: as the detector -- far points carry the heading error the most
ITERS = 10            # Gauss-Newton steps at most (it usually stops after 3-4)
MIN_POINTS = 60       # fewer wall points -> no alignment
MAX_SHIFT = 0.15      # m: larger corrections are refused
MAX_TURN = math.radians(3.0)


class ScanAligner:
    def __init__(self, wall_distance, grid):
        """wall_distance: the detector's WallDistance of the car's map; grid: the same map (OccupancyGrid values,
        row 0 = bottom). Builds the inside-the-walls part of the signed distance field once (~50 ms)."""
        wd = wall_distance
        inside = WallDistance(np.where(np.asarray(grid) >= 65, 0, 100), wd.res, wd.ox, wd.oy, max_dist=0.5)
        self.sd = wd.dist - inside.dist
        self.res, self.ox, self.oy = wd.res, wd.ox, wd.oy
        self.h, self.w = self.sd.shape
        self.stats = dict(aligned=0, few_points=0, too_far=0, no_std=0)

    def field(self, x, y):
        """Signed distance to the nearest wall surface at points (x, y), its gradient, and inside-the-map flags."""
        u = (x - self.ox) / self.res - 0.5                     # cell centres sit at integer (u, v)
        v = (y - self.oy) / self.res - 0.5
        j0, i0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        ok = (i0 >= 0) & (i0 < self.h - 1) & (j0 >= 0) & (j0 < self.w - 1)
        i0, j0 = np.clip(i0, 0, self.h - 2), np.clip(j0, 0, self.w - 2)
        fx, fy = u - j0, v - i0
        a, b = self.sd[i0, j0], self.sd[i0, j0 + 1]
        c, d = self.sd[i0 + 1, j0], self.sd[i0 + 1, j0 + 1]
        val = (1 - fx) * (1 - fy) * a + fx * (1 - fy) * b + (1 - fx) * fy * c + fx * fy * d
        gx = ((1 - fy) * (b - a) + fy * (d - c)) / self.res
        gy = ((1 - fx) * (c - a) + fx * (d - b)) / self.res
        return val, gx, gy, ok

    def align(self, ranges, angle_min, angle_inc, range_min, range_max, pose, pose_std, mount):
        """pose = StateEstimate's (x, y, yaw) at the scan's time, pose_std = its (std_xy, std_psi), mount = LiDAR
        on the car. Returns dict(ok, pose, std (std_xy, std_psi), n wall points, shift [m], turn [rad], reason)."""
        if pose_std is None:
            self.stats['no_std'] += 1
            return dict(ok=False, pose=pose, std=pose_std, n=0, shift=0.0, turn=0.0, reason='no std')
        r = np.asarray(ranges, dtype=float)
        k = np.arange(len(r))
        good = np.isfinite(r) & (r >= range_min) & (r < range_max - 1e-3) & (r <= MAX_RANGE)
        a = mount[2] + angle_min + k[good] * angle_inc         # beam directions in the car frame
        vx = mount[0] + r[good] * np.cos(a)                    # scan points in the car frame
        vy = mount[1] + r[good] * np.sin(a)
        x0, y0, p0 = pose
        x, y, p = pose
        prior = np.diag([1.0 / pose_std[0] ** 2, 1.0 / pose_std[0] ** 2, 1.0 / pose_std[1] ** 2])
        s2 = (SIGMA_R * SOLVE_CORR) ** 2
        n, H, JWJ = 0, prior, None
        for _ in range(ITERS):
            cs, sn = math.cos(p), math.sin(p)
            px, py = x + cs * vx - sn * vy, y + sn * vx + cs * vy
            e, gx, gy, inside = self.field(px, py)
            # use a point only if the wall it is matched to FACES the LiDAR: the field must rise towards the sensor.
            # A point pushed into a wall by the pose error is otherwise pulled out of the wall's FAR side
            # (job 085: the alignment flipped between the right pose and one 11-13 cm off)
            lx, ly = x + cs * mount[0] - sn * mount[1], y + sn * mount[0] + cs * mount[1]
            facing = gx * (lx - px) + gy * (ly - py) > 0.0
            w = np.where(inside & facing & (np.abs(e) < GATE), 1.0 / (1.0 + (e / ROBUST_C) ** 2), 0.0)
            n = int(np.count_nonzero(w))
            if n < MIN_POINTS:
                break
            J = np.stack([gx, gy, gx * (-sn * vx - cs * vy) + gy * (cs * vx - sn * vy)], axis=1)
            JWJ = (J * w[:, None]).T @ J
            H = JWJ / s2 + prior
            dp = np.array([x - x0, y - y0, (p - p0 + math.pi) % (2 * math.pi) - math.pi])
            g = J.T @ (w * e) / s2 + prior @ dp
            step = -np.linalg.solve(H, g)
            x, y, p = x + step[0], y + step[1], p + step[2]
            if math.hypot(step[0], step[1]) < 1e-4 and abs(step[2]) < 1e-5:
                break
        shift = math.hypot(x - x0, y - y0)
        turn = abs((p - p0 + math.pi) % (2 * math.pi) - math.pi)
        if n < MIN_POINTS:
            self.stats['few_points'] += 1
            return dict(ok=False, pose=pose, std=pose_std, n=n, shift=0.0, turn=0.0, reason='few wall points')
        if shift > MAX_SHIFT or turn > MAX_TURN:
            self.stats['too_far'] += 1
            return dict(ok=False, pose=pose, std=pose_std, n=n, shift=shift, turn=turn, reason='correction too large')
        C = np.linalg.inv(JWJ / (SIGMA_R * SCAN_CORR) ** 2 + prior)   # the honest uncertainty (see SCAN_CORR)
        std_xy = math.sqrt(max(np.linalg.eigvalsh(C[:2, :2])))
        self.stats['aligned'] += 1
        return dict(ok=True, pose=(x, y, p), std=(std_xy, math.sqrt(C[2, 2])), n=n, shift=shift, turn=turn,
                    reason='', cov=C)
