"""fusion_core.py -- camera confirmation of the LiDAR obstacles (camera-LiDAR fusion v1, 2026-11-07). numpy only, no ROS.

DECISIONS (Fatma, 2026-11-07)
  - ONE list for planning: /obstacles/fused = the LiDAR obstacle list + camera fields per obstacle.
  - The camera NEVER removes an obstacle: when it clearly looks at one and sees no cone, the obstacle is only MARKED
    (camera_state NO_CONE) -- a missed real obstacle is worse than an extra one.
  - v1 confirms LiDAR obstacles only; camera-only early warnings (cones beyond the LiDAR's range) come later.
  - Design from 2026-10-08: the LiDAR gives WHERE (camera-only distance is poor on a low car: 1 px = 16 cm at 5 m),
    the camera gives WHAT.

PER PICTURE (CameraDetectionArray, stamped with the time the PICTURE was taken)
  1. The car's pose at the picture's time t = stamp + camera_delay_s (PoseHistory: StateEstimate, interpolated; the
     picture waits until a state at or after t exists). StateEstimate not OK at t -> the picture is not used.
  2. Every LiDAR obstacle -> where a cone standing there must appear: projection.cone_box() with the camera mount
     (base_link -> camera_optical_frame) and the lens model. Its expected HORIZONTAL pixel uncertainty:
         sigma_angle^2 = std_psi^2 + (std_xy_car^2 + std_xy_obstacle^2) / d^2 + sigma_mount^2 + (r * sigma_time)^2
         sigma_u       = sqrt((fx * sigma_angle)^2 + sigma_box_px^2)
     (heading error turns the whole picture; a position error moves the obstacle sideways by up to std_xy -> an angle
      of std_xy / d; the mount's yaw is only known to sigma_mount; a stamp off by sigma_time while turning at r rad/s
      turns the view by r * sigma_time; YOLO's box centre itself jitters by sigma_box_px.)
  3. Matching: a box can belong to an obstacle only if
       - horizontal: |u_box - u_pred| <= gate_k * sigma_u                          (centres)
       - size:       box height / predicted height in [1 / size_ratio, size_ratio] (a near obstacle and a far cone in the
                     same direction differ in size -- this is the camera's own distance check)
       - vertical:   the box bottom within pred_height * v_tol_frac + fy * sigma_pitch of the predicted bottom
     then the best pairs win (smallest normalised distance first; one box per obstacle and one obstacle per box).
  4. Evidence per obstacle id (ids are never reused by obstacle_node):
       LOOK = the camera SHOULD have seen it: the LiDAR saw it within look_max_age_s of the picture (so there is a
              line of sight: a remembered obstacle behind a wall is not "looked at"), predicted box at least
              min_visible inside the picture, at least min_box_px tall, within max_range_m.
       HIT  = a box matched it (counted even when it was not a full LOOK, e.g. half outside the picture).
       miss streak = LOOKs in a row without a HIT.
     camera_state:  NO_CONE    if miss_streak >= miss_limit (the camera looked miss_limit times in a row and saw nothing)
                    CONFIRMED  else if hits >= confirm_hits
                    UNKNOWN    otherwise (not looked at enough yet)
     Confidence = running mean of the matched boxes' confidences (EMA, conf_tau pictures).

REAL CAR
  camera_delay_s = the stamp offset measured with latency_probe on the car's camera (0 in the simulator: fake_camera
  stamps exactly); sigma_mount from calib_solve's spread; the camera mount from apex_description sensor_mounts_car.yaml
  (filled from calib_solve's result).
"""
import math
from bisect import bisect_left

import numpy as np

from sensor_fusion import projection as pj

UNKNOWN, CONFIRMED, NO_CONE = 0, 1, 2
STATE_NAMES = {UNKNOWN: 'UNKNOWN', CONFIRMED: 'CONFIRMED', NO_CONE: 'NO_CONE'}


class PoseHistory:
    """StateEstimate samples (t [s], x, y, psi, std_xy, std_psi, r, mode) -> the pose at any time inside the history."""

    def __init__(self, keep_s=2.0, max_extrapolate_s=0.0):
        self.keep_s = float(keep_s)
        self.max_extrapolate_s = float(max_extrapolate_s)
        self.t, self.s = [], []

    def add(self, t, x, y, psi, std_xy, std_psi, r=0.0, mode=1):
        if self.t and t <= self.t[-1]:
            return                                    # out of order / duplicate: ignored
        self.t.append(float(t))
        self.s.append((float(x), float(y), float(psi), float(std_xy), float(std_psi), float(r), int(mode)))
        k = bisect_left(self.t, self.t[-1] - self.keep_s)
        if k > 0:
            del self.t[:k], self.s[:k]

    def newest(self):
        return self.t[-1] if self.t else None

    def at(self, t):
        """-> dict(x, y, psi, std_xy, std_psi, r, ok) or None if t is outside the history (older than kept, or newer
        than the newest state by more than max_extrapolate_s). ok = the state on BOTH sides is mode OK."""
        if not self.t or t < self.t[0] or t > self.t[-1] + self.max_extrapolate_s:
            return None
        k = bisect_left(self.t, t)
        if k >= len(self.t):
            a = b = self.s[-1]
            f = 0.0
        elif self.t[k] == t or k == 0:
            a = b = self.s[k]
            f = 0.0
        else:
            a, b = self.s[k - 1], self.s[k]
            f = (t - self.t[k - 1]) / (self.t[k] - self.t[k - 1])
        dpsi = math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2]))
        return {'x': a[0] + f * (b[0] - a[0]), 'y': a[1] + f * (b[1] - a[1]), 'psi': a[2] + f * dpsi,
                'std_xy': max(a[3], b[3]), 'std_psi': max(a[4], b[4]), 'r': a[5] + f * (b[5] - a[5]),
                'ok': a[6] == 1 and b[6] == 1}


class Evidence:
    __slots__ = ('looks', 'hits', 'miss_streak', 'last_hit_us', 'conf', 'last_look_us')

    def __init__(self):
        self.looks = self.hits = self.miss_streak = 0
        self.last_hit_us = self.last_look_us = 0
        self.conf = 0.0


class CameraFusion:
    """Camera confirmation of LiDAR obstacles. Feed pictures with picture(); read the result with state_of()."""

    DEFAULTS = dict(camera_delay_s=0.0, cone_height=0.25, min_radius=0.05, max_range_m=8.0, min_visible=0.6,
                    min_box_px=12.0, look_max_age_s=0.3, gate_k=3.0, sigma_box_px=3.0, sigma_mount_deg=0.5,
                    sigma_time_s=0.02, sigma_pitch_deg=1.5, size_ratio=1.6, v_tol_frac=0.35, min_conf=0.25,
                    confirm_hits=2, miss_limit=10, conf_tau=5.0)

    def __init__(self, cam, T_base_opt, **cfg):
        unknown = set(cfg) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f'unknown fusion parameters: {sorted(unknown)}')
        self.c = dict(self.DEFAULTS, **cfg)
        if self.c['gate_k'] <= 0 or self.c['size_ratio'] <= 1.0 or self.c['confirm_hits'] < 1 or self.c['miss_limit'] < 1:
            raise ValueError('bad fusion parameters (gate_k > 0, size_ratio > 1, confirm_hits >= 1, miss_limit >= 1)')
        self.cam0 = cam
        self.cams = {(cam.width, cam.height): cam}
        self.T_base_opt = np.asarray(T_base_opt, float)
        self.ev = {}
        self.stats = dict(pictures=0, used=0, skipped_not_ok=0, boxes=0, matched=0, unmatched_boxes=0, looks=0,
                          hits=0, pred=0)
        self.last_picture_t = None
        self.last_matches = []            # [(obstacle id, box index, normalised distance)] of the newest picture

    def camera_for(self, w, h):
        k = (int(w), int(h))
        if k not in self.cams:
            self.cams[k] = self.cam0.scaled(*k)
        return self.cams[k]

    # ---- one picture -----------------------------------------------------------------------------------------
    def picture(self, t_stamp, width, height, boxes, obstacles, poses, picture_us=None):
        """t_stamp [s]: the picture's stamp; boxes: [(class_name, conf, u0, v0, u1, v1)]; obstacles: [dict(id, x, y,
        radius, std_xy, last_seen_s)] (the newest LiDAR list); poses: PoseHistory.
        -> 'wait' (no state at or after the picture's time yet), 'old' (older than the pose history), 'not_ok'
        (StateEstimate not OK at that time: nothing counted) or 'used'."""
        c = self.c
        t = t_stamp + c['camera_delay_s']
        newest = poses.newest()
        if newest is None or t > newest + poses.max_extrapolate_s:
            return 'wait'
        pose = poses.at(t)
        if pose is None:
            return 'old'
        self.stats['pictures'] += 1
        self.last_picture_t = t_stamp
        if not pose['ok']:
            self.stats['skipped_not_ok'] += 1
            return 'not_ok'
        self.stats['used'] += 1
        us = int(round(t * 1e6)) if picture_us is None else int(picture_us)
        cam = self.camera_for(width, height)
        T_map_base = pj.T_from_xyz_rpy((pose['x'], pose['y'], 0.0), (0.0, 0.0, pose['psi']))
        T_map_opt = T_map_base @ self.T_base_opt
        T_opt_map = pj.invert(T_map_opt)
        cam_xy = T_map_opt[:2, 3]
        boxes = [b for b in boxes if b[1] >= c['min_conf'] and b[0] in ('cone', '')]
        self.stats['boxes'] += len(boxes)

        preds = []                                      # (obstacle, predicted box, sigma_u, lookable)
        for ob in obstacles:
            d = math.hypot(ob['x'] - cam_xy[0], ob['y'] - cam_xy[1])
            if d > c['max_range_m'] + 1.0:
                continue
            pb = pj.cone_box(cam, T_opt_map, (ob['x'], ob['y']), max(c['min_radius'], min(ob['radius'], 0.3)),
                             c['cone_height'])
            if pb is None or pb['visible_fraction'] <= 0.0:
                continue
            u0, v0, u1, v1 = pb['box']
            h = v1 - v0
            s_ang2 = (pose['std_psi'] ** 2 + (pose['std_xy'] ** 2 + ob['std_xy'] ** 2) / max(d, 0.3) ** 2
                      + math.radians(c['sigma_mount_deg']) ** 2 + (pose['r'] * c['sigma_time_s']) ** 2)
            sig_u = math.sqrt((cam.fx * math.sqrt(s_ang2)) ** 2 + c['sigma_box_px'] ** 2)
            lookable = (t - ob['last_seen_s'] <= c['look_max_age_s'] and pb['visible_fraction'] >= c['min_visible']
                        and h >= c['min_box_px'] and d <= c['max_range_m'])
            preds.append((ob, (u0, v0, u1, v1), sig_u, lookable))
        self.stats['pred'] += len(preds)

        pairs = []                                      # (normalised distance, pred index, box index)
        v_pitch = cam.fy * math.radians(c['sigma_pitch_deg'])
        for i, (ob, pbox, sig_u, _) in enumerate(preds):
            pu = 0.5 * (pbox[0] + pbox[2])
            ph = max(1.0, pbox[3] - pbox[1])
            for j, b in enumerate(boxes):
                bu = 0.5 * (b[2] + b[4])
                bh = max(1.0, b[5] - b[3])
                du = abs(bu - pu) / sig_u
                if du > c['gate_k']:
                    continue
                ratio = bh / ph
                if not (1.0 / c['size_ratio'] <= ratio <= c['size_ratio']):
                    continue
                dv = abs(b[5] - pbox[3])
                if dv > ph * c['v_tol_frac'] + 2.0 * v_pitch:
                    continue
                pairs.append((du + abs(math.log(ratio)), i, j))
        pairs.sort()
        used_p, used_b, self.last_matches = set(), set(), []
        for score, i, j in pairs:
            if i in used_p or j in used_b:
                continue
            used_p.add(i)
            used_b.add(j)
            self.last_matches.append((preds[i][0]['id'], j, score))
        self.stats['matched'] += len(used_b)
        self.stats['unmatched_boxes'] += len(boxes) - len(used_b)

        for i, (ob, _, _, lookable) in enumerate(preds):
            e = self.ev.setdefault(ob['id'], Evidence())
            if i in used_p:
                j = next(jj for (oid, jj, _) in self.last_matches if oid == ob['id'])
                e.hits += 1
                e.miss_streak = 0
                e.last_hit_us = us
                a = 1.0 / max(1.0, min(e.hits, c['conf_tau']))
                e.conf += a * (boxes[j][1] - e.conf)
                self.stats['hits'] += 1
            if lookable:
                e.looks += 1
                e.last_look_us = us
                self.stats['looks'] += 1
                if i not in used_p:
                    e.miss_streak += 1
        return 'used'

    # ---- results ---------------------------------------------------------------------------------------------
    def state_of(self, oid):
        """-> dict(state, class_name, confidence, looks, hits, last_hit_us) for one obstacle id."""
        e = self.ev.get(oid)
        if e is None:
            return {'state': UNKNOWN, 'class_name': '', 'confidence': 0.0, 'looks': 0, 'hits': 0, 'last_hit_us': 0}
        if e.miss_streak >= self.c['miss_limit']:
            st = NO_CONE
        elif e.hits >= self.c['confirm_hits']:
            st = CONFIRMED
        else:
            st = UNKNOWN
        return {'state': st, 'class_name': 'cone' if st == CONFIRMED else '', 'confidence': e.conf if e.hits else 0.0,
                'looks': e.looks, 'hits': e.hits, 'last_hit_us': e.last_hit_us}

    def forget_except(self, ids):
        """drop the evidence of obstacles that left the LiDAR list (ids are never reused)"""
        keep = set(ids)
        for k in [k for k in self.ev if k not in keep]:
            del self.ev[k]

    def report(self):
        s = self.stats
        r = (f"pictures {s['pictures']} (used {s['used']}, StateEstimate not OK {s['skipped_not_ok']}), boxes {s['boxes']} "
             f"(matched {s['matched']}, unmatched {s['unmatched_boxes']}), obstacle predictions {s['pred']}, looks "
             f"{s['looks']}, hits {s['hits']}")
        for k in s:
            s[k] = 0
        return r
