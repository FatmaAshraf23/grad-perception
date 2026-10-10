#!/usr/bin/env python3
"""obstacle_list.py -- everything the obstacle node does, without ROS (so it can be replayed and tested offline).

    builder = ObstacleList(wall_distance, mount)
    builder.add_pose(t, x, y, yaw, v, r, ok, reset_flag, std_xy, std_psi)   # every StateEstimate message (100 Hz)
    result = builder.process_scan(t, ranges, angle_min, angle_inc, range_min, range_max)   # every scan

process_scan():
  1. POSE at the scan's time: interpolated between the two StateEstimate poses around the scan stamp (or
     extrapolated with speed and yaw rate, at most MAX_EXTRAPOLATION past the newest one). None = no pose.
  2. SAFETY RULES (step 2.6, replays of a 3 s pose fault):
     - JUMP: the pose moved more than speed and yaw rate explain (> JUMP_POS / JUMP_YAW between two
       StateEstimate messages) or FLAG_FILTER_RESET -> the list is EMPTIED (STATUS_RESET, epoch + 1): every
       remembered position was computed with the old pose. (Without: wrong obstacles stayed for 35 s.)
     - StateEstimate not OK (INIT / DEGRADED / LOST) -> the list is NOT updated (STATUS_FROZEN_LOCALIZATION).
     - more than MAX_NEW_SHARE of the scan's points unexplained by the map -> the pose is probably wrong,
       the WALLS look new (normal driving <= 7.4 %, pose 0.3 m + 5 deg off: median 22 %) -> NOT updated
       (STATUS_FROZEN_SCAN). Works even when StateEstimate has not noticed anything.
  2b. ALIGN (step 3.2, when an aligner is given and StateEstimate is OK): the pose is corrected so that the scan's
     wall points lie on the map's walls (scan_align.py) -- obstacle positions then carry the alignment's smaller
     pose error, and its own (honest) std_xy / std_psi replaces StateEstimate's. Refused alignments (too few wall
     points, correction too large) keep StateEstimate's pose. The safety rules above stay as they are.
  3. otherwise DETECT (map_difference) and TRACK (obstacle_tracker: match, average, confirm, remember, forget),
     telling the tracker how sure the pose is (std_xy, std_psi; step 3.1): detections made with an unsure pose
     count less and make the reported sigma larger.
Returns dict(status, epoch, obstacles = confirmed tracks, seen_now = {id: bool}, pose, pose_std, aligned,
correction (shift m, turn rad), new_share, n_points, n_objects) -- or None when there is no pose for the scan's time.
numpy only.
"""
import math
import time
from collections import deque

try:                                    # inside the ROS package lidar_perception
    from lidar_perception.map_difference import lidar_position, new_points, objects, scan_to_map
    from lidar_perception.obstacle_tracker import ObstacleTracker
except ImportError:                     # offline, next to map_difference.py / obstacle_tracker.py
    from map_difference import lidar_position, new_points, objects, scan_to_map
    from obstacle_tracker import ObstacleTracker

STATUS_OK, STATUS_FROZEN_LOCALIZATION, STATUS_FROZEN_SCAN, STATUS_RESET = 0, 1, 2, 3
STATUS_NAMES = {0: 'OK', 1: 'FROZEN_LOCALIZATION', 2: 'FROZEN_SCAN', 3: 'RESET'}
MAX_NEW_SHARE = 0.15                 # share of a scan's points that may be "new" before the scan is distrusted
JUMP_POS = 0.20                      # m beyond what the speed explains, between two poses -> jump
JUMP_YAW = math.radians(5)           # rad beyond what the yaw rate explains -> jump
MAX_EXTRAPOLATION = 0.05             # s: a scan newer than the newest pose by more than this gets no pose
POSE_BUFFER_S = 2.0                  # s of poses kept for interpolation


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def pose_jumped(prev, now, v, w):
    """prev / now = (t, x, y, yaw); v, w = speed and yaw rate at prev. True if the pose moved more than the car can."""
    dt = now[0] - prev[0]
    x_exp = prev[1] + v * dt * math.cos(prev[3])
    y_exp = prev[2] + v * dt * math.sin(prev[3])
    return (math.hypot(now[1] - x_exp, now[2] - y_exp) > JUMP_POS
            or abs(_wrap(now[3] - prev[3] - w * dt)) > JUMP_YAW)


class ObstacleList:
    def __init__(self, wall_distance, mount, aligner=None):
        self.wd = wall_distance
        self.mount = mount                   # (x, y, yaw) of the LiDAR relative to the pose's point
        self.aligner = aligner               # scan_align.ScanAligner or None (step 3.1 behaviour)
        self.tracker = ObstacleTracker()
        self.poses = deque()                 # (t, x, y, yaw unwrapped, v, r, ok, (std_xy, std_psi) or None)
        self.jump_pending = False
        self.epoch = 0
        self.jumps = 0

    def add_pose(self, t, x, y, yaw, v, r, ok, reset_flag=False, std_xy=None, std_psi=None):
        if self.poses:
            pt, px, py, pyaw, pv, pr = self.poses[-1][:6]
            if t <= pt:
                return                                       # old or repeated message
            yaw = pyaw + _wrap(yaw - pyaw)                   # unwrapped, so interpolation never jumps by 2 pi
            if reset_flag or pose_jumped((pt, px, py, pyaw), (t, x, y, yaw), pv, pr):
                self.jump_pending = True
                self.jumps += 1
        std = None if std_xy is None or std_psi is None else (float(std_xy), float(std_psi))
        self.poses.append((t, x, y, yaw, v, r, ok, std))
        while self.poses[0][0] < t - POSE_BUFFER_S:
            self.poses.popleft()

    def newest_pose_t(self):
        return self.poses[-1][0] if self.poses else None

    def pose_at(self, t):
        """(x, y, yaw, ok, (std_xy, std_psi) or None) at time t, or None."""
        if not self.poses or t < self.poses[0][0]:
            return None
        last = self.poses[-1]
        if t >= last[0]:
            dt = t - last[0]
            if dt > MAX_EXTRAPOLATION:
                return None
            _, x, y, yaw, v, r, ok, std = last
            return x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw), yaw + r * dt, ok, std
        lo, hi = 0, len(self.poses) - 1                       # binary search for the pair around t
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.poses[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        a, b = self.poses[lo], self.poses[hi]
        f = (t - a[0]) / (b[0] - a[0])
        return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * (b[3] - a[3]), b[6], b[7]

    def process_scan(self, t, ranges, angle_min, angle_inc, range_min, range_max):
        pose = self.pose_at(t)
        if pose is None:
            return None
        x, y, yaw, ok, pose_std = pose
        aligned, correction, align_us = False, (0.0, 0.0), 0.0
        if ok and self.aligner is not None and pose_std is not None:
            t0 = time.perf_counter()
            al = self.aligner.align(ranges, angle_min, angle_inc, range_min, range_max, (x, y, yaw), pose_std, self.mount)
            align_us = (time.perf_counter() - t0) * 1e6
            if al['ok']:
                (x, y, yaw), pose_std = al['pose'], al['std']
                aligned, correction = True, (al['shift'], al['turn'])
        reset = self.jump_pending
        if reset:
            self.tracker.reset()
            self.epoch += 1
            self.jump_pending = False
        px, py, _, _ = scan_to_map(ranges, angle_min, angle_inc, range_min, range_max, (x, y, yaw), self.mount)
        keep = new_points(self.wd, px, py)
        share = float(keep.mean()) if len(px) else 0.0
        lx, ly, lyaw = lidar_position((x, y, yaw), self.mount)
        n_obj, updated = 0, False
        if not ok:
            status = STATUS_FROZEN_LOCALIZATION
        elif share > MAX_NEW_SHARE:
            status = STATUS_FROZEN_SCAN
        else:
            objs = objects(px[keep], py[keep], (lx, ly))
            n_obj = len(objs)
            scan = dict(ranges=ranges, angle_min=angle_min, angle_inc=angle_inc, range_min=range_min,
                        range_max=range_max, lidar=(lx, ly, lyaw))
            self.tracker.update(t, objs, scan, pose_std)
            status, updated = STATUS_OK, True
        confirmed = [tr for tr in self.tracker.tracks if tr.confirmed]
        return dict(status=STATUS_RESET if reset else status, epoch=self.epoch, obstacles=confirmed,
                    seen_now={tr.id: bool(updated and tr.seen_now) for tr in confirmed},
                    pose=(x, y, yaw), pose_std=pose_std, aligned=aligned, correction=correction, align_us=align_us,
                    new_share=share, n_points=len(px), n_objects=n_obj)
