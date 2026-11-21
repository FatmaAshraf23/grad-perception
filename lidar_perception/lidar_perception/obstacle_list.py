#!/usr/bin/env python3
"""obstacle_list.py -- everything the obstacle node does, without ROS (so it can be replayed and tested offline).

    builder = ObstacleList(wall_distance, mount, aligner, range_offset)
    builder.add_pose(t, x, y, yaw, v, r, ok, reset_flag, std_xy, std_psi)   # every StateEstimate message (100 Hz)
    result = builder.process_scan(t, ranges, angle_min, angle_inc, range_min, range_max)   # every scan

process_scan():
  0. RANGE OFFSET (step 3.3): every real return is shortened by range_offset -- how much too FAR this LiDAR measures
     (the simulator's: 2.2 cm, its ray marching stops up to one map cell inside a surface; a real LiDAR: measure it).
     Beams without a return (inf, NaN, < range_min, >= range_max) are left exactly as they are, so "no return" stays
     "no return" (the forgetting rule must never take it for a beam that went through). The alignment always uses
     the corrected ranges, and so do the detection and the forgetting of every ALIGNED scan. A scan that is NOT
     aligned (alignment off or refused) is detected with the ranges as measured, exactly as in step 3.2: TOL (10 cm,
     step 2.2) was chosen while the overshoot still pushed every wall point ~2 cm INTO the walls -- a hidden margin
     against a pose error towards a wall. Replay of job 086 run B without the alignment but with corrected ranges:
     StateEstimate 8 cm off (still OK) made a wall look new -> a false obstacle for 63 s. Aligned scans don't need
     that margin (pose error across the car ~0.3 cm p95). So range_offset has no effect when align_scans is off.
     range_offset 0 = step 3.2.
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
     freeze_on_refused (phantom fix part a, 2026-11-19): a refusal because the correction is TOO LARGE means this
     scan itself says StateEstimate's pose is more than scan_align.MAX_SHIFT / MAX_TURN off -> the scan does NOT
     update the list (STATUS_FROZEN_SCAN, like the new-point guard). Without it, a pose 15-20 cm off along the
     track (StateEstimate still OK) made dozens of phantom obstacles in the corners, because the refused scans were
     detected with the wrong pose (planning's phantoms near s 37.5 on closed1). Too few wall points proves nothing
     either way -> detected with StateEstimate's pose as before. Off (False) = the behaviour before this fix.
     Measured (phantom_update/phantom_fix_compare.py): 15 recordings (normal laps, EKF fault 0.86 m / 10 deg,
     4.5 m/s) unchanged -- 0 of 8,218 messages differ (the alignment never refused 'too large' there; the fault
     scans were refused for too few wall points and were already RESET / FROZEN_SCAN). Pose-error sweep on job 090
     run A, along +15 / +20 / -15 / -20 cm: 36 / 75 / 5 / 53 -> 6 / 15 / 0 / 9 phantoms. NOT stopped: phantoms
     made in scans the alignment ACCEPTED (along 30-50 cm, yaw 3 deg) -> phantom fix part b.
  2c. POSE CHECK (check_pose, phantom fix part b, 2026-11-19; needs the aligner): the alignment is local -- it only
     uses points already within GATE (15 cm) of a wall, and a 3 cm miss already halves a point's weight. With a
     larger pose error (along 30-50 cm, or 3 deg of heading seen at 3+ m) exactly the points that show the error are
     ignored, the walls parallel to the error fit anyway, and a small wrong correction is ACCEPTED -> wall points
     become phantoms. So after an accepted alignment, pose_check() slides the aligned pose along its heading
     (+-0.6 m, 4 cm steps) and turns it (+-6 deg, 0.5 deg steps) and counts the scan's points within CHECK_TOL
     (5 cm) of a wall surface that FACES the LiDAR (a point pushed into a wall must not fit the wall's far side --
     with thick map walls and unknown behind, an error of one wall thickness looked like a perfect fit). If a pose
     at least 8 cm further along / back, or turned at least 1 deg, fits clearly better (CHECK_GAIN = 1.5 % more of
     the scan's points), the pose is wrong -> the scan does NOT update the list (STATUS_FROZEN_SCAN). Only offsets
     that big count: in clean scans the best fit is always at <= 2 cm / <= 0.5 deg (noise); at >= 8 cm their gain
     is <= 0.0030, at >= 1 deg there is none. Second tier: a gain >= CHECK_GAIN_ANY (4 %) counts at ANY offset --
     after the alignment a COMBINED error (e.g. 20 cm back + 5 cm sideways + 0.3 deg in a corner) shows a big gain
     already at a small slide (4-6 cm); clean scans never gain more than 0.0194 at any offset.
     Measured (phantom_update/phantom_fix_compare.py, both switches on vs off): 15 recordings unchanged (0 scans
     flagged, 0 of 8,218 messages differ). Pose-error sweep on job 090 run A (StateEstimate kept OK), WALL phantoms
     before -> after: along +10/+15/+20/+30/+50 cm 1/36/75/55/39 -> 0, -15/-20 cm 5/53 -> 0, -30 cm 46 -> 1 (7 s,
     corner s 42); drift to 0.3/0.5 m 40/48 -> 0; heading +3/-3/+5 deg 18/57/201 -> 0. At 4.5 m/s (job 271 R3):
     along +15/+20/+30/-20 cm 11/42/35/28 -> 0, -30 cm 17 -> 2 (9 s), heading +-3 deg 26/11 -> 0, drift 19 -> 0.
     NOT caught: (1) a COMBINED error left after the alignment in a corner (pose 30 cm behind: along + sideways +
     heading) can pass in 2 scans in a row -- and such a phantom stays, because an obstacle right in front of a wall
     is never "seen through" by the forgetting rule (StateEstimate's map-mismatch alarm is meant for 30-95 cm);
     (2) an along error on a STRAIGHT puts a REAL cone at the wrong spot (the open cone 30/50 cm off in the sweep):
     walls parallel to the error cannot show it -- StateEstimate's job. While the pose is wrong the list is frozen a
     lot (along 30 cm: 483 of 498 scans), so new cones are not added. Cost ~2 ms per scan (laptop VM).
     Thresholds are simulator values: re-measure on real data (the node logs the largest gains it sees).
     With either switch on, a scan that does not update the list still AGES the tentative obstacles
     (tracker.expire): otherwise one from before a long freeze could be confirmed by a single detection after it.
  3. otherwise DETECT (map_difference) and TRACK (obstacle_tracker: match, average, confirm, remember, forget),
     telling the tracker how sure the pose is (std_xy, std_psi; step 3.1): detections made with an unsure pose
     count less and make the reported sigma larger.
Returns dict(status, epoch, obstacles = confirmed tracks, seen_now = {id: bool}, pose, pose_std, aligned,
correction (shift m, turn rad), refused (the alignment's reason when it refused the scan, else ''), check (the
pose check's numbers, see pose_check(), or None), check_us, new_share, n_points, n_objects) -- or None when there
is no pose for the scan's time.
numpy only.
"""
import math
import time
from collections import deque

import numpy as np

try:                                    # inside the ROS package lidar_perception
    from lidar_perception.map_difference import MAX_RANGE, lidar_position, new_points, objects, scan_to_map
    from lidar_perception.obstacle_tracker import ObstacleTracker
except ImportError:                     # offline, next to map_difference.py / obstacle_tracker.py
    from map_difference import MAX_RANGE, lidar_position, new_points, objects, scan_to_map
    from obstacle_tracker import ObstacleTracker

STATUS_OK, STATUS_FROZEN_LOCALIZATION, STATUS_FROZEN_SCAN, STATUS_RESET = 0, 1, 2, 3
STATUS_NAMES = {0: 'OK', 1: 'FROZEN_LOCALIZATION', 2: 'FROZEN_SCAN', 3: 'RESET'}
MAX_NEW_SHARE = 0.15                 # share of a scan's points that may be "new" before the scan is distrusted
JUMP_POS = 0.20                      # m beyond what the speed explains, between two poses -> jump
JUMP_YAW = math.radians(5)           # rad beyond what the yaw rate explains -> jump
MAX_EXTRAPOLATION = 0.05             # s: a scan newer than the newest pose by more than this gets no pose
POSE_BUFFER_S = 2.0                  # s of poses kept for interpolation
REFUSED_TOO_FAR = 'correction too large'   # scan_align's reason when the scan says the pose is too far off
CHECK_ALONG = np.linspace(-0.60, 0.60, 31)            # m: pose check, shifts along the heading (4 cm steps, 0 = middle)
CHECK_YAW = np.radians(np.linspace(-6.0, 6.0, 25))    # rad: pose check, heading offsets (0.5 deg steps, 0 = middle)
CHECK_TOL = 0.05                     # m: a point "fits" when it is this close to a wall surface
CHECK_GAIN = 0.015                   # share of the scan's points that must fit better at the offset
CHECK_MIN_ALONG = 0.07               # m: BETWEEN the 4 cm steps -> offsets >= 8 cm count (clean gain there <= 0.0030)
CHECK_MIN_YAW = math.radians(0.875)  # rad: BETWEEN the 0.5 deg steps -> offsets >= 1 deg count (clean: never there)
CHECK_GAIN_ANY = 0.04                # a gain this big counts at ANY offset (combined errors; clean max 0.0194)
CHECK_BEAM_STEP = 1                  # pose check uses every n-th real return. 2 halves the work but DOUBLES the clean
                                     # noise (any-offset gain 0.033, too close to CHECK_GAIN_ANY) -> keep 1


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def correct_ranges(ranges, range_min, range_max, offset):
    """The scan's ranges with `offset` [m] subtracted from every REAL return (finite, >= range_min, < range_max);
    beams without a return keep their value. A copy -- the caller's array is not changed."""
    r = np.array(ranges, dtype=float)
    if offset:
        real = np.isfinite(r) & (r >= range_min) & (r < range_max - 1e-3)
        r[real] -= offset
    return r


def _best_offset(fit, offsets):
    """(gain, offset): best fit minus the fit at offset 0 (the middle element), and the middle of the best plateau."""
    top = fit.max()
    on_top = np.flatnonzero(fit >= top - 1e-12)
    return float(top - fit[len(fit) // 2]), float(offsets[on_top[len(on_top) // 2]])


def pose_check(field, ranges, angle_min, angle_inc, range_min, range_max, pose, mount):
    """Phantom fix part b (see 2c): would a pose further along / back or turned fit the walls clearly better than
    `pose`? field = ScanAligner.field (signed distance to the nearest wall surface, its gradient, inside-the-map
    flags). A point fits when it is within CHECK_TOL of a wall surface that FACES the LiDAR.
    Returns dict(fit, gain_along, best_along [m], gain_yaw, best_yaw [rad], suspect)."""
    r = np.asarray(ranges, dtype=float)
    k = np.arange(len(r))
    good = np.isfinite(r) & (r >= range_min) & (r < range_max - 1e-3) & (r <= MAX_RANGE)
    use = k[good][::CHECK_BEAM_STEP]
    if not len(use):
        return dict(fit=0.0, gain_along=0.0, best_along=0.0, gain_yaw=0.0, best_yaw=0.0, suspect=False)
    a = mount[2] + angle_min + use * angle_inc
    vx, vy = mount[0] + r[use] * np.cos(a), mount[1] + r[use] * np.sin(a)        # points in the car frame
    x, y, yaw = pose
    n_a, n_y = len(CHECK_ALONG), len(CHECK_YAW)
    xs = np.concatenate([x + CHECK_ALONG * math.cos(yaw), np.full(n_y, x)])
    ys = np.concatenate([y + CHECK_ALONG * math.sin(yaw), np.full(n_y, y)])
    yaws = np.concatenate([np.full(n_a, yaw), yaw + CHECK_YAW])
    cs, sn = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
    px, py = xs[:, None] + cs * vx - sn * vy, ys[:, None] + sn * vx + cs * vy
    lx = xs + np.cos(yaws) * mount[0] - np.sin(yaws) * mount[1]          # the LiDAR, for every candidate pose
    ly = ys + np.sin(yaws) * mount[0] + np.cos(yaws) * mount[1]
    e, gx, gy, inside = (v.reshape(px.shape) for v in field(px.ravel(), py.ravel()))
    # a point fits only on a wall surface that FACES the LiDAR (the field rises towards the sensor), as in
    # scan_align: a point pushed INTO a wall by a pose error must not fit the wall's far side (thick map walls with
    # unknown behind: an error of one wall thickness would otherwise look like a perfect fit)
    facing = gx * (lx[:, None] - px) + gy * (ly[:, None] - py) > 0.0
    fit = np.mean(inside & facing & (np.abs(e) <= CHECK_TOL), axis=1)
    ga, ba = _best_offset(fit[:n_a], CHECK_ALONG)
    gy, by = _best_offset(fit[n_a:], CHECK_YAW)
    suspect = ((ga >= CHECK_GAIN and abs(ba) >= CHECK_MIN_ALONG) or (gy >= CHECK_GAIN and abs(by) >= CHECK_MIN_YAW)
               or max(ga, gy) >= CHECK_GAIN_ANY)
    return dict(fit=float(fit[n_a // 2]), gain_along=ga, best_along=ba, gain_yaw=gy, best_yaw=by,
                suspect=bool(suspect))


def pose_jumped(prev, now, v, w):
    """prev / now = (t, x, y, yaw); v, w = speed and yaw rate at prev. True if the pose moved more than the car can."""
    dt = now[0] - prev[0]
    x_exp = prev[1] + v * dt * math.cos(prev[3])
    y_exp = prev[2] + v * dt * math.sin(prev[3])
    return (math.hypot(now[1] - x_exp, now[2] - y_exp) > JUMP_POS
            or abs(_wrap(now[3] - prev[3] - w * dt)) > JUMP_YAW)


class ObstacleList:
    def __init__(self, wall_distance, mount, aligner=None, range_offset=0.0, freeze_on_refused=False,
                 check_pose=False):
        self.wd = wall_distance
        self.mount = mount                   # (x, y, yaw) of the LiDAR relative to the pose's point
        self.aligner = aligner               # scan_align.ScanAligner or None (step 3.1 behaviour)
        self.range_offset = float(range_offset)   # m this LiDAR measures too far (step 3.3); 0 = no correction
        self.freeze_on_refused = bool(freeze_on_refused)   # phantom fix part a (see 2b); False = before the fix
        self.check_pose = bool(check_pose)                 # phantom fix part b (see 2c); False = before the fix
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
        aligned, correction, align_us, refused = False, (0.0, 0.0), 0.0, ''
        check, check_us = None, 0.0
        if ok and self.aligner is not None and pose_std is not None:
            corrected = correct_ranges(ranges, range_min, range_max, self.range_offset) if self.range_offset else ranges
            t0 = time.perf_counter()
            al = self.aligner.align(corrected, angle_min, angle_inc, range_min, range_max, (x, y, yaw), pose_std,
                                    self.mount)
            align_us = (time.perf_counter() - t0) * 1e6
            if al['ok']:
                (x, y, yaw), pose_std = al['pose'], al['std']
                aligned, correction = True, (al['shift'], al['turn'])
                ranges = corrected                             # see 0.: only an aligned scan uses them
                if self.check_pose:                            # 2c
                    t1 = time.perf_counter()
                    check = pose_check(self.aligner.field, ranges, angle_min, angle_inc, range_min, range_max,
                                       (x, y, yaw), self.mount)
                    check_us = (time.perf_counter() - t1) * 1e6
            else:
                refused = al.get('reason') or 'refused'
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
        elif (share > MAX_NEW_SHARE or (self.freeze_on_refused and refused == REFUSED_TOO_FAR)
              or (check is not None and check['suspect'])):
            status = STATUS_FROZEN_SCAN
        else:
            objs = objects(px[keep], py[keep], (lx, ly))
            n_obj = len(objs)
            scan = dict(ranges=ranges, angle_min=angle_min, angle_inc=angle_inc, range_min=range_min,
                        range_max=range_max, lidar=(lx, ly, lyaw))
            self.tracker.update(t, objs, scan, pose_std)
            status, updated = STATUS_OK, True
        if not updated and (self.freeze_on_refused or self.check_pose):
            self.tracker.expire(t)     # a frozen scan still ages TENTATIVE obstacles (ObstacleTracker.expire)
        confirmed = [tr for tr in self.tracker.tracks if tr.confirmed]
        return dict(status=STATUS_RESET if reset else status, epoch=self.epoch, obstacles=confirmed,
                    seen_now={tr.id: bool(updated and tr.seen_now) for tr in confirmed},
                    pose=(x, y, yaw), pose_std=pose_std, aligned=aligned, correction=correction, align_us=align_us,
                    refused=refused, check=check, check_us=check_us,
                    new_share=share, n_points=len(px), n_objects=n_obj)
