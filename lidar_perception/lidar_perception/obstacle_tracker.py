#!/usr/bin/env python3
"""obstacle_tracker.py -- obstacle detection step 2.4: from objects per scan to ONE stable obstacle list.

map_difference.detect() judges every scan on its own. Seen that way the obstacles
  - flicker: a cone 5 m away is found in some scans and not in others (too few LiDAR points),
  - jitter:  the estimated centre moves by a few cm from scan to scan,
  - vanish:  a cone is forgotten as soon as it is out of sight (behind a corner, behind the car).
The tracker keeps a list of obstacles ("tracks") over time. For every scan:
  1. MATCH     each new object to the nearest known obstacle closer than GATE, closest pairs first.
  2. UPDATE    a matched obstacle's position = an AVERAGE of its detections, done PASS BY PASS (step 3.1):
               a detection = the car's pose + what the LiDAR measured from there, so its error has two parts:
                 (a) the LiDAR part (which beams hit it, noise, the centre guess): SIGMA_DET per scan, but
                     similar in neighbouring scans (correlation RHO_DET: the view changes slowly)
                 (b) the POSE part (StateEstimate's position error + heading error x range): the SAME for all
                     scans while the car drives past once (= one PASS), different on the next lap.
               So: average the scans of one pass (removes most of (a), none of (b)) -> the pass counts as ONE
               measurement with variance  V = SIGMA_DET^2 * mean_var_factor(n) + S^2,  S = the pose sigma
               StateEstimate reported (std_xy, std_psi x range); detections made with a surer pose weigh more.
               Passes = drive-bys (no detection for > PASS_GAP) are averaged, each weighted by 1 / V: a lap
               with a good pose counts more than one right after a localization jump, and the pose errors of
               different laps average out. Between passes Q_PASS is added ("it may have been nudged").
               Before (step 2.4-3): a Kalman filter with R = (4 cm)^2 for every detection and P growing by
               (1 cm)^2 per SECOND -- it ignored how sure the pose was (live run B: a cone listed right after
               the EKF jumped back was 6-8.6 cm off while sigma said 2.3 cm) and after ~25 s out of sight the
               first far detection got ~60 % weight (remembered cones jumped by up to 9 cm when seen again).
  3. NEW       an object that matches nothing starts a TENTATIVE obstacle.
  4. CONFIRM   a tentative obstacle after CONFIRM_HITS detections. A one-scan blip never gets there;
               a tentative obstacle that is not seen again within TENTATIVE_TIMEOUT is dropped.
               The price: an obstacle is reported one scan (~0.14 s) after it is first found.
  5. REMEMBER  confirmed obstacles stay in the list while out of sight -- a cone does not walk away.
  6. FORGET    (step 2.5, only when update() gets the scan) ...unless it really is gone. NOT being
               detected proves nothing (too far, behind a corner, only one LiDAR point): absence of
               evidence. Proof is EVIDENCE OF ABSENCE: a laser beam that should have hit the obstacle
               went straight THROUGH its place and hit something behind it (look_through()). After
               DELETE_AFTER such scans in a row, with no beam hitting it in between, it is deleted.
               A beam with NO return never counts: a dark or shiny cone can swallow the laser, and
               "I got nothing back" must not turn into "there is nothing there".
Only CONFIRMED obstacles are the output (what planning gets). Positions are in the map frame.

Reported uncertainty (sigma, per axis) = sqrt(variance of the pass average + SIGMA_FLOOR^2). The floor is the
part that repeats on EVERY lap (the viewing geometry is the same each time) and so never averages out.
History of this: step 2.4 found consecutive errors 50-80 % correlated and added a 2 cm floor to an otherwise
"independent detections" filter; step 3.1 measured where the correlation comes from (step3_1_noise.py, job 078
recording, every scan detected with the TRUE and with the EKF pose): LiDAR part 2.6 cm per scan with
correlation 0.71 between neighbouring scans, a pass average 0.2-1.6 cm from the true cone; the rest is the pose.
Same lesson as the EKF's AMCL covariance (2026-10-06): correlated errors make a filter overconfident.
numpy only, no ROS.
"""
import math

import numpy as np

GATE = 0.20               # m: a detection farther than this from an obstacle is something else.
                          #    Step 2.3: single detections are at most ~11 cm off. 20 cm kept one id per
                          #    cone and confirmed 2-3x fewer fake blips than 35 cm. Real car: if one cone
                          #    splits into two ids, raise it (noisier LiDAR, round cones).
SIGMA_DET = 0.03          # m per axis: LiDAR part of one detection's error (step 3.1, true pose: 2.6 cm)
RHO_DET = 0.7             # correlation of that part between neighbouring scans (step 3.1, true pose: 0.71)
SIGMA_FLOOR = 0.01        # m per axis: viewing-geometry error that repeats every lap (step 3.1: pass averages
                          #    with the true pose 0.2-1.6 cm off); added to the reported sigma
PASS_GAP = 3.0            # s: a detection after this long without one starts a new PASS (drive-by). The pose
                          #    error stays correlated for ~2-3 s (sigma study 2026-10-06: AMCL along-track);
                          #    in the recordings every gap inside a drive-by is < 1 s, laps are ~30 s apart
Q_PASS = 0.02 ** 2        # m^2 added to the earlier passes' estimate when a new pass starts: the obstacle may
                          #    have been nudged in between -- keeps new laps able to correct the position
DEFAULT_POSE_STD = (0.03, math.radians(0.3))   # (std_xy m, std_psi rad) when update() is not told the pose's
                          #    uncertainty (offline scripts); live: StateEstimate's std_xy / std_psi
CONFIRM_HITS = 2          # detections needed before an obstacle is reported
TENTATIVE_TIMEOUT = 0.5   # s: a tentative obstacle not seen again within this is dropped (~3 scans at 7 Hz)
# step 2.5 -- forgetting obstacles that are gone (look_through)
CHECK_RANGE = 4.0         # m: only judge obstacles this close: up to 4 m every cone was found in 100 % of
                          #    the scans (step 2.3); farther, a missed detection is normal
MIN_CHECK_RANGE = 0.3     # m: closer than this the car's own body / the LiDAR's minimum range get in the way
R_CORE = 0.04             # m: use only beams passing within 4 cm of the obstacle's centre -- they must have hit a
                          #    15 cm cone if it were there, even with the estimate a few cm off
NEAR_MARGIN = 0.25        # m: a beam ending up to this far IN FRONT of the centre counts as hitting the obstacle
PASS_MARGIN = 0.20        # m: a beam ending more than this far BEHIND the centre went through its place
DELETE_AFTER = 3          # scans in a row with beams going through (and none hitting) -> delete (~0.4 s at 7 Hz)


def pose_sigma(pose_std, rng):
    """Per-axis error [m] the POSE adds to a detection `rng` metres from the LiDAR: position + heading x range."""
    std_xy, std_psi = pose_std
    return math.sqrt(std_xy ** 2 + (rng * std_psi) ** 2)


def mean_var_factor(n, rho=RHO_DET):
    """Variance of the average of n detections whose errors are correlated rho from one scan to the next (AR(1)),
    as a share of one detection's variance: 1 for n = 1, (1 + rho) / 2 for n = 2, ~(1 + rho) / ((1 - rho) n) for
    large n. rho 0.7: averaging 50 scans is worth ~9 independent ones, not 50."""
    if n <= 1:
        return 1.0
    s = rho / (1.0 - rho) - rho * (1.0 - rho ** n) / (n * (1.0 - rho) ** 2)
    return (1.0 + 2.0 * s) / n


def combine(a, b):
    """Two independent estimates (x, y, variance) of the same point -> their inverse-variance average."""
    if a is None:
        return b
    wa = b[2] / (a[2] + b[2])                                  # a's weight = how much MORE unsure b is
    return (wa * a[0] + (1.0 - wa) * b[0], wa * a[1] + (1.0 - wa) * b[1], a[2] * b[2] / (a[2] + b[2]))


class Track:
    """One obstacle the tracker knows about."""

    def __init__(self, track_id, obj, t, pose_std=DEFAULT_POSE_STD):
        self.id = track_id
        self.history = None                   # (x, y, variance) of all FINISHED passes combined
        self.passes = 0                       # number of finished passes
        self.cur = None                       # running sums of the current pass
        self.hits = 0
        self.first_t = self.last_t = t
        self.confirmed = False
        self.confirm_t = None
        self.size = obj['size']
        self.seen_now = True                  # matched in the latest update?
        self.gone = 0                         # scans in a row in which the LiDAR looked through its place
        self.update(obj, t, pose_std)

    @property
    def sigma(self):
        """Reported 1-sigma uncertainty of the position [m] (per axis): passes + the floor."""
        return math.sqrt(self.var + SIGMA_FLOOR ** 2)

    def pass_estimate(self):
        """(x, y, variance) of the current pass: detections weighted by how sure the pose was; its variance =
        the LiDAR part after averaging + the pose part, which averaging within one pass does not reduce."""
        c = self.cur
        n_eff = c['sw'] ** 2 / c['sw2']                         # = n when all weights are equal
        pose = c['sws'] / c['sw']                               # weighted average pose sigma of the pass
        return c['swx'] / c['sw'], c['swy'] / c['sw'], SIGMA_DET ** 2 * mean_var_factor(n_eff) + pose ** 2

    def update(self, obj, t, pose_std=DEFAULT_POSE_STD):
        if self.cur is not None and t - self.last_t > PASS_GAP:            # a new drive-by: close the old pass
            hist = None if self.history is None else self.history[:2] + (self.history[2] + Q_PASS,)
            self.history = combine(hist, self.pass_estimate())
            self.passes += 1
            self.cur = None
        if self.cur is None:
            self.cur = dict(sw=0.0, sw2=0.0, swx=0.0, swy=0.0, sws=0.0)
        s = pose_sigma(pose_std, obj.get('range', 0.0))
        w = 1.0 / (SIGMA_DET ** 2 + s ** 2)
        c = self.cur
        c['sw'] += w
        c['sw2'] += w * w
        c['swx'] += w * obj['x']
        c['swy'] += w * obj['y']
        c['sws'] += w * s
        hist = None if self.history is None else self.history[:2] + (self.history[2] + Q_PASS,)
        self.x, self.y, self.var = combine(hist, self.pass_estimate())
        self.hits += 1
        self.last_t = t
        self.size = max(self.size, obj['size'])
        self.seen_now = True
        if not self.confirmed and self.hits >= CONFIRM_HITS:
            self.confirmed, self.confirm_t = True, t


def look_through(x, y, scan):
    """What one scan says about an obstacle believed to be at (x, y): 'hit', 'through' or None.
    scan = dict(ranges, angle_min, angle_inc, range_min, range_max, lidar=(x, y, yaw) of the LiDAR in the map).
    Only beams that come back with a real range AND pass within R_CORE of the centre are used:
      'hit'      a beam ends near the obstacle (NEAR_MARGIN in front of its centre .. PASS_MARGIN behind)
      'through'  none does, and a beam ends more than PASS_MARGIN BEHIND the centre: the laser went through
      None       no information: out of [MIN_CHECK_RANGE, CHECK_RANGE], no beam close enough, or every beam
                 ends in front of it (something else blocks the view)"""
    lx, ly, lyaw = scan['lidar']
    dx, dy = x - lx, y - ly
    if not MIN_CHECK_RANGE <= math.hypot(dx, dy) <= CHECK_RANGE:
        return None
    r = np.asarray(scan['ranges'], dtype=float)
    a = lyaw + scan['angle_min'] + np.arange(len(r)) * scan['angle_inc']
    along = np.cos(a) * dx + np.sin(a) * dy              # distance along each beam to the centre's foot point
    across = np.abs(np.cos(a) * dy - np.sin(a) * dx)     # how far each beam passes beside the centre
    valid = np.isfinite(r) & (r >= scan['range_min']) & (r < scan['range_max'] - 1e-3)   # no return = no info
    core = valid & (along > 0) & (across < R_CORE)
    if not core.any():
        return None
    rc, ac = r[core], along[core]
    if np.any((rc >= ac - NEAR_MARGIN) & (rc <= ac + PASS_MARGIN)):
        return 'hit'
    if np.any(rc > ac + PASS_MARGIN):
        return 'through'
    return None


class ObstacleTracker:
    def __init__(self):
        self.tracks = []
        self.deleted = []                     # (track, time) of every obstacle forgotten by rule 6
        self._next_id = 1

    def reset(self):
        """Forget everything (ids keep counting up). For a jump of the car's pose: every remembered position
        was computed with the old pose, and nothing tells which of them are still right (step 2.6)."""
        self.tracks = []

    def update(self, t, objects, scan=None, pose_std=None):
        """Feed the objects of one scan (time t [s]); with `scan` (see look_through) also rule 6 FORGET.
        pose_std = (std_xy [m], std_psi [rad]) of the pose the objects were computed with (StateEstimate's own
        uncertainty); None = DEFAULT_POSE_STD. Returns (confirmed tracks, the track each object went to)."""
        pose_std = DEFAULT_POSE_STD if pose_std is None else pose_std
        for tr in self.tracks:
            tr.seen_now = False
        # 1. MATCH: all (track, object) pairs closer than GATE, closest first; each used once
        pairs = sorted((math.hypot(tr.x - o['x'], tr.y - o['y']), ti, oi)
                       for ti, tr in enumerate(self.tracks) for oi, o in enumerate(objects))
        used_t, used_o, went_to = set(), set(), [None] * len(objects)
        for d, ti, oi in pairs:
            if d >= GATE:
                break
            if ti in used_t or oi in used_o:
                continue
            used_t.add(ti); used_o.add(oi)
            self.tracks[ti].update(objects[oi], t, pose_std)   # 2. UPDATE
            went_to[oi] = self.tracks[ti]
        for oi, o in enumerate(objects):                     # 3. NEW
            if oi not in used_o:
                tr = Track(self._next_id, o, t, pose_std)
                self._next_id += 1
                self.tracks.append(tr)
                went_to[oi] = tr
        # 4. drop tentative obstacles that were not seen again in time (confirmed ones are REMEMBERED)
        self.tracks = [tr for tr in self.tracks if tr.confirmed or t - tr.last_t <= TENTATIVE_TIMEOUT]
        # 6. FORGET a remembered obstacle the LiDAR looked straight through in DELETE_AFTER scans in a row
        if scan is not None:
            keep = []
            for tr in self.tracks:
                if tr.seen_now:
                    tr.gone = 0
                elif tr.confirmed:
                    verdict = look_through(tr.x, tr.y, scan)
                    if verdict == 'hit':
                        tr.gone = 0
                    elif verdict == 'through':
                        tr.gone += 1
                    if tr.gone >= DELETE_AFTER:
                        self.deleted.append((tr, t))
                        continue
                keep.append(tr)
            self.tracks = keep
        return [tr for tr in self.tracks if tr.confirmed], went_to
