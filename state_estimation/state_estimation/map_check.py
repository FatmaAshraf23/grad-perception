#!/usr/bin/env python3
"""map_check.py -- FAST LiDAR-vs-map check of the EKF's ALONG-TRACK position (2026-10-21). numpy only, no ROS.

WHY
    On a straight, AMCL only knows the car's position ACROSS the track ("corridor" fixes). An
    along-track error of the EKF (wheel over-read / spin, a pose kick, ...) therefore stays
    invisible until the next corner: fault scenarios F05 / F06 / F08 (jobs 175-177) had the
    mode OK for 1.0-2.3 s with 0.5-0.9 m error. But the LiDAR also sees the walls AHEAD and
    BEHIND (end of the straight, the next corner). Sliding the EKF pose along its heading until
    the scan fits the map best MEASURES the along error (offline jobs 178-179: median 5 cm).

HOW -- AND WHY IT IS CHEAP ENOUGH FOR THE PI 5
    The offline reference ray-cast all 180 beams again for each of the 41 candidate shifts
    (~70 ms per scan on the laptop -- half a core at 7 Hz). Here:
    1. ONCE, when the map arrives: a DISTANCE FIELD = for every map cell, the distance to the
       nearest wall cell (capped at cap_m). Built with scipy if present, else numpy (~0.1 s).
    2. Per scan: the END POINT of every beam (where it hit something) in map coordinates. A beam
       fits if its end point lies within tol of a wall.
    3. Moving the pose by d along the heading moves EVERY end point by the same vector
       d * (cos yaw, sin yaw). So each candidate shift costs ONE table lookup per beam:
       41 shifts x 180 beams = 7 380 lookups, all in a few vectorized numpy calls.
    Side-wall beams slide ALONG their wall when the pose moves along a straight -> their fit
    does not change. Only beams on walls facing the direction of travel react -> this is
    exactly the along-track information, and it is not diluted by AMCL's odometry.
    Background: the "likelihood field" range-sensor model (Thrun, Burgard, Fox, Probabilistic
    Robotics, ch. 6.4) and lookup-table correlative scan matching (E. Olson, ICRA 2009).

SIGN
    shift > 0: the scan fits best with the pose moved FORWARD -> the EKF is BEHIND the truth:
    true_pose ~= ekf_pose + shift * (cos yaw, sin yaw); EKF along error = -shift.

USE (ekf_node, ekf_global)
    field = DistanceField.from_grid(...)              # once, when /map arrives
    res = shift_search(field, pose, laser, angles, ranges, SHIFTS)   # per scan
    alarm = MismatchAlarm(...).update(res, t_scan_s)  # contract v0.3: on / off with hysteresis
    budget.every_n                                    # run the check only on every n-th scan
    (ShiftMonitor = the first, shadow-mode rule of 2026-10-21: along-track only, no hysteresis.)
"""
import math
import time
from collections import namedtuple

import numpy as np

ShiftResult = namedtuple('ShiftResult', 'shift gain width fit0 fit_best n_beams')


def make_shifts(max_shift=1.0, step=0.05):
    """Symmetric candidate shifts including 0 (index len // 2)."""
    n = int(round(max_shift / step))
    return (np.arange(-n, n + 1) * step).astype(np.float64)


def _distance_cells(occ, cap_cells, use_scipy=True):
    """Euclidean distance (in cells) from every cell to the nearest occupied cell, capped.
    scipy if installed (instant), else numpy over all offsets inside the cap (~0.1-0.5 s, once)."""
    if not occ.any():
        return np.full(occ.shape, float(cap_cells), np.float32)
    if use_scipy:
        try:
            from scipy.ndimage import distance_transform_edt
            return np.minimum(distance_transform_edt(~occ), cap_cells).astype(np.float32)
        except ImportError:
            pass
    h, w = occ.shape
    r = int(math.ceil(cap_cells))
    dist = np.full((h, w), float(cap_cells), np.float32)
    pad = np.zeros((h + 2 * r, w + 2 * r), dtype=bool)
    pad[r:r + h, r:r + w] = occ
    offs = sorted((math.hypot(dx, dy), dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                  if math.hypot(dx, dy) < cap_cells)
    for dd, dx, dy in offs:                       # cell c is dd from a wall if c + (dx, dy) is a wall
        hit = pad[r + dy:r + dy + h, r + dx:r + dx + w]
        np.minimum(dist, np.where(hit, np.float32(dd), np.float32(cap_cells)), out=dist)
    return dist


class DistanceField:
    """Distance [m] from each map cell to the nearest wall cell, capped at cap_m. Same cell
    convention as ekf_node.WallMap / ROS OccupancyGrid: cell (row, col) spans
    origin + [col, col + 1) * res in x and origin + [row, row + 1) * res in y, row 0 = BOTTOM.
    Points outside the map read cap_m (= 'far from any wall')."""

    def __init__(self, occupied, res, ox, oy, cap_m=0.5):
        occ = np.asarray(occupied, dtype=bool)
        self.res, self.ox, self.oy, self.cap = float(res), float(ox), float(oy), float(cap_m)
        self.h, self.w = occ.shape
        d = _distance_cells(occ, self.cap / self.res) * np.float32(self.res)
        self.grid = np.full((self.h + 2, self.w + 2), np.float32(self.cap), np.float32)
        self.grid[1:-1, 1:-1] = d                 # 1-cell border of cap_m: clipped indices land there
        self.n_walls = int(occ.sum())

    @classmethod
    def from_grid(cls, data, width, height, res, ox, oy, occupied_from=65, cap_m=0.5):
        """OccupancyGrid data: 0 free, 100 wall, -1 unknown (unknown = not a wall, like WallMap)."""
        grid = np.asarray(data, dtype=np.int16).reshape(height, width)
        return cls(grid >= occupied_from, res, ox, oy, cap_m)

    def lookup(self, xs, ys):
        col = np.floor((xs - self.ox) / self.res).astype(np.int32) + 1
        row = np.floor((ys - self.oy) / self.res).astype(np.int32) + 1
        np.clip(col, 0, self.w + 1, out=col)
        np.clip(row, 0, self.h + 1, out=row)
        return self.grid[row, col]


def shift_search(field, pose, laser, beam_angles, ranges, shifts, tol=0.10, min_beams=20, width_band=0.02,
                 peak='center', sigma=0.05):
    """Best along-heading shift of `pose` (x, y, yaw of base_link) for one scan.
    laser = (x, y, yaw) of the LiDAR on the car; beam_angles / ranges in the LiDAR frame,
    inf / nan = no return (ignored). Returns ShiftResult or None (< min_beams returns).
    fit = share of beams whose end point lies within tol of a wall. It is flat on top: a beam
    keeps fitting while its end point moves +-tol across the wall, so the best fit is a plateau
    ~2 tol wide around the true shift. Where in that plateau is the answer (peak):
      'center'  the middle of the plateau around the best fit (default; unbiased)
      'soft'    the maximum of mean(exp(-d^2 / 2 sigma^2)) (likelihood-field score; one sharp peak)
      'nearest' the plateau edge closest to 0 (biased towards 0 by ~tol; first version)
    gain / width / fit0 always come from the plain fit, so the alarm rule does not change."""
    r = np.asarray(ranges, dtype=np.float64)
    ok = np.isfinite(r)
    n = int(ok.sum())
    if n < min_beams:
        return None
    a = np.asarray(beam_angles, dtype=np.float64)[ok]
    r = r[ok]
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = math.cos(yaw), math.sin(yaw)
    lx = x + c * laser[0] - s * laser[1]
    ly = y + s * laser[0] + c * laser[1]
    th = yaw + laser[2] + a
    ex = lx + r * np.cos(th)                      # end points at shift 0 (map frame)
    ey = ly + r * np.sin(th)
    d = field.lookup(ex[None, :] + (shifts * c)[:, None], ey[None, :] + (shifts * s)[:, None])
    fits = np.count_nonzero(d < tol, axis=1) / float(n)
    i0 = len(shifts) // 2
    best = float(fits.max())
    top = np.flatnonzero(fits >= best - 1e-12)
    b = int(top[np.argmin(np.abs(shifts[top]))])  # the best-fit shift closest to 0
    if peak == 'center':                           # middle of the contiguous plateau that contains b
        lo = hi = b
        while lo > 0 and fits[lo - 1] >= best - 1e-12:
            lo -= 1
        while hi < len(fits) - 1 and fits[hi + 1] >= best - 1e-12:
            hi += 1
        shift = 0.5 * (float(shifts[lo]) + float(shifts[hi]))
    elif peak == 'soft':
        soft = np.exp(-0.5 * (d / sigma) ** 2).mean(axis=1)
        st = np.flatnonzero(soft >= soft.max() - 1e-12)
        shift = float(shifts[st[np.argmin(np.abs(shifts[st]))]])
    else:
        shift = float(shifts[b])
    near = shifts[fits >= best - width_band]
    return ShiftResult(shift, best - float(fits[i0]), float(near.max() - near.min()),
                       float(fits[i0]), best, n)


class ShiftMonitor:
    """Alarm when k results in a row show a clear along-track error:
    s_min <= |shift| <= s_edge (a best shift at the edge of the search window is no measurement:
    the true optimum may lie further out), gain >= g_min (the scan fits clearly better there)
    and width <= w_max (one sharp optimum, not a flat corridor)."""

    def __init__(self, s_min=0.3, s_edge=0.95, g_min=0.05, w_max=0.3, k=2):
        self.s_min, self.s_edge, self.g_min, self.w_max, self.k = s_min, s_edge, g_min, w_max, int(k)
        self.run = 0
        self.alarm = False

    def valid(self, res):
        return (res is not None and self.s_min <= abs(res.shift) <= self.s_edge
                and res.gain >= self.g_min and res.width <= self.w_max)

    def update(self, res):
        self.run = self.run + 1 if self.valid(res) else 0
        self.alarm = self.run >= self.k
        return self.alarm


# Scan classes of MismatchAlarm (also field 10 of ekf_node's <diag_prefix>/map_check message)
NO_RESULT, NEITHER, CLEAN, BAD_ALONG, BAD_FIT, BAD_BOTH = -1, 0, 1, 2, 3, 4
CLASS_NAMES = {NO_RESULT: 'no result', NEITHER: 'neither', CLEAN: 'clean', BAD_ALONG: 'bad (along)',
               BAD_FIT: 'bad (fit)', BAD_BOTH: 'bad (along + fit)'}


class MismatchAlarm:
    """MAP-MISMATCH ALARM WITH HYSTERESIS (contract v0.3, FLAG_MAP_MISMATCH = 1024), 2026-10-24.

    Every checked scan (the shift_search result at the EKF pose, or None) is put in one class:
      BAD     the scan says the pose is wrong, for one of two reasons (or both):
              along  s_min <= |shift| <= s_edge, gain >= g_min, width <= w_max: one clear optimum that
                     far along the heading (the ShiftMonitor rule; it MEASURES along-track errors on
                     straights, where AMCL only gives corridor fixes);
              fit    fit0 < fit_bad: the scan hardly fits the map at the EKF pose at all (a sideways or
                     heading error, or a pose further off than the search window).
      CLEAN   nothing fits clearly better than the pose itself (gain <= g_clear) AND the scan fits the
              map well there (fit0 >= fit_clear). The fit part matters: a pose far off fits badly at
              EVERY shift, so its gain is small too -- without it, that pose would look clean.
      NEITHER everything else (results between the ON and OFF thresholds, or no result = too few
              returns): both counts restart and the alarm KEEPS ITS STATE.
    ON   after k_on BAD scans in a row.
    OFF  after k_clear CLEAN scans in a row, and only once it has been on for >= min_on_s (scan time).
    The gap between the thresholds (g_clear < gain < g_min, fit_bad <= fit0 < fit_clear) is the
    hysteresis: one noisy value cannot switch the alarm on and off again.

    Data (simulator, 2026-10-24): 12 clean runs, 4744 driving scans at the EKF pose: fit0 >= 0.87 (p0.1 0.92,
    p1 0.97), gain <= 0.02 in 99.5 % of the scans. Fault runs, EKF > 30 cm off (208 scans): fit0 median 0.08;
    only a pose 5.9 m off in a featureless corridor (F04) ever looked clean (2 scans with fit0 0.8-0.85),
    and an exact synthetic corridor scan 3 m off gives fit0 0.867, gain 0.022 -> fit_clear 0.9, not lower:
    in a corridor only the few beams on the end walls tell a far-off pose from the right one.
    Real car: refit fit_bad / fit_clear on real laps (an imperfect map or people near the walls lower fit0)."""

    def __init__(self, s_min=0.30, s_edge=0.95, g_min=0.05, w_max=0.30, k_on=2, fit_bad=0.5,
                 g_clear=0.02, fit_clear=0.9, k_clear=4, min_on_s=0.5):
        self.along = ShiftMonitor(s_min, s_edge, g_min, w_max, 1)        # its valid() = the along rule
        self.fit_bad, self.g_clear, self.fit_clear = float(fit_bad), float(g_clear), float(fit_clear)
        self.k_on, self.k_clear, self.min_on_s = int(k_on), int(k_clear), float(min_on_s)
        if not self.fit_bad < self.fit_clear or not self.g_clear < g_min:
            raise ValueError('MismatchAlarm needs fit_bad < fit_clear and g_clear < g_min (hysteresis gap)')
        self.reset()

    def reset(self):
        self.alarm = False
        self.n_bad = 0                         # BAD scans in a row
        self.n_clean = 0                       # CLEAN scans in a row
        self.t_on = None                       # scan time the alarm went on
        self.cause = NEITHER                   # class of the scan that switched it on
        self.last_class = NO_RESULT

    def classify(self, res):
        if res is None:
            return NO_RESULT
        along = self.along.valid(res)
        fit = res.fit0 < self.fit_bad
        if along or fit:
            return BAD_BOTH if (along and fit) else (BAD_ALONG if along else BAD_FIT)
        if res.gain <= self.g_clear and res.fit0 >= self.fit_clear:
            return CLEAN
        return NEITHER

    def update(self, res, t_s):
        """One checked scan (t_s = its stamp in s) -> the alarm state after it."""
        c = self.classify(res)
        self.last_class = c
        self.n_bad = self.n_bad + 1 if c >= BAD_ALONG else 0
        self.n_clean = self.n_clean + 1 if c == CLEAN else 0
        if not self.alarm:
            if self.n_bad >= self.k_on:
                self.alarm, self.t_on, self.cause = True, float(t_s), c
        elif self.n_clean >= self.k_clear and float(t_s) - self.t_on >= self.min_on_s:
            self.alarm, self.t_on = False, None
        return self.alarm


class CpuBudget:
    """Keeps the check's CPU time per scan within budget_ms ON AVERAGE: measures every run
    (exponential mean) and runs the check only on every n-th scan if one run costs more
    (n = ceil(mean / budget), 1..max_every). On the Pi 5 the check should cost ~1 ms
    -> n = 1; if the Pi is busier than expected it thins out instead of starving the EKF."""

    def __init__(self, budget_ms=3.0, max_every=7, alpha=0.1):
        self.budget_ms, self.max_every, self.alpha = float(budget_ms), int(max_every), float(alpha)
        self.mean_ms = None
        self.max_ms = 0.0
        self.every_n = 1
        self.count = 0
        self.runs = 0

    def due(self):
        """Call once per scan: True if the check should run on this scan."""
        self.count += 1
        return (self.count % self.every_n) == 0

    def start(self):
        return time.perf_counter()

    def stop(self, t0):
        ms = (time.perf_counter() - t0) * 1000.0
        self.add(ms)
        return ms

    def add(self, ms):
        self.runs += 1
        self.max_ms = max(self.max_ms, ms)
        self.mean_ms = ms if self.mean_ms is None else (1 - self.alpha) * self.mean_ms + self.alpha * ms
        self.every_n = int(min(self.max_every, max(1, math.ceil(self.mean_ms / self.budget_ms))))
