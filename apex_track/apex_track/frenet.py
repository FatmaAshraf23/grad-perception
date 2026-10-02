#!/usr/bin/env python3
"""Frenet (track-relative) coordinates on a CLOSED centre line.  Pure numpy.

Converts a pose (x, y, yaw) in the map frame into the coordinates control's
MPC uses (APEX state contract):
    s_abs  [m]   distance along the centre line, CONTINUOUS across laps
                 (lap 2 starts at L, lap 3 at 2L, ...); s = s_abs mod L
    e_y    [m]   lateral offset from the centre line, + = LEFT of it
    e_psi  [rad] heading error = yaw - centre-line tangent angle, in (-pi, pi]
    kappa  [1/m] centre-line curvature at s (+ = turning left) -- control
                 needs it for the Frenet dynamics  s_dot = (vx cos e_psi -
                 vy sin e_psi) / (1 - kappa e_y)

WHY SMOOTH THE CENTRE LINE FIRST
    The centre line comes from map pixels (make_centerline.py), so its
    direction jumps up to 30 deg between neighbouring 0.2 m points in the
    corners. Used raw, e_psi would jump by the same amount when the car passes
    a waypoint. The loop is therefore smoothed (circular Gaussian along the
    arc, `smooth_m`) and resampled every `ds` metres; tangent angle and
    curvature are computed from the smooth curve and interpolated linearly.
    CONTROL MUST USE THE SAME SMOOTHED TRACK (same file + same smooth_m), or
    s / e_y / e_psi mean slightly different things for the two of us.

HOW THE PROJECTION WORKS
    The closest point on the (dense) polyline is searched only in a window
    around the previous s (fast, and it can never jump to a neighbouring
    corridor). If there is no previous s, or the result is implausible
    (|e_y| > max_e_y), a global search over the whole loop is done.

LAP COUNTING
    s wraps from ~L to ~0 at the start line (the first centre-line point).
    A jump of more than L/2 between two calls = one lap forward (or back).

Real car: identical. Only the centre-line CSV changes (from the slam_toolbox
map of our own track, made with make_centerline.py). No ROS imports here, so
it runs anywhere (Pi 4 included) and can be tested offline.
"""
import csv
import math

import numpy as np


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def load_xy_csv(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].strip().startswith(('x', '#'))]
    return np.array([[float(r[0]), float(r[1])] for r in rows])


def _resample_closed(xy, ds):
    """Points every ds metres along a closed polyline (first point kept)."""
    closed = np.vstack([xy, xy[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    n = max(int(round(L / ds)), 8)
    t = np.linspace(0.0, L, n, endpoint=False)
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


def _smooth_closed(xy, sigma_pts):
    """Circular Gaussian smoothing of a closed loop (sigma in points)."""
    if sigma_pts <= 0:
        return xy.copy()
    half = int(math.ceil(4 * sigma_pts))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_pts) ** 2)
    k /= k.sum()
    n = len(xy)
    idx = (np.arange(n)[:, None] + np.arange(-half, half + 1)[None, :]) % n
    return np.column_stack([xy[idx, 0] @ k, xy[idx, 1] @ k])


class FrenetTrack:

    def __init__(self, xy, smooth_m=0.5, ds=0.05, window_m=3.0, max_e_y=2.0):
        xy = np.asarray(xy, dtype=float)
        dense = _resample_closed(xy, ds)
        dense = _smooth_closed(dense, smooth_m / ds)
        dense = _resample_closed(dense, ds)          # uniform spacing again after smoothing
        self.p = dense                               # (n, 2) points
        nxt = np.roll(dense, -1, axis=0)
        self.d = nxt - dense                         # segment vectors
        self.seg_len = np.hypot(self.d[:, 0], self.d[:, 1])
        self.s0 = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])   # s at each point
        self.L = float(self.s0[-1] + self.seg_len[-1])
        self.n = len(dense)
        # tangent angle per segment, unwrapped so it can be interpolated
        self.theta_seg = np.unwrap(np.arctan2(self.d[:, 1], self.d[:, 0]))
        # curvature: d(theta)/ds, central difference on the closed loop
        th = self.theta_seg
        # over +-m segments (~ +-10 cm), so a polygon corner doesn't give a spike
        m = max(1, int(round(0.1 / ds)))
        dth = np.angle(np.exp(1j * (np.roll(th, -m) - np.roll(th, m))))
        self.kappa_seg = dth / (2 * m * ds)
        self.window_n = max(int(window_m / ds), 5)
        self.max_e_y = max_e_y
        self.s_prev = None           # s in [0, L) of the last call
        self.lap = 0

    @classmethod
    def from_csv(cls, path, **kw):
        return cls(load_xy_csv(path), **kw)

    # --- geometry --------------------------------------------------------------

    def _project(self, x, y, idx):
        """Closest point on the segments `idx`. Returns (i, t, dist2)."""
        p, d = self.p[idx], self.d[idx]
        L2 = np.maximum(self.seg_len[idx] ** 2, 1e-12)
        t = np.clip(((x - p[:, 0]) * d[:, 0] + (y - p[:, 1]) * d[:, 1]) / L2, 0.0, 1.0)
        qx, qy = p[:, 0] + t * d[:, 0], p[:, 1] + t * d[:, 1]
        dist2 = (x - qx) ** 2 + (y - qy) ** 2
        j = int(np.argmin(dist2))
        return int(idx[j]), float(t[j]), float(dist2[j])

    def _at(self, i, t):
        """theta and kappa at point i + t (interpolated between segment values)."""
        i2 = (i + 1) % self.n
        # theta: interpolate from the middle of segment i to the middle of i+1,
        # so the tangent turns smoothly instead of stepping at every point
        if t >= 0.5:
            a, b, f = i, i2, t - 0.5
        else:
            a, b, f = (i - 1) % self.n, i, t + 0.5
        th_a = self.theta_seg[a]
        th_b = th_a + wrap(self.theta_seg[b] - th_a)
        theta = wrap(th_a + f * (th_b - th_a))
        kappa = (1 - f) * self.kappa_seg[a] + f * self.kappa_seg[b]
        return theta, float(kappa)

    def _eval(self, s):
        """Centre-line point, smooth tangent angle and curvature at s in [0, L)."""
        i = min(max(int(np.searchsorted(self.s0, s, side='right')) - 1, 0), self.n - 1)
        t = (s - self.s0[i]) / self.seg_len[i]
        theta, kappa = self._at(i, t)
        return self.p[i, 0] + t * self.d[i, 0], self.p[i, 1] + t * self.d[i, 1], theta, kappa

    def _refine(self, x, y, s):
        """Move s until (pose - c(s)) is exactly perpendicular to the SMOOTH tangent.

        Why (geometry v2, 2026-10-02): the closest point on the 5 cm polyline
        and the smoothly interpolated tangent disagree near each polyline
        kink; in Levine's sharpest corners that gave up to 11 mm in s and
        17 mrad in e_psi on a pose -> Frenet -> pose round trip. A few Newton
        steps on f(s) = (p - c(s)) . T(s) = 0 make position and direction come
        from the same s, so the round trip is exact.
        f'(s) = -(1 - kappa*e_y), the same factor as in the Frenet dynamics.
        """
        for _ in range(8):
            cx, cy, th, k = self._eval(s)
            dx, dy = x - cx, y - cy
            f = dx * math.cos(th) + dy * math.sin(th)          # along-track miss
            if abs(f) < 1e-9:
                break
            e_y = -dx * math.sin(th) + dy * math.cos(th)
            den = max(1.0 - k * e_y, 0.2)                     # stay away from the singular point
            step = max(-2 * self.seg_len.max(), min(2 * self.seg_len.max(), f / den))
            s = (s + step) % self.L
        cx, cy, th, k = self._eval(s)
        return s, cx, cy, th, k

    def project(self, x, y, yaw):
        """(s_abs, s, e_y, e_psi, kappa) for a pose; updates the lap counter."""
        if self.s_prev is None:
            i, t, _ = self._project(x, y, np.arange(self.n))
        else:
            c = int(self.s_prev / self.L * self.n) % self.n
            idx = (c + np.arange(-self.window_n, self.window_n + 1)) % self.n
            i, t, d2 = self._project(x, y, idx)
            if d2 > self.max_e_y ** 2:              # lost the track -> global search
                i, t, _ = self._project(x, y, np.arange(self.n))

        s = (self.s0[i] + t * self.seg_len[i]) % self.L
        s, qx, qy, theta, kappa = self._refine(x, y, s)
        e_y = math.cos(theta) * (y - qy) - math.sin(theta) * (x - qx)   # + = left
        e_psi = wrap(yaw - theta)

        if self.s_prev is not None:
            if s - self.s_prev < -0.5 * self.L:
                self.lap += 1
            elif s - self.s_prev > 0.5 * self.L:
                self.lap -= 1
        self.s_prev = s
        return self.lap * self.L + s, s, e_y, e_psi, kappa

    def reset(self, lap=0):
        self.s_prev = None
        self.lap = lap

    def point(self, s):
        """Centre-line (x, y, theta) at s (any s, wraps)."""
        s = s % self.L
        i = min(int(np.searchsorted(self.s0, s, side='right')) - 1, self.n - 1)
        t = (s - self.s0[i]) / self.seg_len[i]
        th, _ = self._at(i, t)
        return (self.p[i, 0] + t * self.d[i, 0], self.p[i, 1] + t * self.d[i, 1], th)

    def kappa_at(self, s):
        """Centre-line curvature [1/m] at s (+ = left turn). Any s (wraps, so
        s_abs works too). s can be one number or a numpy array -- control's MPC
        asks for the whole prediction horizon in one call.

        Uses exactly the same interpolation as project(), so
        kappa_at(s returned by project) == kappa returned by project.
        """
        s_arr = np.atleast_1d(np.asarray(s, dtype=float)) % self.L
        i = np.searchsorted(self.s0, s_arr, side='right') - 1
        i = np.clip(i, 0, self.n - 1)
        t = (s_arr - self.s0[i]) / self.seg_len[i]
        # same rule as _at(): between the middles of two neighbouring segments
        upper = t >= 0.5
        a = np.where(upper, i, (i - 1) % self.n)
        b = np.where(upper, (i + 1) % self.n, i)
        f = np.where(upper, t - 0.5, t + 0.5)
        k = (1 - f) * self.kappa_seg[a] + f * self.kappa_seg[b]
        return float(k[0]) if np.ndim(s) == 0 else k
