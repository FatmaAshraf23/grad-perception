"""Rotating-LiDAR timing and de-skew math (no ROS) -- used by a1_emulator (simulation) and scan_deskew.

A spinning LiDAR (RPLIDAR A1: one turn ~1/7 s) measures its beams one after another while the car
moves. This module answers two questions:
  1. WHEN was element j of a scan message measured?      -> beam_offsets()
  2. WHERE would its point be, seen from the LiDAR at one common reference time?  -> deskew()

2-D poses are (x, y, psi). A pose P maps a point p from its own frame into the parent frame:
    P * p = (x + cos(psi) p_x - sin(psi) p_y,   y + sin(psi) p_x + cos(psi) p_y)
"""
import numpy as np


def beam_offsets(n, time_increment, order='reverse'):
    """Seconds after the scan's stamp at which element j (0..n-1) of the ranges array was measured.

    'reverse' = rplidar_ros (Slamtec ROS 2 driver, A1 launch defaults): the array runs in increasing
                angle, but the LiDAR turns clockwise starting behind the car, so the LAST element is
                measured FIRST: (n-1-j) * time_increment
    'forward' = the standard sensor_msgs/LaserScan meaning: j * time_increment
    """
    j = np.arange(n, dtype=float)
    if order == 'reverse':
        return (n - 1 - j) * time_increment
    if order == 'forward':
        return j * time_increment
    raise ValueError(f"time order '{order}': use 'reverse' (rplidar_ros) or 'forward' (LaserScan)")


def assemble(state_t, state_r, t_beam, edge_jump=0.15):
    """Ranges of a ROTATING scan made from instantaneous scans (simulation).

    state_t: (S,) increasing times of the simulator's physics states; state_r: (S, N) their ranges;
    t_beam: (N,) the moment each element is measured. Element j is interpolated linearly in time
    between the two states around t_beam[j] -- or taken from the nearer state when the range jumps
    by more than edge_jump (an object edge: interpolating would invent a point in the air)."""
    state_t = np.asarray(state_t, float)
    state_r = np.asarray(state_r, float)
    s, n = state_r.shape
    i = np.clip(np.searchsorted(state_t, t_beam, side='right') - 1, 0, s - 1)
    k = np.minimum(i + 1, s - 1)
    ta, tb = state_t[i], state_t[k]
    w = np.where(tb > ta, np.clip((t_beam - ta) / np.maximum(tb - ta, 1e-9), 0.0, 1.0), 0.0)
    cols = np.arange(n)
    ra, rb = state_r[i, cols], state_r[k, cols]
    smooth = np.isfinite(ra) & np.isfinite(rb) & (np.abs(rb - ra) <= edge_jump)
    nearest = np.where(w < 0.5, ra, rb)
    return np.where(smooth, ra + w * (rb - ra), nearest)


def compose(a, b):
    """a * b for 2-D poses (tuples of scalars or equal-length arrays)."""
    ax, ay, ap = a
    bx, by, bp = b
    c, s = np.cos(ap), np.sin(ap)
    return ax + c * bx - s * by, ay + s * bx + c * by, ap + bp


def inverse(a):
    """a^-1 for a 2-D pose."""
    ax, ay, ap = a
    c, s = np.cos(ap), np.sin(ap)
    return -c * ax - s * ay, s * ax - c * ay, -ap


def interp_poses(t_odo, x, y, psi, v, w, t_query, max_extrap):
    """Car poses at t_query from an odometry history (t_odo increasing, psi unwrapped).

    Inside the history: linear interpolation. After the newest message: constant speed v and yaw
    rate w of that message, for at most max_extrap seconds (the newest odometry is usually a few ms
    older than the scan's last beam). Returns (x, y, psi, ok); ok = False where neither works."""
    tq = np.atleast_1d(np.asarray(t_query, float))
    xi, yi, pi = np.interp(tq, t_odo, x), np.interp(tq, t_odo, y), np.interp(tq, t_odo, psi)
    late = tq > t_odo[-1]
    if np.any(late):
        dt = tq[late] - t_odo[-1]
        pm = psi[-1] + 0.5 * w[-1] * dt                 # heading in the middle of the step
        xi[late] = x[-1] + v[-1] * np.cos(pm) * dt
        yi[late] = y[-1] + v[-1] * np.sin(pm) * dt
        pi[late] = psi[-1] + w[-1] * dt
    ok = (tq >= t_odo[0] - 1e-6) & (tq <= t_odo[-1] + max_extrap)
    return xi, yi, pi, ok


def deskew(ranges, angles, poses_k, pose_ref, mount):
    """Points of one scan, each moved into the LiDAR frame at the reference time.

    ranges, angles: (N,) the beams as measured; poses_k: car poses (x, y, psi arrays) at each beam's
    moment; pose_ref: the car pose at the reference time; mount: the LiDAR's pose on the car.
    p_ref = L(t_ref)^-1 * L(t_k) * p_k  with  L(t) = B(t) * mount  (the LiDAR moves on a circle when
    the car turns, so the mount offset matters: 0.275 m x 0.3 rad = 8 cm). Returns (qx, qy)."""
    ranges = np.asarray(ranges, float)
    ranges = np.where(np.isfinite(ranges), ranges, 0.0)    # no return: computed harmlessly, masked later
    angles = np.asarray(angles, float)
    mx, my, mp = mount
    ones = np.ones_like(ranges)
    l_k = compose(poses_k, (mx * ones, my * ones, mp * ones))
    l_ref = compose(tuple(float(v) for v in pose_ref), (mx, my, mp))
    rx, ry, rp = compose(inverse(l_ref), l_k)       # the LiDAR at t_k, seen from the LiDAR at t_ref
    px, py = ranges * np.cos(angles), ranges * np.sin(angles)
    c, s = np.cos(rp), np.sin(rp)
    return rx + c * px - s * py, ry + s * px + c * py


def rebin(qx, qy, use, angle_min, angle_inc, n):
    """Put points back into the scan's angle bins: range = |q|, bin = the nearest beam angle.

    Two points in one bin -> the nearer one wins (it hides the other); bins without a point -> inf
    (= no return, which AMCL skips). A full-circle scan wraps around (the 'behind' direction)."""
    out = np.full(n, np.inf)
    if not np.any(use):
        return out
    r = np.hypot(qx[use], qy[use])
    a = np.arctan2(qy[use], qx[use])
    b = np.rint((a - angle_min) / angle_inc).astype(int)
    n_circle = int(round(2.0 * np.pi / angle_inc))
    if n_circle <= n:                                   # full circle: wrap
        b = np.mod(b, n_circle)
    keep = (b >= 0) & (b < n)
    np.minimum.at(out, b[keep], r[keep])
    return out


def resample(qx, qy, use, angle_min, angle_inc, n, edge_jump=0.15, max_span=3):
    """Put de-skewed points back into the scan's angle bins WITHOUT holes ("connect the dots").

    After the de-skew the points are no longer exactly one bin apart (the scan was stretched or
    squeezed by the motion), so rebin() alone leaves 12-20 % of the bins empty at 2-6 m/s (offline,
    closed1). Here two CONSECUTIVE beams j, j+1 (neighbours in the measurement) are joined by a straight
    line, and every bin direction between them gets the range interpolated along that line. No line
    across an object edge (range jump > max(edge_jump, 5 % of the range)) or across more than max_span
    bins. Overlapping lines -> the nearer one wins (it hides the other). Bins no line covers fall back
    to rebin() (the nearest point), the rest stays inf."""
    r = np.hypot(qx, qy)
    a = np.arctan2(qy, qx)
    j = np.arange(len(r) - 1)
    close = np.abs(r[j] - r[j + 1]) < np.maximum(edge_jump, 0.05 * np.minimum(r[j], r[j + 1]))
    ok = use[j] & use[j + 1] & close
    a0, r0, r1 = a[j][ok], r[j][ok], r[j + 1][ok]
    d = np.angle(np.exp(1j * (a[j + 1][ok] - a0)))     # signed angle from point j to j+1 (no +-pi jump)
    keep = np.abs(d) <= max_span * angle_inc
    a0, d, r0, r1 = a0[keep], d[keep], r0[keep], r1[keep]
    lo, hi = np.minimum(a0, a0 + d), np.maximum(a0, a0 + d)
    b_lo = np.ceil((lo - angle_min) / angle_inc - 1e-9).astype(int)
    b_hi = np.floor((hi - angle_min) / angle_inc + 1e-9).astype(int)
    cnt = np.maximum(b_hi - b_lo + 1, 0)                 # bins each line covers (usually 0-2)
    seg = np.repeat(np.arange(len(a0)), cnt)
    b = np.repeat(b_lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    ds = d[seg]
    nz = np.abs(ds) > 1e-12
    w = np.where(nz, (angle_min + b * angle_inc - a0[seg]) / np.where(nz, ds, 1.0), 0.0)
    rv = r0[seg] + np.clip(w, 0.0, 1.0) * (r1[seg] - r0[seg])
    n_circle = int(round(2.0 * np.pi / angle_inc))
    if n_circle <= n:
        b = np.mod(b, n_circle)
    inside = (b >= 0) & (b < n)
    out = np.full(n, np.inf)
    np.minimum.at(out, b[inside], rv[inside])
    near = rebin(qx, qy, use, angle_min, angle_inc, n)
    return np.where(np.isfinite(out), out, near)
