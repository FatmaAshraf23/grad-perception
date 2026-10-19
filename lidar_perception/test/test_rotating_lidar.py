#!/usr/bin/env python3
"""Tests for lidar_perception/rotating_lidar.py (rotating-LiDAR timing + de-skew), 2026-10-19 (task 3b).
Run: python3 test_rotating_lidar.py   (or pytest)"""
import math
import os
import sys

import numpy as np

try:
    from lidar_perception.rotating_lidar import (assemble, beam_offsets, compose, deskew, interp_poses,
                                                 rebin, resample)
except ImportError:                                   # run from the source tree without ROS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lidar_perception'))
    from rotating_lidar import assemble, beam_offsets, compose, deskew, interp_poses, rebin, resample

np.seterr(invalid='ignore')                          # beams without a wall carry inf ranges in these tests
N = 360
ANG = np.linspace(-math.pi, math.pi, N)              # the simulator's grid (-180 .. 180 deg, 360 beams)
T = 1.0 / 7.0                                        # one turn
MOUNT = (0.275, 0.0, 0.0)


def wall_ahead_scan(v, wall_x=5.0, order='reverse'):
    """car driving +x at v; a wall across the road at x = wall_x; a real rotating scan (rplidar order)"""
    t = beam_offsets(N, T / (N - 1), order)
    xl = v * t + MOUNT[0]                            # the LiDAR's x when each beam is measured
    fwd = np.abs(ANG) < 1.2
    rho = np.where(fwd, (wall_x - xl) / np.cos(ANG), np.inf)
    return t, rho, fwd


def test_reverse_order_is_rplidar_ros():
    """rplidar_ros fills ranges[N-1-i] = node i: the last element (angle +pi, behind) is measured first"""
    off = beam_offsets(N, T / (N - 1), 'reverse')
    assert off[-1] == 0.0 and abs(off[0] - T) < 1e-12
    assert np.all(np.diff(off) < 0)
    fwd = beam_offsets(N, T / (N - 1), 'forward')
    assert fwd[0] == 0.0 and abs(fwd[-1] - T) < 1e-12


def test_straight_driving_wall_ahead_is_restored():
    """6 m/s: the raw scan puts the wall up to 0.6 m too close (the front is measured mid-turn);
    de-skew -> exact; the WRONG time order is no fix (still ~0.3 m off)"""
    v = 6.0
    t, rho, fwd = wall_ahead_scan(v)
    t_ref = t.max()
    truth_x = 5.0 - (v * t_ref + MOUNT[0])           # the wall seen from the LiDAR at t_ref
    poses = (v * t, np.zeros(N), np.zeros(N))
    qx, qy = deskew(rho, ANG, poses, (v * t_ref, 0.0, 0.0), MOUNT)
    assert np.max(np.abs(qx[fwd] - truth_x)) < 1e-9
    naive = np.max(np.abs(rho[fwd] * np.cos(ANG[fwd]) - truth_x))
    assert 0.5 < naive <= v * T + 1e-9, naive
    t_wrong = beam_offsets(N, T / (N - 1), 'forward')
    qx_w, _ = deskew(rho, ANG, (v * t_wrong, np.zeros(N), np.zeros(N)), (v * t_wrong.max(), 0.0, 0.0), MOUNT)
    wrong = np.max(np.abs(qx_w[fwd] - (5.0 - (v * t_wrong.max() + MOUNT[0]))))
    assert wrong > 0.25, wrong


def test_turning_on_the_spot_wall_is_straight_again():
    """car turning on the spot at 2 rad/s (16 deg per turn), wall y = 2 m: raw scan bends the wall by
    tens of cm; de-skewed points all lie on y = 2 (the LiDAR mount circles around the car: included)"""
    om = 2.0
    t = beam_offsets(N, T / (N - 1), 'reverse')
    psi = om * t
    lx, ly, lp = compose((np.zeros(N), np.zeros(N), psi), (MOUNT[0] * np.ones(N), np.zeros(N), np.zeros(N)))
    d = lp + ANG
    hit = np.sin(d) > 0.3
    rho = np.where(hit, (2.0 - ly) / np.where(hit, np.sin(d), 1.0), np.inf)
    t_ref = t.max()
    pose_ref = (0.0, 0.0, om * t_ref)
    qx, qy = deskew(rho, ANG, (np.zeros(N), np.zeros(N), psi), pose_ref, MOUNT)
    wx, wy, _ = compose(compose(pose_ref, MOUNT), (qx, qy, np.zeros(N)))      # back to the world
    assert np.max(np.abs(wy[hit] - 2.0)) < 1e-9
    nx, ny, _ = compose(compose(pose_ref, MOUNT),
                        (rho * np.cos(ANG), rho * np.sin(ANG), np.zeros(N)))   # raw, as AMCL would read it
    assert np.max(np.abs(ny[hit] - 2.0)) > 0.3


def test_wrong_time_order_tears_the_seam_twice_as_far():
    """THE REAL-CAR SPIN TEST, in numbers: turning on the spot (2 rad/s) inside a round wall (r = 3 m).
    The first and the last beam of a turn both look backwards. Correct model: they agree (error 0).
    No de-skew: they disagree by about omega*T*3 m. Wrong time order: twice that -> the seam behind
    the car tears open -- easy to see in RViz while turning the car by hand."""
    om = 2.0

    def scan_and_errors(order_used):
        t = beam_offsets(N, T / (N - 1), 'reverse')                     # the truth: rplidar order
        psi = om * t
        lx, ly, lp = compose((np.zeros(N), np.zeros(N), psi), (MOUNT[0] * np.ones(N), np.zeros(N), np.zeros(N)))
        ux, uy = np.cos(lp + ANG), np.sin(lp + ANG)
        b = lx * ux + ly * uy
        rho = -b + np.sqrt(b * b - (lx * lx + ly * ly - 9.0))
        wx, wy = lx + rho * ux, ly + rho * uy                              # where the beams really hit
        tu = beam_offsets(N, T / (N - 1), order_used) if order_used else np.full(N, t.max())
        ref = (0.0, 0.0, om * tu.max())
        qx, qy = deskew(rho, ANG, (np.zeros(N), np.zeros(N), om * tu), ref, MOUNT)
        ex, ey, _ = compose(compose(ref, MOUNT), (qx, qy, np.zeros(N)))
        # the reference pose is the TRUE pose at the assumed reference time: compare in the frame that
        # time defines -> errors of the first (element N-1) and last (element 0) measured beams
        e_first = np.hypot(ex[-1] - wx[-1], ey[-1] - wy[-1])
        e_last = np.hypot(ex[0] - wx[0], ey[0] - wy[0])
        return e_first, e_last, np.hypot((ex[-1] - wx[-1]) - (ex[0] - wx[0]), (ey[-1] - wy[-1]) - (ey[0] - wy[0]))

    _, _, tear_ok = scan_and_errors('reverse')
    _, _, tear_none = scan_and_errors(None)
    _, _, tear_wrong = scan_and_errors('forward')
    assert tear_ok < 1e-9, tear_ok
    assert tear_none > 0.5, tear_none
    assert abs(tear_wrong / tear_none - 2.0) < 0.2, (tear_wrong, tear_none)


def test_rebin_nearest_wins_empty_is_inf_and_wraps():
    inc = ANG[1] - ANG[0]
    qx = np.array([1.0, 2.0, 3.0, -1.0])
    qy = np.array([0.0, 0.0, 0.0, -1e-4])            # three points at 0 deg, one just past -180 deg
    out = rebin(qx, qy, np.ones(4, bool), -math.pi, inc, N)
    j0 = int(round(math.pi / inc))
    assert abs(out[j0] - 1.0) < 1e-12                # the nearest of the three at 0 deg
    assert np.isinf(out[j0 + 5])
    assert np.isfinite(out[0]) or np.isfinite(out[N - 1])   # -180 deg landed on the wrap-around bin


def test_resample_fills_the_holes_rebin_leaves():
    """a wall 2 m to the left; the de-skew stretched the points' angles by 15 % (fast driving): nearest-bin
    placement leaves holes, connecting the dots leaves none and gives the right range in every bin"""
    inc = ANG[1] - ANG[0]
    sel = (ANG > math.radians(40)) & (ANG < math.radians(120))
    a = np.where(sel, math.radians(40) + 1.15 * (ANG - math.radians(40)), 0.0)   # stretched directions
    r = np.where(sel, 2.0 / np.where(sel, np.sin(a), 1.0), np.inf)
    qx, qy = r * np.cos(a), r * np.sin(a)
    covered = (ANG >= math.radians(41)) & (ANG <= math.radians(40) + 1.15 * math.radians(79))
    holes_rebin = np.sum(~np.isfinite(rebin(qx, qy, sel, -math.pi, inc, N)[covered]))
    out = resample(qx, qy, sel, -math.pi, inc, N)
    assert holes_rebin >= 5, holes_rebin
    assert np.all(np.isfinite(out[covered]))
    assert np.max(np.abs(out[covered] - 2.0 / np.sin(ANG[covered]))) < 2e-3


def test_resample_does_not_bridge_edges_and_nearer_wins():
    """a box 2 m ahead (-5..5 deg) in front of a wall 5 m ahead: no fake points between them"""
    inc = ANG[1] - ANG[0]
    deg = np.degrees(ANG)
    sel = np.abs(deg) < 30
    r = np.where(np.abs(deg) < 5, 2.0, 5.0)
    r = np.where(sel, r, np.inf)
    out = resample(r * np.cos(ANG), r * np.sin(ANG), sel, -math.pi, inc, N)
    fin = out[np.isfinite(out)]
    assert not np.any((fin > 2.2) & (fin < 4.8)), fin[(fin > 2.2) & (fin < 4.8)]
    assert abs(out[np.argmin(np.abs(deg - 2))] - 2.0) < 1e-9 and abs(out[np.argmin(np.abs(deg - 12))] - 5.0) < 1e-9


def test_assemble_takes_each_beam_from_its_own_moment():
    st = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    sr = np.tile((10.0 + st)[:, None], (1, 4))       # range = 10 m + the state's time
    sr[3:, 2] += 1.0                                 # beam 2 jumps by 1 m between states 2 and 3 (an edge)
    r = assemble(st, sr, np.array([0.025, 0.005, 0.026, 0.04]), edge_jump=0.15)
    assert abs(r[0] - 10.025) < 1e-12 and abs(r[1] - 10.005) < 1e-12 and abs(r[3] - 10.04) < 1e-12
    assert r[2] == sr[3, 2]                          # edge: the nearer state (0.026 is closer to 0.03), no blend


def test_odometry_interpolation_and_extrapolation():
    to = np.arange(0.0, 0.5, 0.01)
    v, w = 3.0, 1.0
    psi = w * to
    x = np.cumsum(np.r_[0.0, v * np.cos(0.5 * (psi[1:] + psi[:-1])) * 0.01])
    y = np.cumsum(np.r_[0.0, v * np.sin(0.5 * (psi[1:] + psi[:-1])) * 0.01])
    tq = np.array([0.105, to[-1] + 0.02, to[-1] + 0.2, -0.1])
    xi, yi, pi, ok = interp_poses(to, x, y, psi, v * np.ones_like(to), w * np.ones_like(to), tq, 0.05)
    assert list(ok) == [True, True, False, False]
    assert abs(pi[1] - w * (to[-1] + 0.02)) < 1e-9
    ex = x[-1] + v * math.cos(psi[-1] + 0.01 * w) * 0.02
    assert abs(xi[1] - ex) < 1e-12


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS  {name}')
        except Exception as ex:                       # noqa: BLE001
            fails += 1
            print(f'FAIL  {name}: {type(ex).__name__}: {ex}')
    print(f'\n{len(tests) - fails} of {len(tests)} passed')
    sys.exit(1 if fails else 0)
