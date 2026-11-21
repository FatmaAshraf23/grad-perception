#!/usr/bin/env python3
"""Tests for the phantom fix in lidar_perception/obstacle_list.py (2026-11-19): part a (freeze_on_refused: a scan whose
alignment is refused as 'correction too large' does not update the list) and part b (check_pose / pose_check: after
an accepted alignment, a scan is not used when a clearly different pose fits the walls clearly better).
World: a synthetic corridor (5 cm cells, 1.6 m lane, 15 cm walls, unknown behind) -- a 20 m straight along x that
turns left at its end. Scans are ray-cast from a TRUE pose; StateEstimate's pose is given an error on purpose.
Run: python3 test_phantom_fix.py   (or pytest)"""
import math
import os
import sys

import numpy as np

try:
    from lidar_perception import obstacle_list as ol
    from lidar_perception.map_difference import objects
    from lidar_perception.obstacle_tracker import ObstacleTracker
    from lidar_perception.scan_align import ScanAligner
    from lidar_perception.wall_distance import WallDistance
except ImportError:                                   # run from the source tree without ROS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lidar_perception'))
    import obstacle_list as ol
    from map_difference import objects
    from obstacle_tracker import ObstacleTracker
    from scan_align import ScanAligner
    from wall_distance import WallDistance

np.seterr(invalid='ignore')                           # beams without a wall carry inf ranges
RES = 0.05
MOUNT = (0.275, 0.0, 0.0)
N = 360                                               # the simulator's A1: 360 beams, -180 .. 180 deg
ANG_MIN, ANG_INC = -math.pi, 2.0 * math.pi / N
R_MIN, R_MAX = 0.15, 12.0
CORNER = (16.5, 1.3, 0.0)        # 3.7 m before the end wall, the left turn in view: walls ACROSS the heading
STRAIGHT = (10.0, 1.3, 0.0)      # both ends of the straight > 8 m away: only walls PARALLEL to the heading
T_SCAN = 10.02


def corridor():
    """OccupancyGrid values (row 0 = bottom): lane y 0.5..2.1 for x 0.5..20.5, then x 18.9..20.5 up to y 7.5."""
    w, h = int(round(21.0 / RES)), int(round(8.0 / RES))
    X, Y = np.meshgrid((np.arange(w) + 0.5) * RES, (np.arange(h) + 0.5) * RES)
    free = (((X > 0.5) & (X < 20.5) & (Y > 0.5) & (Y < 2.1))
            | ((X > 18.9) & (X < 20.5) & (Y > 0.5) & (Y < 7.5)))
    wall = np.zeros_like(free)
    for di in range(-3, 4):                           # 3 cells = 15 cm of wall around the lane
        for dj in range(-3, 4):
            wall |= np.roll(np.roll(free, di, axis=0), dj, axis=1)
    grid = np.full((h, w), -1, dtype=np.int16)
    grid[wall] = 100
    grid[free] = 0
    return grid


GRID = corridor()
WD = WallDistance(GRID, RES, 0.0, 0.0)
ALIGNER = ScanAligner(WD, GRID)


def scan_from(pose):
    """Ray-cast scan from the TRUE pose: range to the first wall cell (5 mm steps), inf = no return."""
    x, y, yaw = pose
    lx, ly = x + MOUNT[0] * math.cos(yaw), y + MOUNT[0] * math.sin(yaw)
    a = yaw + MOUNT[2] + ANG_MIN + np.arange(N) * ANG_INC
    d = np.arange(0.005, R_MAX, 0.005)
    col = np.floor((lx + np.cos(a)[:, None] * d) / RES).astype(int)
    row = np.floor((ly + np.sin(a)[:, None] * d) / RES).astype(int)
    h, w = GRID.shape
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    hit = np.zeros(col.shape, dtype=bool)
    hit[inside] = GRID[row[inside], col[inside]] >= 65
    return np.where(hit.any(axis=1), d[hit.argmax(axis=1)], np.inf)


def moved(pose, along=0.0, yaw_deg=0.0):
    x, y, yaw = pose
    return x + along * math.cos(yaw), y + along * math.sin(yaw), yaw + math.radians(yaw_deg)


def check_at(true_pose, est_pose):
    return ol.pose_check(ALIGNER.field, scan_from(true_pose), ANG_MIN, ANG_INC, R_MIN, R_MAX, est_pose, MOUNT)


def run_list(true_pose, est_pose, aligner=ALIGNER, **kw):
    """One scan through ObstacleList; StateEstimate stands at est_pose (mode OK, std 5 cm / 0.5 deg)."""
    b = ol.ObstacleList(WD, MOUNT, aligner, **kw)
    for k in range(5):
        b.add_pose(10.0 + 0.01 * k, *est_pose, 0.0, 0.0, True, std_xy=0.05, std_psi=math.radians(0.5))
    return b.process_scan(T_SCAN, scan_from(true_pose), ANG_MIN, ANG_INC, R_MIN, R_MAX)


# ---------------------------------------------------------------------------------------------- tests
def test_limits_sit_between_the_grid_steps():
    """a grid value exactly ON a limit would be decided by rounding luck (0.05 * 1.5 = 0.07500000000000001)"""
    assert ol.CHECK_ALONG[len(ol.CHECK_ALONG) // 2] == 0.0 and ol.CHECK_YAW[len(ol.CHECK_YAW) // 2] == 0.0
    assert np.min(np.abs(np.abs(ol.CHECK_ALONG) - ol.CHECK_MIN_ALONG)) > 0.005
    assert np.min(np.abs(np.abs(ol.CHECK_YAW) - ol.CHECK_MIN_YAW)) > math.radians(0.05)


def test_pose_check_quiet_at_the_true_pose():
    for pose in (CORNER, STRAIGHT):
        c = check_at(pose, pose)
        assert not c['suspect'] and c['gain_along'] <= 0.005 and c['gain_yaw'] <= 0.005, (pose, c)


def test_pose_check_sees_an_along_error_where_walls_cross_the_heading():
    for err in (0.15, 0.30, -0.30, 0.50):
        c = check_at(CORNER, moved(CORNER, along=err))
        assert c['suspect'] and c['gain_along'] >= ol.CHECK_GAIN, (err, c)
        assert abs(c['best_along'] + err) <= 0.041, (err, c)          # the slide that undoes the error


def test_pose_check_counts_only_wall_faces_that_face_the_lidar():
    """15 cm ahead pushes the end wall's points 15 cm INTO the 15 cm thick wall = onto its far side (unknown behind).
    Counting any wall surface, that looked like a perfect fit (gain 0.003, not suspect)."""
    c = check_at(CORNER, moved(CORNER, along=0.15))
    assert c['suspect'] and c['gain_along'] > 0.05, c


def test_pose_check_sees_a_heading_error():
    for deg in (3.0, -3.0):
        c = check_at(CORNER, moved(CORNER, yaw_deg=deg))
        assert c['suspect'] and abs(c['best_yaw'] + math.radians(deg)) <= math.radians(0.3), (deg, c)


def test_pose_check_is_blind_to_an_along_error_on_a_long_straight():
    """known limit: walls PARALLEL to the error cannot show it -- that is StateEstimate's job"""
    c = check_at(STRAIGHT, moved(STRAIGHT, along=0.30))
    assert not c['suspect'] and c['gain_along'] == 0.0, c


def test_check_pose_freezes_the_scan_the_alignment_accepted_wrongly():
    est = moved(CORNER, along=0.30)
    before = run_list(CORNER, est)                                     # the code before the fix
    assert before['status'] == ol.STATUS_OK and before['aligned'] and before['n_objects'] >= 1, before['status']
    assert before['correction'][0] < 0.05                              # the alignment ACCEPTED a ~2 cm correction
    after = run_list(CORNER, est, check_pose=True)
    assert after['status'] == ol.STATUS_FROZEN_SCAN and after['check']['suspect'], after['status']


def test_check_pose_leaves_good_scans_alone():
    r = run_list(CORNER, CORNER, check_pose=True, freeze_on_refused=True)
    assert r['status'] == ol.STATUS_OK and not r['check']['suspect'] and r['n_objects'] == 0
    r = run_list(CORNER, moved(CORNER, yaw_deg=3.0), check_pose=True)  # the alignment fixes 3 deg here by itself
    assert r['status'] == ol.STATUS_OK and r['aligned'] and not r['check']['suspect'] and r['n_objects'] == 0


class _Refusing:
    """an aligner that always refuses with the given reason"""
    def __init__(self, reason):
        self.reason = reason

    def align(self, ranges, angle_min, angle_inc, range_min, range_max, pose, pose_std, mount):
        return dict(ok=False, pose=pose, std=pose_std, n=200, shift=0.3, turn=0.0, reason=self.reason)


def test_freeze_on_refused_only_for_a_too_large_correction_and_only_when_on():
    tl = run_list(CORNER, CORNER, aligner=_Refusing('correction too large'))
    assert tl['status'] == ol.STATUS_OK and tl['refused'] == 'correction too large'     # before the fix
    tl = run_list(CORNER, CORNER, aligner=_Refusing('correction too large'), freeze_on_refused=True)
    assert tl['status'] == ol.STATUS_FROZEN_SCAN
    few = run_list(CORNER, CORNER, aligner=_Refusing('few wall points'), freeze_on_refused=True)
    assert few['status'] == ol.STATUS_OK and few['refused'] == 'few wall points'      # proves nothing -> detect


def _one_object(x):
    return objects(np.array([x, x + 0.02, x + 0.04]), np.array([0.5, 0.5, 0.5]), (0.0, 0.0))


def test_tracker_expire_closes_the_frozen_gap():
    """a tentative obstacle seen at t = 0, then 1 s of frozen scans, then seen again: without expire() update()
    matches it first and CONFIRMS it with that single later detection"""
    tr = ObstacleTracker()
    tr.update(0.0, _one_object(2.0))
    tr.update(1.0, _one_object(2.0))
    assert [t.confirmed for t in tr.tracks] == [True]                  # the gap: confirmed across 1 s
    tr = ObstacleTracker()
    tr.update(0.0, _one_object(2.0))
    tr.expire(1.0)                                                     # what a frozen scan now does
    assert tr.tracks == []
    tr.update(1.0, _one_object(2.0))
    assert [t.confirmed for t in tr.tracks] == [False]                 # starts again as a tentative obstacle
    tr.update(1.14, _one_object(2.0))
    assert [t.confirmed for t in tr.tracks] == [True]                  # confirmed by the next scan, as normal


def test_expire_keeps_confirmed_and_recent_obstacles():
    tr = ObstacleTracker()
    tr.update(0.0, _one_object(2.0) + _one_object(4.0))
    tr.update(0.14, _one_object(2.0))                                  # 2.0 confirmed, 4.0 still tentative
    tr.expire(0.4)
    assert sorted((round(t.x, 1), t.confirmed) for t in tr.tracks) == [(2.1, True), (4.1, False)]
    tr.expire(10.0)
    assert [(round(t.x, 1), t.confirmed) for t in tr.tracks] == [(2.1, True)]


if __name__ == '__main__':
    names = [n for n in sorted(globals()) if n.startswith('test_')]
    for n in names:
        globals()[n]()
        print('ok  ', n)
    print(f'{len(names)} tests passed')
