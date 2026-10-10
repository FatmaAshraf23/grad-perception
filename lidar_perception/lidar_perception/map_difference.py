#!/usr/bin/env python3
"""map_difference.py -- obstacle detection, approach B: find what the car's map can't explain.

  scan --(EKF pose)--> points in the map --(far from every wall?)--> new points --(group)--> objects

Step 2.2  scan_to_map(): every valid beam (range r, angle a) becomes a point in the MAP frame, using the
          car's pose at the scan's time (the EKF pose -- the car has nothing better) and where the LiDAR
          sits on the car:  LiDAR position = car position + mount offset turned by the car's heading;
          point = LiDAR position + r * (cos(yaw + a), sin(yaw + a)).
          new_points(): "new" = on the track surface (a free cell, not unknown space behind a wall) AND
          farther than TOL from every wall of the car's map (WallDistance.at(): one lookup per point).
Step 2.3  cluster(): new points closer than GAP to each other belong to the same object.
          objects(): every group with >= MIN_POINTS points -> centre, size, number of points.
detect() runs the whole chain for one scan. numpy only.
"""
import math

import numpy as np

TOL = 0.10          # m: a point counts as new if it is farther than this from every wall (step 2.2:
                    #    5 cm -> 1.7 false points per scan, 10 cm -> 0.002, cones kept 99-100 %)
MAX_RANGE = 8.0     # m: farther points are not used (see scan_to_map)
GAP = 0.25          # m: new points closer than this are one object. Points on a 15 cm cone are <= 0.21 m
                    #    apart; neighbouring 1-degree beams are 0.14 m apart at 8 m. Two cones closer
                    #    than 25 cm become one object (planning must avoid both anyway).
MIN_POINTS = 2      # one point alone could be noise and has no size
CONE_RADIUS = 0.075 # m: the LiDAR sees only the side facing it -> the centre is ~one radius behind the
                    #    points. Sim cone: 15 x 15 cm. Real cone: half its width AT THE LIDAR'S HEIGHT.


def lidar_position(pose, mount):
    """(x, y, yaw) of the LiDAR in the map, from the car's pose and the LiDAR's mount on the car."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * mount[0] - s * mount[1], y + s * mount[0] + c * mount[1], yaw + mount[2]


def scan_to_map(ranges, angle_min, angle_inc, range_min, range_max, pose, mount=(0.275, 0.0, 0.0),
                max_range=MAX_RANGE):
    """LiDAR scan -> map-frame points.
    pose  = (x, y, yaw) of base_link in the map at the scan's time.
    mount = (x, y, yaw) of the LiDAR on the car (base_link -> laser).
    Returns px, py (map frame [m]), beam index and range of every valid point. Valid = a real
    return between range_min and range_max, and not farther than max_range (None = no limit).
    Why max_range (default 8 m):
      - a beam that hits NOTHING comes back as range_max (the simulator: 12 m + noise, so 11.94..12.0);
        such a "point" lands in the middle of a far-away lane and looks like an obstacle (job 078: 291
        such ghost points over 2 laps, all kept as "new")
      - the heading error moves a point by distance x error: far points are the least trustworthy
      - a 15 cm cone gives >= 2 points only up to ~5-7 m anyway (step 1)"""
    r = np.asarray(ranges, dtype=float)
    beam = np.arange(len(r))
    ok = np.isfinite(r) & (r >= range_min) & (r < range_max - 1e-3)
    if max_range is not None:
        ok &= r <= max_range
    lx, ly, lyaw = lidar_position(pose, mount)
    a = lyaw + angle_min + beam[ok] * angle_inc
    return lx + r[ok] * np.cos(a), ly + r[ok] * np.sin(a), beam[ok], r[ok]


def new_points(wall_distance, px, py, tol=TOL):
    """True for every point the map can't explain: on the track surface and > tol from every wall."""
    return wall_distance.on_free(px, py) & (wall_distance.at(px, py) > tol)


def cluster(px, py, gap=GAP):
    """Group points: two points closer than `gap` are in the same group, and so is a chain of such
    neighbours. Returns a group number for every point (0, 1, 2, ...)."""
    px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
    n = len(px)
    group = np.full(n, -1, dtype=int)
    close = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :]) < gap   # n x n yes/no table
    g = 0
    for i in range(n):
        if group[i] >= 0:
            continue
        group[i] = g
        todo = [i]
        while todo:                                  # spread the number to every point linked to point i
            j = todo.pop()
            for k in np.flatnonzero(close[j] & (group < 0)):
                group[k] = g
                todo.append(k)
        g += 1
    return group


def objects(px, py, lidar_xy, gap=GAP, min_points=MIN_POINTS, radius=CONE_RADIUS):
    """New points -> objects (list of dicts): x, y = estimated centre (the points' average pushed back by
    `radius` along the line from the LiDAR), x_raw, y_raw = the points' average, n = number of points,
    size = largest distance between two of its points, range = distance from the LiDAR, idx = its points."""
    px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
    group = cluster(px, py, gap)
    lx, ly = lidar_xy
    out = []
    for g in range(int(group.max()) + 1 if len(group) else 0):
        idx = np.flatnonzero(group == g)
        if len(idx) < min_points:
            continue
        cx, cy = float(px[idx].mean()), float(py[idx].mean())
        dx, dy = cx - lx, cy - ly
        dist = math.hypot(dx, dy)
        size = float(np.max(np.hypot(px[idx][:, None] - px[idx][None, :], py[idx][:, None] - py[idx][None, :])))
        out.append(dict(x=cx + dx / dist * radius, y=cy + dy / dist * radius, x_raw=cx, y_raw=cy,
                        n=len(idx), size=size, range=dist, idx=idx))
    return out


def detect(ranges, angle_min, angle_inc, range_min, range_max, pose, mount, wall_distance, tol=TOL,
           max_range=MAX_RANGE, gap=GAP, min_points=MIN_POINTS, radius=CONE_RADIUS):
    """The whole chain for one scan -> (objects, px, py, beam of the new points)."""
    px, py, beam, _ = scan_to_map(ranges, angle_min, angle_inc, range_min, range_max, pose, mount, max_range)
    keep = new_points(wall_distance, px, py, tol)
    lx, ly, _ = lidar_position(pose, mount)
    return objects(px[keep], py[keep], (lx, ly), gap, min_points, radius), px[keep], py[keep], beam[keep]
