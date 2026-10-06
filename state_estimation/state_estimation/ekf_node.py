#!/usr/bin/env python3
"""EKF version 3: wheel speed + gyro, corrected by AMCL (LiDAR vs wall map).

The track is bounded by walls (no cones), so the LiDAR correction now comes
from AMCL: a particle filter that matches each scan against the wall map and
reports a pose (x, y, yaw) with a covariance.

TWO INSTANCES OF THIS NODE RUN AT THE SAME TIME (parameter `mode`)

  mode = 'odom'  -> the LOCAL EKF ("ekf_local")
      Wheel + gyro + ZUPT only. Never jumps, drifts slowly.
      Publishes TF  odom -> est/base_link  (smooth odometry for AMCL).
  mode = 'map'   -> the GLOBAL EKF ("ekf_global")
      Same prediction, plus the AMCL pose update. Drift-free.
      Publishes /odom/ekf in the map frame: the output for planning and MPC.

  Why two? AMCL uses odometry (odom -> base_link) to move its particles. If
  that odometry already contained AMCL's own corrections, AMCL would be fed
  its own output (a feedback loop that makes errors look smaller than they
  are). So AMCL gets the pure odometry of the local EKF, and the global EKF
  fuses AMCL on top. This is the standard ROS layout (REP-105):
        map --(AMCL)--> odom --(ekf_local)--> est/base_link --> est/laser

STATE (both modes)
    x = [ px, py, yaw, b, k ]
        px, py, yaw : pose in the node's frame (map or odom)
        b           : gyro bias                 [rad/s]
        k           : wheel-speed scale, v = k * v_meas
                      (only observable in the global EKF, via AMCL)

PREDICT (100 Hz, every IMU message) -- unchanged from v2
    v = k*v_meas,  w = w_meas - b
    px += v cos(yaw+beta) dt,  py += v sin(yaw+beta) dt,  yaw += w dt
    P  = F P F^T + G Qu G^T + Q_b + Q_k

UPDATE 1 -- ZUPT while stopped (both modes) -- unchanged

UPDATE 2 -- AMCL pose (global EKF only)                       NEW in v3
    z = [x, y, yaw] from /amcl_pose,   h(x) = [px, py, yaw],
    H = [ I3 | 0 ]   (AMCL measures the first three states directly)
    R = AMCL's own 3x3 covariance * amcl_cov_scale, with a minimum
        (inflated because AMCL and this EKF share the same odometry, so their
         errors are correlated -- treating them as independent would make the
         EKF overconfident)
        amcl_cov_scale 4 (2026-10-06, was 2): with AMCL's motion noise lowered
        to 0.02 (amcl_sim.yaml) AMCL reports a much smaller covariance, but its
        errors still change slowly (~7 fixes/s that share one error), so the EKF
        averaged them as if independent and became overconfident (run H: only
        80-93 % of position errors inside 2 sigma). Offline replay: scale 4 ->
        98-99 % inside 2 sigma, s error p95 4.8 -> 4.2 cm.
    HALF-CELL CORRECTION (2026-10-06): z uses AMCL's x and y + amcl_offset_m.
        nav2 AMCL puts the CENTRE of map cell i at  origin + i * resolution
        (nav2_amcl map.hpp MAP_GXWX, amcl_node.cpp convertMap), but ROS -- the
        map_server / RViz convention, the simulator, our apex_track track files --
        puts it at  origin + (i + 0.5) * resolution. So AMCL sees every wall half
        a cell towards -x and -y, and its pose comes out shifted by
        (-res/2, -res/2): measured (-2.9, -2.1) cm on closed1 and (-3.1, -2.4) cm
        on Levine (5 cm maps). On the track that is an along-track error whose
        sign depends on the driving direction (behind when driving +x, ahead
        when driving -x), and in a corner it becomes kappa * ds of heading
        error. Offline replay: s error p95 5.9 -> 3.3 cm; run K re-projected:
        corner e_psi p95 2.48 -> 1.03 deg, whole lap 1.88 -> 0.66 deg.
        amcl_offset_m must be HALF THE RESOLUTION OF THE MAP AMCL USES (0.025 for
        a 5 cm map, 0.0125 for 2.5 cm); the node compares it with /map once at
        start-up and warns if it doesn't match. 0 = no correction.
        Not corrected: AMCL's own /amcl_pose and its TF map -> odom.
    Gate: NIS < 11.34 (chi-square, 3 degrees of freedom, 99 %)

    DELAYED MEASUREMENT: the AMCL pose belongs to the moment the scan was
    taken (~0.1-0.2 s ago); the car has moved since. So the innovation is
    computed against the EKF state stored for THAT moment:
        nu = z - x_history(t_scan)
    and the correction K*nu is added to the current state (and to the stored
    history, so the next measurement doesn't correct the same error twice).

    SPEED-SCALE CALIBRATION FILE                                 NEW 2026-10-01
    On a straight corridor AMCL cannot see along-track error, so k is only
    learned at the first corner. Starting from k = 1 with a 3 % wheel error
    gave ~20 cm along-track error after 8 m in every evaluation run. Now:
      - at start-up BOTH EKFs read k from `scale_file` (if it exists) and
        start with sigma_k = max(saved sigma, scale_file_min_sigma).
        `scale_file` is the ANCHOR: written only by `wheel_calib` (tape-
        measured straight drive) or by hand -- NEVER by this node;
      - the GLOBAL EKF writes the k it learned to `scale_learned_file`
        (every 10 s after `scale_save_min_distance` m, and at shutdown) and
        warns if it differs from the anchor by more than `scale_warn` (1 %)
        -> on the real car: tyres changed/worn, redo wheel_calib.
    Why not write the learned k back into scale_file? Tried 2026-10-01: it is
    a feedback loop (file -> local EKF -> AMCL odometry -> AMCL pose on
    straights -> global EKF -> file). In 3 runs k slid 0.9706 -> 0.9689 ->
    0.9683 -> 0.9679 (true 0.9709) while every uncalibrated run ended within
    0.05 % of the truth.

    STEERING CALIBRATION (2026-10-03): beta uses the steering corrected with the
    file of steer_calib: delta = gain * (delta_measured - offset) (calibration.py,
    parameter steer_calib_file; no file = no correction). A +0.01 rad steering
    offset made beta 0.30 deg too large and the heading 0.30 deg too low.

    RECOVERY = JUMP TO AMCL, ONLY WHEN AMCL IS CLEARLY RIGHT (2026-10-06).
    Before: after `reset_after_rejects` (10) AMCL poses rejected in a row the EKF
    assumed IT was lost and jumped to AMCL. When AMCL was the one that was lost
    (closed1 with a solid-black map, jobs 068-070) the EKF followed it into the
    walls, 10-20 m off, and reported OK. Now:
      1. AMCL poses with sqrt(cov_xx + cov_yy) > amcl_max_sigma (0.35 m) are
         ignored -- AMCL itself says it is unsure (lost AMCL: > 0.2 m 99.9 % of
         the time; normal laps: max 0.29 m).
      2. After `reset_after_rejects` confident AMCL poses rejected in a row the
         EKF jumps only if BOTH checks pass, otherwise it keeps going on odometry
         (StateEstimate reports LOST after 1 s of driving without a fix):
         a. AMCL MOVED LIKE THE ODOMETRY over those poses (motion_consistent:
            first pose to every later one, each in its own frame, within
            reset_tol_pos + reset_tol_frac * distance and reset_tol_yaw_deg);
         b. THE LIDAR SCAN FITS THE MAP CLEARLY BETTER AT AMCL'S POSE than at
            the EKF's (scan_fit: share of scan_beams beams whose measured range
            matches the map's within scan_tol; AMCL >= reset_scan_min and
            >= EKF + reset_scan_margin). (a) alone is not enough: a lost AMCL
            moves with the odometry too.
         Otherwise the oldest of the poses is dropped and the next one re-checks.
    Offline (robust.py, recorded laps): AMCL lost (12 solid-map runs): old rule
    52 jumps onto the wrong pose, EKF up to 23 m off -> new rule 0 jumps, EKF
    error p95 0.24 m / max 0.34 m; EKF lost (40 injected faults, AMCL fine):
    recovered in 1.4 s (up to ~6 s when the error lies along a featureless
    corridor, where the scan looks the same); normal laps unchanged.
    The scan check needs the map (map_topic) and the scan AMCL uses (scan_topic),
    and where the LiDAR sits on the car (laser_x, laser_y, laser_yaw).
    test_fault (TEST ONLY, default off): 't_s,dx,dy,dyaw_deg' moves ekf_global's
    pose once, t_s seconds after its first IMU message -- a "kidnapped" EKF while
    AMCL is fine, to test the jump live.

Topics
  in : /imu/data, /vehicle/measured, /amcl_pose (global), /initialpose (global)
       <map_topic> (global, read once: checks amcl_offset_m against the resolution;
                    the walls for the scan check), <scan_topic> (global, scan check)
       <truth_topic> (global, simulation truth for the 2-s report only;
                      default 'none' = off -- the real car has no truth)
  out: <odom_topic>, <path_topic> (only if path_period_s > 0; for viewing),
       <diag_prefix>/bias, /bias_sigma, /speed_scale, /nis ;
       TF odom -> est/base_link (local only)

CPU (2026-10-03): in Python ROS 2 the cost is mostly MESSAGES, not math. The sim
truth arrives at 250 Hz, and the path message carries up to path_max_len poses;
both are off by default so nothing sim-only or view-only runs on the Pi.
state_pipeline.py runs both EKFs + state_estimate in ONE process: one IMU and one
wheel subscription for all three; odom_hooks / fix_hooks hand ekf_global's output
to state_estimate without a message.
"""
import math
import os
import time
from collections import deque

import numpy as np

CHI2_3DOF_99 = 11.34


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# ----------------------------------------------------------------------------
# The filter itself: plain numpy, no ROS -> can be tested offline
# ----------------------------------------------------------------------------

class LocalizationEKF:
    PX, PY, YAW, B, K = range(5)
    N = 5

    def __init__(self, x0, P0_diag, sigma_v, sigma_w, sigma_bw, sigma_kw):
        self.x = np.array(x0, dtype=float)
        self.P = np.diag(np.array(P0_diag, dtype=float) ** 2)
        self.sigma_v = sigma_v      # wheel speed noise              [m/s]
        self.sigma_w = sigma_w      # gyro white noise per sample    [rad/s]
        self.sigma_bw = sigma_bw    # gyro bias random walk          [rad/s/sqrt(s)]
        self.sigma_kw = sigma_kw    # speed-scale random walk        [1/sqrt(s)]

    # --- prediction ----------------------------------------------------------

    def predict(self, v_meas, w_meas, beta, dt):
        px, py, yaw, b, k = self.x
        v = k * v_meas
        heading = yaw + beta
        c, s = math.cos(heading), math.sin(heading)

        self.x[self.PX] = px + v * c * dt
        self.x[self.PY] = py + v * s * dt
        self.x[self.YAW] = wrap(yaw + (w_meas - b) * dt)

        F = np.eye(self.N)
        F[0, 2] = -v * s * dt
        F[1, 2] = v * c * dt
        F[0, 4] = v_meas * c * dt
        F[1, 4] = v_meas * s * dt
        F[2, 3] = -dt
        G = np.zeros((self.N, 2))
        G[0, 0] = k * c * dt
        G[1, 0] = k * s * dt
        G[2, 1] = dt
        Qu = np.diag([self.sigma_v ** 2, self.sigma_w ** 2])
        self.P = F @ self.P @ F.T + G @ Qu @ G.T
        self.P[self.B, self.B] += self.sigma_bw ** 2 * dt
        self.P[self.K, self.K] += self.sigma_kw ** 2 * dt

    def stationary_predict(self, dt):
        self.P[self.B, self.B] += self.sigma_bw ** 2 * dt
        self.P[self.K, self.K] += self.sigma_kw ** 2 * dt

    # --- generic EKF update --------------------------------------------------

    def _update(self, nu, H, R):
        """Standard EKF update. Returns (NIS, correction dx)."""
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        K = self.P @ H.T @ S_inv
        dx = K @ nu
        self.x = self.x + dx
        self.x[self.YAW] = wrap(self.x[self.YAW])
        I_KH = np.eye(self.N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T      # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        return float(nu @ S_inv @ nu), dx

    # --- update 1: zero velocity ---------------------------------------------

    def zupt(self, w_meas):
        H = np.zeros((1, self.N))
        H[0, self.B] = 1.0
        nu = np.array([w_meas - self.x[self.B]])
        R = np.array([[self.sigma_w ** 2]])
        self._update(nu, H, R)

    # --- update 2: pose (from AMCL) ------------------------------------------

    H_POSE = np.hstack([np.eye(3), np.zeros((3, 2))])

    def pose_nis(self, nu, R):
        S = self.H_POSE @ self.P @ self.H_POSE.T + R
        return float(nu @ np.linalg.solve(S, nu))

    def pose_update(self, nu, R):
        return self._update(nu, self.H_POSE, R)

    # --- helpers -------------------------------------------------------------

    def reset_pose(self, px, py, yaw, P3=None):
        """New pose; bias and speed scale are kept (they belong to the sensors)."""
        self.x[:3] = [px, py, yaw]
        self.P[:3, :] = 0.0
        self.P[:, :3] = 0.0
        if P3 is None:
            P3 = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(2.0) ** 2])
        self.P[:3, :3] = P3

    def sigma(self, i):
        return math.sqrt(max(self.P[i, i], 0.0))

    def sigma_pos(self):
        return math.sqrt(max(self.P[0, 0] + self.P[1, 1], 0.0))


class StateHistory:
    """Past states, to compare a delayed measurement with the right moment."""

    def __init__(self, max_age_s):
        self.max_age_ns = int(max_age_s * 1e9)
        self.buf = deque()                     # (t_ns, state copy)

    def add(self, t_ns, x):
        self.buf.append((t_ns, x.copy()))
        while self.buf and t_ns - self.buf[0][0] > self.max_age_ns:
            self.buf.popleft()

    def at(self, t_ns):
        """State closest in time to t_ns, or None if t_ns is outside the buffer."""
        if not self.buf or t_ns < self.buf[0][0] - 20_000_000:
            return None
        return min(self.buf, key=lambda e: abs(e[0] - t_ns))[1]

    def shift(self, dx):
        """Apply a correction to the stored past too (it was wrong by the same amount)."""
        for _, x in self.buf:
            x += dx
            x[2] = wrap(x[2])


def amcl_measurement_noise(cov6x6, scale, min_sigma_pos, min_sigma_yaw):
    """3x3 R (x, y, yaw) from AMCL's 6x6 covariance, inflated and floored."""
    c = np.asarray(cov6x6, dtype=float).reshape(6, 6)
    R = c[np.ix_([0, 1, 5], [0, 1, 5])] * scale
    R[0, 0] = max(R[0, 0], min_sigma_pos ** 2)
    R[1, 1] = max(R[1, 1], min_sigma_pos ** 2)
    R[2, 2] = max(R[2, 2], min_sigma_yaw ** 2)
    return 0.5 * (R + R.T)


# ----------------------------------------------------------------------------
# Jump-to-AMCL check: plain numpy, no ROS -> tested offline (see RECOVERY)
# ----------------------------------------------------------------------------

class WallMap:
    """The walls of the map AMCL uses, for ray casting. ROS cell convention (map_server,
    RViz, the simulator): cell (row, col) spans origin + [col, col + 1) * res in x and
    origin + [row, row + 1) * res in y; row 0 = the BOTTOM row (OccupancyGrid order).
    Unknown cells do not stop a ray (like the simulator)."""

    def __init__(self, occupied, res, ox, oy):
        self.occ = np.asarray(occupied, dtype=bool)
        self.h, self.w = self.occ.shape
        self.res, self.ox, self.oy = float(res), float(ox), float(oy)

    @classmethod
    def from_grid(cls, data, width, height, res, ox, oy, occupied_from=65):
        """OccupancyGrid data: 0 free, 100 wall, -1 unknown."""
        grid = np.asarray(data, dtype=np.int16).reshape(height, width)
        return cls(grid >= occupied_from, res, ox, oy)

    def ranges(self, lx, ly, beam_yaws, max_range):
        """Distance from (lx, ly) to the first wall cell along each beam (half-cell
        steps); inf = no wall within max_range."""
        ts = np.arange(0.5 * self.res, max_range + 1e-9, 0.5 * self.res)
        xs = lx + np.outer(np.cos(beam_yaws), ts)
        ys = ly + np.outer(np.sin(beam_yaws), ts)
        col = np.floor((xs - self.ox) / self.res).astype(np.int64)
        row = np.floor((ys - self.oy) / self.res).astype(np.int64)
        inside = (col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)
        hit = np.zeros(xs.shape, dtype=bool)
        hit[inside] = self.occ[row[inside], col[inside]]
        return np.where(hit.any(axis=1), ts[hit.argmax(axis=1)], np.inf)


def scan_fit(wall_map, pose, laser, beam_angles, r_meas, max_range, tol):
    """Share of the beams the map explains if the car is at `pose` (x, y, yaw of
    base_link): measured and map range within `tol`, or both 'no return' (inf).
    laser = (x, y, yaw) of the LiDAR on the car. 1.0 = the scan fits perfectly."""
    x, y, yaw = pose[0], pose[1], pose[2]
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = x + c * laser[0] - s * laser[1], y + s * laser[0] + c * laser[1]
    r_exp = wall_map.ranges(lx, ly, yaw + laser[2] + np.asarray(beam_angles), max_range)
    meas_far, exp_far = ~np.isfinite(r_meas), ~np.isfinite(r_exp)
    both = ~meas_far & ~exp_far
    close = np.zeros(len(r_exp), dtype=bool)
    close[both] = np.abs(r_meas[both] - r_exp[both]) < tol
    return float(np.mean(close | (meas_far & exp_far)))


def scan_beams(ranges, angle_min, angle_increment, range_min, range_max, n_beams):
    """n_beams evenly spread beams of a LaserScan: (measured ranges, inf = no valid
    return; their angles in the LiDAR frame)."""
    r_all = np.asarray(ranges, dtype=float)
    idx = np.unique(np.linspace(0, len(r_all) - 1, n_beams).round().astype(np.int64))
    r = r_all[idx]
    r = np.where(np.isfinite(r) & (r >= range_min) & (r <= range_max), r, np.inf)
    return r, angle_min + idx * angle_increment


def body_motion(a, b):
    """Motion from pose a to pose b in a's frame: (forward, left, turn)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])


def motion_consistent(streak, tol_pos, tol_frac, tol_yaw):
    """streak = [(t_ns, AMCL pose, EKF pose at that time), ...] of rejected AMCL poses
    (the EKF accepted nothing meanwhile, so its motion = odometry). Did AMCL move the way
    the odometry says the car moved? From the first pose to every later one, each in its
    own frame, so a constant offset between the two does not matter.
    -> (ok, worst position excess over the tolerance [m], worst turn difference [rad])"""
    z0, x0 = streak[0][1], streak[0][2]
    ok, worst_pos, worst_yaw = True, -math.inf, 0.0
    for _, z, x in streak[1:]:
        a, e = body_motion(z0, z), body_motion(x0, x)
        excess = math.hypot(a[0] - e[0], a[1] - e[1]) - (tol_pos + tol_frac * math.hypot(e[0], e[1]))
        dyaw = abs(wrap(a[2] - e[2]))
        worst_pos, worst_yaw = max(worst_pos, excess), max(worst_yaw, dyaw)
        ok = ok and excess <= 0.0 and dyaw <= tol_yaw
    return ok, worst_pos, worst_yaw


# ----------------------------------------------------------------------------
# ROS 2 node
# ----------------------------------------------------------------------------

try:
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
    from nav_msgs.msg import OccupancyGrid, Odometry, Path
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import Imu, LaserScan
    from std_msgs.msg import Float64
    from tf2_ros import TransformBroadcaster

    from state_estimation.calibration import load_steering
except ImportError:
    Node = object


def quat_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class EkfNode(Node):

    def __init__(self, node_name='ekf_node', own_inputs=True):
        # node_name / own_inputs=False: used by state_pipeline.py (both EKFs + state_estimate
        # in ONE process, the IMU and wheel messages arrive through ONE shared subscription)
        super().__init__(node_name)
        d = self.declare_parameter
        d('mode', 'map')                    # 'map' = global EKF, 'odom' = local EKF
        d('frame_id', 'map')                # frame of the estimated pose
        d('child_frame_id', 'est/base_link')
        d('publish_tf', False)              # True for the local EKF: odom -> est/base_link
        d('odom_topic', '/odom/ekf')
        d('path_topic', '/odom/ekf_path')
        d('diag_prefix', '/ekf')
        d('path_max_len', 1500)             # poses kept in the path (5 per s -> 5 min)
        d('path_period_s', 0.0)             # s between path messages; 0 = no path (default)
        d('truth_topic', 'none')            # sim truth for the report; 'none' = off (real car)
        d('diag_every', 10)                 # publish bias/scale every Nth IMU step (10 Hz)
        # Start pose and its uncertainty
        d('x0', 0.0); d('y0', 0.0); d('yaw0', 0.0)
        d('sigma_pos0', 0.1)                # m
        d('sigma_yaw0_deg', 2.0)            # deg
        d('sigma_bias0', 0.05)              # rad/s
        d('sigma_scale0', 0.05)             # 5 %
        # Process noise
        d('sigma_v', 0.05)
        d('sigma_w', 0.005)
        d('sigma_bw', 0.0001)
        d('sigma_kw', 0.0005)
        # AMCL update (global EKF)
        d('use_amcl', True)
        d('amcl_topic', '/amcl_pose')
        d('amcl_cov_scale', 4.0)            # inflate AMCL covariance (shared odometry; 2 -> 4 2026-10-06)
        d('amcl_min_sigma_pos', 0.10)       # m (0.10 chosen 2026-10-01, was 0.05)
        d('amcl_min_sigma_yaw_deg', 1.0)    # deg
        d('amcl_offset_m', 0.025)           # added to AMCL x and y: nav2 half-cell offset = map resolution / 2
        d('map_topic', '/map')              # read once, only to check amcl_offset_m ('' = no check)
        d('gate', CHI2_3DOF_99)
        d('reset_after_rejects', 10)        # confident AMCL poses rejected in a row before a jump is considered
        # Robust jump to AMCL (2026-10-06, docstring RECOVERY)
        d('amcl_max_sigma', 0.35)           # m: AMCL poses with sqrt(cov_xx + cov_yy) above this are ignored
        d('reset_tol_pos', 0.15)            # m   AMCL motion vs odometry over the rejected poses ...
        d('reset_tol_frac', 0.05)           #     ... + 5 % of the distance
        d('reset_tol_yaw_deg', 3.0)         # deg
        d('reset_scan_check', True)         # also require the scan to fit the map clearly better at AMCL's pose
        d('scan_topic', '/scan_amcl')       # the scan AMCL uses
        d('scan_beams', 60)                 # beams used by the scan check
        d('scan_tol', 0.10)                 # m: a beam fits if measured and map range differ less
        d('reset_scan_min', 0.6)            # AMCL's pose must explain >= 60 % of the beams ...
        d('reset_scan_margin', 0.1)         # ... and 10 points more than the EKF's pose
        d('laser_x', 0.275)                 # m   where the LiDAR sits on the car (base_link -> laser;
        d('laser_y', 0.0)                   # m   sim: scan_distance_to_base_link)
        d('laser_yaw', 0.0)                 # rad
        d('test_fault', '')                 # TEST ONLY: 't_s,dx,dy,dyaw_deg' moves ekf_global's pose once
        d('history_s', 2.0)
        # Vehicle geometry for the slip angle
        d('lf', 0.15875); d('lr', 0.17145)
        # Zero-velocity detection
        d('stopped_speed', 0.01)
        d('stopped_time', 0.3)
        # Steering calibration (steer_calib); 'none' = use the measured steering as it is
        d('steer_calib_file', '~/.ros/steering_calibration.yaml')
        # Speed-scale calibration file (see docstring)
        d('scale_file', 'none')             # anchor (READ only); 'none' = start with k = 1
        d('scale_learned_file', '~/.ros/ekf_speed_scale_learned.yaml')  # global EKF writes
        d('scale_warn', 0.01)               # warn if learned k differs > 1 % from the anchor
        d('scale_file_min_sigma', 0.01)     # never trust the file more than 1 %
        d('scale_save_min_distance', 20.0)  # m driven before the global EKF saves


        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.mode = p('mode')
        self.global_mode = self.mode == 'map'
        sf = str(p('scale_file')).strip()
        self.scale_file = os.path.expanduser(sf) if sf and sf.lower() != 'none' else ''
        lf_ = str(p('scale_learned_file')).strip()
        self.learned_file = os.path.expanduser(lf_) if lf_ and lf_.lower() != 'none' else ''
        k0, sigma_k0 = self.load_scale(1.0, p('sigma_scale0'))
        self.k_anchor = k0 if self.scale_file and k0 != 1.0 else None
        self.warned_scale = False
        self.steer_cal, problem = load_steering(p('steer_calib_file'))
        if problem:
            self.get_logger().warn(problem)
        else:
            self.get_logger().info(self.steer_cal.describe())
        self.ekf = LocalizationEKF(
            x0=[p('x0'), p('y0'), p('yaw0'), 0.0, k0],
            P0_diag=[p('sigma_pos0'), p('sigma_pos0'), math.radians(p('sigma_yaw0_deg')),
                     p('sigma_bias0'), sigma_k0],
            sigma_v=p('sigma_v'), sigma_w=p('sigma_w'),
            sigma_bw=p('sigma_bw'), sigma_kw=p('sigma_kw'))
        self.history = StateHistory(p('history_s'))

        self.v = 0.0
        self.steer = 0.0
        self.stopped_since = None
        self.last_imu_t = None
        self.zupt_count = 0
        self.distance = 0.0
        self.truth = None
        self.truth_hist = deque(maxlen=400)
        self.stats = self.new_stats()
        self.streak = []                    # rejected confident AMCL poses in a row: (t_ns, z, x_then)
        self.imu_count = 0
        self.map_checked = False
        self.wall_map = None                # walls of the map, for the scan check
        self.scans = deque(maxlen=40)       # recent scans: (t_ns, msg)
        self.last_refuse_log = -1e9
        self.fault = None                   # test_fault (global EKF only)
        self.fault_t0 = None
        tf_ = str(p('test_fault')).strip()
        if tf_ and self.global_mode:
            vals = [float(v) for v in tf_.split(',')]
            if len(vals) != 4:
                raise ValueError("test_fault must be 't_s,dx,dy,dyaw_deg'")
            self.fault = vals

        self.path = Path()
        self.path.header.frame_id = p('frame_id')
        self.tf_pub = TransformBroadcaster(self) if p('publish_tf') else None

        # In-process consumers (state_pipeline): called with every published odometry and with
        # the NIS message of every ACCEPTED AMCL fix -- the same data the topics carry.
        self.odom_hooks = []
        self.fix_hooks = []
        if own_inputs:
            self.create_subscription(Imu, '/imu/data', self.imu_cb, 100)
            self.create_subscription(AckermannDriveStamped, '/vehicle/measured', self.meas_cb, 50)
        if self.global_mode:
            self.create_subscription(PoseWithCovarianceStamped, p('amcl_topic'), self.amcl_cb, 10)
            self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.reset_cb, 10)
            if p('use_amcl') and str(p('map_topic')).strip():
                # map_server publishes the map once, "latched" (transient local) -> late joiners get it
                self.create_subscription(OccupancyGrid, p('map_topic'), self.map_cb, QoSProfile(
                    depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    reliability=ReliabilityPolicy.RELIABLE))
            if p('use_amcl') and p('reset_scan_check'):
                self.create_subscription(LaserScan, p('scan_topic'), self.scan_cb, qos_profile_sensor_data)
            tt = str(p('truth_topic')).strip()
            if tt and tt.lower() != 'none':
                self.create_subscription(Odometry, tt, self.truth_cb, 10)
        self.odom_pub = self.create_publisher(Odometry, p('odom_topic'), 10)
        self.path_pub = self.create_publisher(Path, p('path_topic'), 10)
        pre = p('diag_prefix')
        self.bias_pub = self.create_publisher(Float64, pre + '/bias', 10)
        self.bias_sigma_pub = self.create_publisher(Float64, pre + '/bias_sigma', 10)
        self.scale_pub = self.create_publisher(Float64, pre + '/speed_scale', 10)
        self.nis_pub = self.create_publisher(Float64, pre + '/nis', 10)
        if p('path_period_s') > 0.0:
            self.create_timer(p('path_period_s'), self.publish_path)
        self.create_timer(2.0, self.report)
        if self.global_mode and self.learned_file:
            self.create_timer(10.0, self.save_scale)
        what = ('GLOBAL (map frame, wheel + gyro + AMCL)' if self.global_mode
                else f'LOCAL (odom frame, wheel + gyro), TF {p("frame_id")} -> {p("child_frame_id")}')
        self.get_logger().info(f'EKF v3 {what}')
        if self.global_mode and p('use_amcl'):
            self.get_logger().info(
                f"AMCL half-cell correction: +{100.0 * p('amcl_offset_m'):.2f} cm in x and y "
                '(amcl_offset_m = map resolution / 2; 0 = off)')
            self.get_logger().info(
                f"jump to AMCL only if it is clearly right: ignore AMCL sigma > {p('amcl_max_sigma'):.2f} m; "
                f"after {p('reset_after_rejects')} rejected poses AMCL must move like the odometry"
                + (f" AND the scan ({p('scan_topic')}) must fit AMCL's pose >= {p('reset_scan_min'):.2f} and "
                   f"{p('reset_scan_margin'):.2f} better than the EKF's" if p('reset_scan_check') else
                   ' (scan check OFF)'))
        if self.fault is not None:
            self.get_logger().warn(f'TEST FAULT armed: ekf_global pose will move by ({self.fault[1]}, '
                                   f'{self.fault[2]}) m, {self.fault[3]} deg, {self.fault[0]} s after the first IMU')

    # --- speed-scale calibration file ----------------------------------------

    def load_scale(self, k_default, sigma_default):
        """(k0, sigma_k0) from the calibration file, or the defaults."""
        if not self.scale_file:
            return k_default, sigma_default
        try:
            vals = {}
            with open(self.scale_file, encoding='utf-8') as f:
                for line in f:
                    if ':' in line and not line.lstrip().startswith('#'):
                        key, val = line.split(':', 1)
                        vals[key.strip()] = val.strip()
            k = float(vals['speed_scale'])
            sig = max(float(vals.get('speed_scale_sigma', sigma_default)),
                      self.get_parameter('scale_file_min_sigma').value)
            if not 0.8 < k < 1.2:
                raise ValueError(f'speed_scale {k} outside 0.8..1.2')
        except FileNotFoundError:
            self.get_logger().warn(f'No calibration file {self.scale_file} -> starting with '
                                   f'k = {k_default}. Create it with wheel_calib.')
            return k_default, sigma_default
        except (KeyError, ValueError) as e:
            self.get_logger().error(f'Bad calibration file {self.scale_file} ({e}) -> k = {k_default}')
            return k_default, sigma_default
        self.get_logger().info(f'Speed scale from {self.scale_file}: k0 = {k:.4f} (+-{sig:.3f})')
        return k, sig

    def save_scale(self):
        """Global EKF only: write the LEARNED k to its own file (never the anchor)."""
        if not (self.global_mode and self.learned_file):
            return
        if self.distance < self.get_parameter('scale_save_min_distance').value:
            return
        k, sig = float(self.ekf.x[4]), self.ekf.sigma(4)
        if self.k_anchor is not None and not self.warned_scale:
            diff = (k - self.k_anchor) / self.k_anchor
            if abs(diff) > self.get_parameter('scale_warn').value:
                self.get_logger().warn(
                    f'Learned speed scale {k:.4f} differs {100 * diff:+.1f} % from the '
                    f'calibration {self.k_anchor:.4f} -> tyres changed/worn? Redo wheel_calib.')
                self.warned_scale = True
        tmp = self.learned_file + '.tmp'
        try:
            os.makedirs(os.path.dirname(self.learned_file) or '.', exist_ok=True)
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write('# Wheel-speed scale learned by ekf_global (v_true = k * v_measured)\n')
                f.write(f'speed_scale: {k:.6f}\n')
                f.write(f'speed_scale_sigma: {sig:.6f}\n')
                f.write(f'distance_m: {self.distance:.1f}\n')
                if self.k_anchor is not None:
                    f.write(f'anchor_speed_scale: {self.k_anchor:.6f}\n')
                f.write(f"saved: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            os.replace(tmp, self.learned_file)
        except OSError as e:
            self.get_logger().error(f'Could not save {self.learned_file}: {e}')

    # --- inputs --------------------------------------------------------------

    def meas_cb(self, msg):
        self.v = msg.drive.speed
        self.steer = self.steer_cal.correct(msg.drive.steering_angle)
        now = Time.from_msg(msg.header.stamp)
        if abs(self.v) < self.get_parameter('stopped_speed').value:
            if self.stopped_since is None:
                self.stopped_since = now
        else:
            self.stopped_since = None

    def is_stopped(self, now):
        if self.stopped_since is None:
            return False
        waited = (now - self.stopped_since).nanoseconds * 1e-9
        return waited >= self.get_parameter('stopped_time').value

    def imu_cb(self, msg):
        t = Time.from_msg(msg.header.stamp)
        if self.last_imu_t is None:
            self.last_imu_t = t
            return
        dt = (t - self.last_imu_t).nanoseconds * 1e-9
        self.last_imu_t = t
        if dt <= 0.0 or dt > 0.5:
            return
        w_m = msg.angular_velocity.z

        if self.is_stopped(t):
            self.ekf.stationary_predict(dt)
            self.ekf.zupt(w_m)
            self.zupt_count += 1
        else:
            lf, lr = self.get_parameter('lf').value, self.get_parameter('lr').value
            beta = math.atan(lr / (lf + lr) * math.tan(self.steer))
            self.ekf.predict(self.v, w_m, beta, dt)
            self.distance += abs(self.ekf.x[4] * self.v) * dt

        if self.fault is not None:                  # TEST ONLY (test_fault)
            if self.fault_t0 is None:
                self.fault_t0 = t
            elif (t - self.fault_t0).nanoseconds * 1e-9 >= self.fault[0]:
                _, dx_, dy_, dyaw_ = self.fault
                self.ekf.x[0] += dx_
                self.ekf.x[1] += dy_
                self.ekf.x[2] = wrap(self.ekf.x[2] + math.radians(dyaw_))
                self.history.buf.clear()
                self.get_logger().warn(f'TEST FAULT: ekf_global pose moved by ({dx_}, {dy_}) m, {dyaw_} deg')
                self.fault = None
        self.history.add(t.nanoseconds, self.ekf.x)
        self.publish_odom(msg.header.stamp, w_m)

    def amcl_cb(self, msg):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        if not p('use_amcl'):
            return
        t_ns = Time.from_msg(msg.header.stamp).nanoseconds
        x_then = self.history.at(t_ns)
        if x_then is None:
            return                                  # too old, or no IMU yet
        pose = msg.pose.pose
        off = p('amcl_offset_m')                    # nav2 AMCL half-cell offset (see docstring)
        z = np.array([pose.position.x + off, pose.position.y + off, yaw_from_quat(pose.orientation)])
        nu = z - x_then[:3]
        nu[2] = wrap(nu[2])
        R = amcl_measurement_noise(msg.pose.covariance, p('amcl_cov_scale'),
                                   p('amcl_min_sigma_pos'),
                                   math.radians(p('amcl_min_sigma_yaw_deg')))
        self.record_amcl_error(t_ns, z)
        cov = msg.pose.covariance
        if math.sqrt(max(cov[0] + cov[7], 0.0)) > p('amcl_max_sigma'):
            self.stats['unsure'] += 1               # AMCL itself is unsure (lost?) -> ignore this pose
            return

        nis = self.ekf.pose_nis(nu, R)
        if nis > p('gate'):
            self.stats['rej'] += 1
            self.streak.append((t_ns, z.copy(), x_then[:3].copy()))
            if len(self.streak) >= p('reset_after_rejects'):
                ok, why = self.jump_check(t_ns, z, x_then)
                if ok:
                    self.get_logger().warn(f'{len(self.streak)} AMCL poses rejected in a row; {why} '
                                           '-> EKF jumps to AMCL')
                    self.ekf.reset_pose(z[0], z[1], z[2], R)
                    self.history.buf.clear()
                    self.streak = []
                    self.stats['jumps'] += 1
                else:
                    self.streak.pop(0)              # slide: the next rejected pose re-checks
                    self.stats['refused'] += 1
                    now = time.monotonic()
                    if now - self.last_refuse_log > 2.0:
                        self.last_refuse_log = now
                        self.get_logger().warn(f'AMCL rejected {p("reset_after_rejects")}+ times in a row, '
                                               f'NOT jumping: {why} -> EKF continues on odometry')
            return

        self.streak = []
        nis, dx = self.ekf.pose_update(nu, R)
        self.history.shift(dx)
        self.stats['acc'] += 1
        self.stats['nis'] += nis
        nis_msg = Float64(data=nis)
        self.nis_pub.publish(nis_msg)
        for hook in self.fix_hooks:
            hook(nis_msg)

    def reset_cb(self, msg):
        pose = msg.pose.pose
        self.ekf.reset_pose(pose.position.x, pose.position.y, yaw_from_quat(pose.orientation),
                            np.diag([0.3 ** 2, 0.3 ** 2, math.radians(10.0) ** 2]))
        self.history.buf.clear()
        self.distance = 0.0
        self.path.poses.clear()
        self.get_logger().info('EKF pose reset from /initialpose (bias and speed scale kept)')

    def jump_check(self, t_ns, z, x_then):
        """-> (ok, reason): may the EKF jump to AMCL? (docstring RECOVERY)"""
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        ok, worst_pos, worst_yaw = motion_consistent(self.streak, p('reset_tol_pos'), p('reset_tol_frac'),
                                                     math.radians(p('reset_tol_yaw_deg')))
        if not ok:
            return False, (f'AMCL did not move like the odometry ({worst_pos:+.2f} m over the tolerance, '
                           f'turn {math.degrees(worst_yaw):.1f} deg)')
        if not p('reset_scan_check'):
            return True, 'AMCL moved like the odometry (scan check off)'
        scan = self.scan_at(t_ns)
        if self.wall_map is None or scan is None:
            return False, 'AMCL moved like the odometry, but no map or no scan for the scan check'
        fit_amcl, fit_ekf = self.scan_fits(scan, (z, x_then[:3]))
        if fit_amcl < p('reset_scan_min') or fit_amcl - fit_ekf < p('reset_scan_margin'):
            return False, (f'AMCL moved like the odometry, but the scan does not fit it clearly better '
                           f'(scan fit AMCL {fit_amcl:.2f}, EKF {fit_ekf:.2f})')
        return True, (f'AMCL moved like the odometry and the scan fits it better '
                      f'(scan fit AMCL {fit_amcl:.2f}, EKF {fit_ekf:.2f})')

    def scan_fits(self, scan, poses):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        r, ang = scan_beams(scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min,
                            scan.range_max, int(p('scan_beams')))
        laser = (p('laser_x'), p('laser_y'), p('laser_yaw'))
        return [scan_fit(self.wall_map, q, laser, ang, r, scan.range_max, p('scan_tol')) for q in poses]

    def scan_cb(self, msg):
        self.scans.append((Time.from_msg(msg.header.stamp).nanoseconds, msg))

    def scan_at(self, t_ns, max_dt_ns=50_000_000):
        """The scan taken at t_ns (AMCL's pose carries the stamp of the scan it used)."""
        if not self.scans:
            return None
        ts, msg = min(self.scans, key=lambda e: abs(e[0] - t_ns))
        return msg if abs(ts - t_ns) <= max_dt_ns else None

    def map_cb(self, msg):
        """Once: is amcl_offset_m half the resolution of the map AMCL uses? + keep its walls."""
        if self.map_checked:
            return
        self.map_checked = True
        info = msg.info
        o = info.origin
        if abs(o.orientation.z) > 1e-6 or abs(o.orientation.x) > 1e-6 or abs(o.orientation.y) > 1e-6:
            self.get_logger().warn('map origin is rotated -> no scan check (jumps to AMCL refused)')
        else:
            self.wall_map = WallMap.from_grid(msg.data, info.width, info.height, info.resolution,
                                              o.position.x, o.position.y)
            self.get_logger().info(f'scan check: map {info.width} x {info.height} cells @ {info.resolution:.3f} m, '
                                   f'{int(self.wall_map.occ.sum())} wall cells')
        off, res = self.get_parameter('amcl_offset_m').value, msg.info.resolution
        if abs(off - 0.5 * res) > 1e-4:
            self.get_logger().warn(
                f'amcl_offset_m is {off:.4f} m but the map resolution is {res:.4f} m: it should be '
                f'{0.5 * res:.4f} m (nav2 AMCL half-cell offset, see the ekf_node docstring)')
        else:
            self.get_logger().info(f'amcl_offset_m {off:.4f} m = half the map resolution ({res:.3f} m): OK')

    def truth_cb(self, msg):
        self.truth = msg
        tp = msg.pose.pose
        self.truth_hist.append((Time.from_msg(msg.header.stamp).nanoseconds,
                                tp.position.x, tp.position.y))

    def record_amcl_error(self, t_ns, z):
        """AMCL's own error vs. truth at the scan time (simulation only)."""
        if not self.truth_hist:
            return
        _, tx, ty = min(self.truth_hist, key=lambda e: abs(e[0] - t_ns))
        self.stats['amcl_err'].append(math.hypot(z[0] - tx, z[1] - ty))

    # --- outputs -------------------------------------------------------------

    def publish_odom(self, stamp, w_m):
        px, py, yaw, b, k = (float(v) for v in self.ekf.x)
        P = self.ekf.P
        q = quat_from_yaw(yaw)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.path.header.frame_id
        odom.child_frame_id = self.get_parameter('child_frame_id').value
        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        (odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
         odom.pose.pose.orientation.z, odom.pose.pose.orientation.w) = q
        cov = [0.0] * 36
        cov[0], cov[1], cov[5] = P[0, 0], P[0, 1], P[0, 2]
        cov[6], cov[7], cov[11] = P[1, 0], P[1, 1], P[1, 2]
        cov[30], cov[31], cov[35] = P[2, 0], P[2, 1], P[2, 2]
        odom.pose.covariance = [float(c) for c in cov]
        odom.twist.twist.linear.x = k * self.v
        odom.twist.twist.angular.z = float(w_m) - b
        self.odom_pub.publish(odom)
        for hook in self.odom_hooks:
            hook(odom)
        self.imu_count += 1
        if self.imu_count % self.get_parameter('diag_every').value == 0:
            self.bias_pub.publish(Float64(data=b))
            self.bias_sigma_pub.publish(Float64(data=self.ekf.sigma(3)))
            self.scale_pub.publish(Float64(data=k))

        if self.tf_pub is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.path.header.frame_id
            tf.child_frame_id = odom.child_frame_id
            tf.transform.translation.x = px
            tf.transform.translation.y = py
            (tf.transform.rotation.x, tf.transform.rotation.y,
             tf.transform.rotation.z, tf.transform.rotation.w) = q
            self.tf_pub.sendTransform(tf)

    def publish_path(self):
        px, py, yaw = (float(v) for v in self.ekf.x[:3])
        ps = PoseStamped()
        ps.header.frame_id = self.path.header.frame_id
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = px, py
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = quat_from_yaw(yaw)
        self.path.poses.append(ps)
        del self.path.poses[:-self.get_parameter('path_max_len').value]
        self.path.header.stamp = ps.header.stamp
        self.path_pub.publish(self.path)

    def report(self):
        px, py, yaw, b, k = self.ekf.x
        line = f'[{self.mode}] driven {self.distance:5.1f} m'
        if self.global_mode and self.truth is not None:
            tp = self.truth.pose.pose
            pos_err = math.hypot(px - tp.position.x, py - tp.position.y)
            yaw_err = math.degrees(wrap(yaw - yaw_from_quat(tp.orientation)))
            line += (f' | pos err {pos_err:5.2f} m (+-{2 * self.ekf.sigma_pos():4.2f})'
                     f' | heading err {yaw_err:+6.2f} deg'
                     f' (+-{2 * math.degrees(self.ekf.sigma(2)):4.2f})')
        line += (f' | bias {math.degrees(b):+5.2f} deg/s'
                 f' (+-{2 * math.degrees(self.ekf.sigma(3)):4.2f})')
        if self.global_mode:
            s = self.stats
            line += f' | speed scale {k:5.3f} (+-{2 * self.ekf.sigma(4):5.3f})'
            mean_nis = s['nis'] / s['acc'] if s['acc'] else float('nan')
            amcl = (f"{sum(s['amcl_err']) / len(s['amcl_err']):4.2f} m"
                    if s['amcl_err'] else ' n/a')
            line += (f" | AMCL used {s['acc']:2d} rej {s['rej']:2d} unsure {s['unsure']:2d}"
                     f' NIS {mean_nis:4.1f} AMCL err {amcl}')
            if s['refused'] or s['jumps']:
                line += f" | jump refused {s['refused']} JUMPED {s['jumps']}"
            fit = self.scan_fit_now()
            if fit is not None:
                line += f' | scan fit {fit:.2f}'
        line += f' | ZUPTs {self.zupt_count}'
        self.stats = self.new_stats()
        self.get_logger().info(line)

    @staticmethod
    def new_stats():
        return {'acc': 0, 'rej': 0, 'nis': 0.0, 'amcl_err': [], 'unsure': 0, 'refused': 0, 'jumps': 0}

    def scan_fit_now(self):
        """Health check every 2 s: how well does the latest scan fit the map at the EKF's
        pose? (~0.9-1.0 when localized; proves the scan check sees what it should.)"""
        if self.wall_map is None or not self.scans:
            return None
        t_ns, scan = self.scans[-1]
        x = self.history.at(t_ns)
        return None if x is None else self.scan_fits(scan, (x[:3],))[0]


def main(args=None):
    rclpy.init(args=args)
    node = EkfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_scale()                       # last learned k, at Ctrl+C
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
