#!/usr/bin/env python3
"""Record everything needed to evaluate the localization, 10 rows per second.

One CSV row every 0.1 s:
  t, s                      time since start [s], distance driven (truth) [m]
  tx, ty, tyaw              simulator ground truth AT THE STAMP of the EKF pose in this row
                            (2026-10-18; before: the newest truth message -- see "RACING SPEED")
  ex, ey, eyaw, e_sig, e_sig_yaw   global EKF (/odom/ekf) and its 1-sigma
                                   (e_sig = sqrt(Pxx + Pyy)): the newest EKF pose whose truth is known
  wx, wy, wyaw              wheel-only odometry (/odom/wheel), newest
  a_err, a_err_yaw          AMCL error vs truth AT THE SCAN TIME (latest pose)
  k, bias, true_bias        EKF speed scale, EKF gyro bias, simulated true bias
  v                         true speed (at the EKF stamp)
StateEstimate (what control receives, /state_estimate) -- added 2026-10-02:
  se_s, se_ey, se_epsi      s_abs [m], e_y [m], e_psi [rad] from the message
  se_vx, se_vy, se_r        body velocities [m/s] and yaw rate [rad/s] from the message
  se_std_ey, se_std_epsi    its 1-sigma for e_y and e_psi
  se_mode, se_flags         mode (0 INIT, 1 OK, 2 DEGRADED, 3 LOST) and flag bits
  se_lat_ms                 publish_us - stamp_us (estimator latency)
  se_age_ms                 arrival here - stamp_us (what a subscriber like control sees)
  se_seq                    sequence_id (its increase per second = output rate)
  tr_s, tr_ey, tr_epsi, tr_kappa   TRUE Frenet state, computed with apex_track from the
                            true pose AT THE MESSAGE'S STAMP (not the newest truth: a 10 ms
                            mismatch would look like 1.5 cm of error at 1.5 m/s)
  tr_vx, tr_vy              true body velocities (simulator twist) at that stamp
  tr_r                      true yaw rate of the gyro reading the message was made from:
                            /sim/imu_ideal z (fake_imu publishes, for every reading, the same
                            stamp + the noise- and bias-free yaw rate it used) -> se_r - tr_r
                            = gyro noise + bias-estimate error, exactly. Fallback if that
                            reading is missing: the yaw-rate state (twist.angular.z) of the
                            newest NEW physics state first published >= STATE_DELIVERY_NS
                            (1 ms) before the stamp. 0 while the car stands (< 0.05 m/s; the
                            state has been seen stuck after stops). How many rows used which
                            source is in the log.
                            Why not the truth message nearest to the stamp (used 2026-10-02
                            .. 10-03): the simulator's yaw rate JUMPS at every 10 ms physics
                            step (its steering actuator moves the wheels at full speed,
                            3.2 rad/s, until within 1e-4 rad of the target -> on straights
                            they flick by 0.032 rad every step), and the nearest message can
                            already carry the NEXT state, which the gyro has not seen yet.
                            fake_imu's 100 Hz timer keeps a fixed phase to the physics steps
                            for long stretches, so this "error" was 0.07 rad/s p95 in some
                            runs and 0.01 (= the gyro noise) in others (jobs 047-048); even
                            "the state published >= 1 ms before" is ambiguous when a reading
                            falls 0-2 ms after a new state (job 048 run 1: p95 0.019).
                            Before 2026-10-02: the heading change over +-20 ms (counts 3, 4 or
                            5 physics steps, p95 0.034 rad/s, job 023).

RACING SPEED (2026-10-18): positions, heading and body velocities are now INTERPOLATED at every
stamp between the two physics states around it (truth_interp), for the StateEstimate (tr_*),
the EKF (tx, ty, tyaw, v) and AMCL (a_err).  Before, the StateEstimate used the truth MESSAGE
nearest to its stamp (up to half a 10 ms physics step off, or already the next state: up to 3 cm
at 6 m/s) and the EKF columns compared the NEWEST EKF pose with the NEWEST truth (up to ~10 ms
apart: up to 6 cm "error" at 6 m/s that the EKF did not make).  A row now uses the newest EKF pose
and StateEstimate whose truth is FINAL (the next physics state has arrived), i.e. ~10-20 ms older
than the newest message.  The log reports, for moving rows, how far the old nearest-message truth
was from the interpolated one (that is what the old method added to the errors).
File: ~/eval_logs/<run_name>_<YYYYmmdd_HHMMSS>.csv  (plot with plot_eval.py)
"""
import math
import os
import time
from collections import deque

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float64

from ament_index_python.packages import get_package_share_directory
from apex_msgs.msg import StateEstimate
from apex_track import load_track

SE_COLS = ['se_s', 'se_ey', 'se_epsi', 'se_vx', 'se_vy', 'se_r', 'se_std_ey', 'se_std_epsi',
           'se_mode', 'se_flags', 'se_lat_ms', 'se_age_ms', 'se_seq',
           'tr_s', 'tr_ey', 'tr_epsi', 'tr_vx', 'tr_vy', 'tr_r', 'tr_kappa']
COLS = ['t', 's', 'tx', 'ty', 'tyaw', 'ex', 'ey', 'eyaw', 'e_sig', 'e_sig_yaw',
        'wx', 'wy', 'wyaw', 'a_err', 'a_err_yaw', 'k', 'bias', 'true_bias', 'v'] + SE_COLS
STATE_DELIVERY_NS = 1_000_000    # a physics state reaches fake_imu ~0.7 ms after it is published
PHYS_NS = 10_000_000             # simulator physics step (10 ms)
PUB_NS = 4_000_000               # simulator publish timer (4 ms): a new state is out within this


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def interp_states(s0, s1, t_ns):
    """Linear interpolation between two physics states (t_first_ns, x, y, yaw, vx, vy, wz) at t_ns,
    over the last physics step before s1 (a longer gap = standing / paused: s0 holds until then).
    Returns ((x, y, yaw, vx, vy, wz), 'interp' | 'hold').  Pure function (tested offline)."""
    w = min(s1[0] - s0[0], PHYS_NS)
    f = (t_ns - (s1[0] - w)) / w if w > 0 else 1.0
    if f <= 0.0:
        return tuple(s0[1:]), 'hold'
    f = min(f, 1.0)
    return ((s0[1] + f * (s1[1] - s0[1]), s0[2] + f * (s1[2] - s0[2]),
             wrap(s0[3] + f * wrap(s1[3] - s0[3])),
             s0[4] + f * (s1[4] - s0[4]), s0[5] + f * (s1[5] - s0[5]), s0[6] + f * (s1[6] - s0[6])),
            'interp')


def pct(values, q):
    if not values:
        return float('nan')
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q / 100.0 * (len(v) - 1))))]


class EvalLogger(Node):

    def __init__(self):
        super().__init__('eval_logger')
        self.declare_parameter('run_name', 'run')
        self.declare_parameter('out_dir', os.path.expanduser('~/eval_logs'))
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('track', 'levine')       # apex_track track for the TRUE Frenet state
        self.declare_parameter('track_yaml', '')        # '' = apex_track's tracks/<track>/track.yaml

        out_dir = self.get_parameter('out_dir').value
        os.makedirs(out_dir, exist_ok=True)
        name = f"{self.get_parameter('run_name').value}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.path = os.path.join(out_dir, name)
        self.f = open(self.path, 'w')
        self.f.write(','.join(COLS) + '\n')

        self.truth = None
        self.truth_hist = deque(maxlen=1500)         # (t_ns, x, y, yaw, vx, vy, wz), a few s
        self.state_hist = deque(maxlen=600)          # (first stamp ns, wz) of each NEW physics state
        self.states = deque(maxlen=3000)             # NEW physics states (first stamp ns, x, y, yaw, vx, vy, wz), ~30 s
        self.last_msg_ns = None                      # stamp of the newest truth message (re-sends included)
        self.ideal = {}                              # gyro reading stamp [us] -> its true yaw rate
        self.ideal_keys = deque()                    # same keys, oldest first (keep ~3 s)
        self.r_src = {'reading': 0, 'state': 0, 'standing': 0}   # source of tr_r per row
        self.truth_kind = {'interp': 0, 'hold': 0, 'none': 0}   # how tr_* was obtained per row
        self.near_diff = []                          # |nearest-message truth - interpolated| [m], moving rows
        self.last_xy = None
        self.s = 0.0
        self.ekf_hist = deque(maxlen=200)            # /odom/ekf messages, ~2 s
        self.wheel = None
        self.amcl = (float('nan'), float('nan'))
        self.k = self.bias = self.true_bias = float('nan')
        self.t0 = None
        self.rows = 0
        self.se_hist = deque(maxlen=200)             # (StateEstimate, arrival ns), ~2 s
        ty = self.get_parameter('track_yaml').value or (
            get_package_share_directory('apex_track')
            + f"/tracks/{self.get_parameter('track').value}/track.yaml")
        self.track = load_track(ty)                  # own lap counter for the truth
        self.track_first = True

        self.create_subscription(Odometry, '/ego_racecar/odom', self.truth_cb, 10)
        self.create_subscription(Odometry, '/odom/ekf', lambda m: self.ekf_hist.append(m), 50)
        self.create_subscription(Odometry, '/odom/wheel', lambda m: setattr(self, 'wheel', m), 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_cb, 10)
        self.create_subscription(Float64, '/ekf/speed_scale', lambda m: setattr(self, 'k', m.data), 10)
        self.create_subscription(Float64, '/ekf/bias', lambda m: setattr(self, 'bias', m.data), 10)
        self.create_subscription(StateEstimate, '/state_estimate', self.se_cb, 50)
        self.create_subscription(Float64, '/sim/true_gyro_bias',
                                 lambda m: setattr(self, 'true_bias', m.data), 10)
        self.create_subscription(Vector3Stamped, '/sim/imu_ideal', self.ideal_cb, 50)
        self.create_timer(1.0 / self.get_parameter('rate_hz').value, self.write_row)
        self.create_timer(10.0, self.report)
        self.get_logger().info(f'Logging to {self.path} (truth interpolated at every stamp)')

    def truth_cb(self, msg):
        pp = msg.pose.pose
        x, y = pp.position.x, pp.position.y
        t_ns = Time.from_msg(msg.header.stamp).nanoseconds
        tw = msg.twist.twist
        yaw = yaw_from_quat(pp.orientation)
        if (x, y) != self.last_xy:                   # a NEW physics state (re-sends repeat it)
            self.state_hist.append((t_ns, tw.angular.z))
            self.states.append((t_ns, x, y, yaw, tw.linear.x, tw.linear.y, tw.angular.z))
        if self.last_xy is not None:
            step = math.hypot(x - self.last_xy[0], y - self.last_xy[1])
            if step < 0.5:                           # ignore teleports
                self.s += step
        self.last_xy = (x, y)
        self.truth = msg
        self.last_msg_ns = t_ns
        self.truth_hist.append((t_ns, x, y, yaw, tw.linear.x, tw.linear.y, tw.angular.z))

    # ---------------------------------------------------------------- truth at a stamp (2026-10-18)
    def known_until(self):
        """Latest time whose truth is final.  While the car moves: just before the newest physics
        state (a later time may still get a newer state).  If no new state came for more than one
        step + one publish period, the car stands (or the simulator paused): the newest state
        holds until the newest message."""
        if not self.states or self.last_msg_ns is None:
            return None
        t_last = self.states[-1][0]
        if self.last_msg_ns - t_last > PHYS_NS + PUB_NS:
            return self.last_msg_ns - PUB_NS
        return t_last - 1

    def truth_interp(self, t_ns):
        """True (x, y, yaw, vx, vy, wz) at t_ns and how ('interp' | 'hold'), or None.
        The simulator computes a new state every 10 ms (RK4: the car moves continuously in between)
        and our simulated sensors report that continuous motion, so the estimate at a stamp means
        "the pose at that moment" -> interpolate between the state first published before t_ns and
        the next one.  A state's time = when it was FIRST published (re-sends repeat it) -- the
        same clock the simulated sensors work on."""
        st = self.states
        if not st or t_ns < st[0][0]:
            return None
        i = len(st) - 1
        while i >= 0 and st[i][0] > t_ns:
            i -= 1
        if i == len(st) - 1:
            return tuple(st[i][1:]), 'hold'
        return interp_states(st[i], st[i + 1], t_ns)

    def truth_nearest(self, t_ns):
        """The old method (kept only for comparison): the truth MESSAGE nearest to t_ns."""
        return min(self.truth_hist, key=lambda e: abs(e[0] - t_ns))[1:]

    def amcl_cb(self, msg):
        if not self.truth_hist:
            return
        t = Time.from_msg(msg.header.stamp).nanoseconds
        got = self.truth_interp(t)
        tx, ty, tyaw = (got[0] if got is not None else self.truth_nearest(t))[:3]
        pp = msg.pose.pose
        self.amcl = (math.hypot(pp.position.x - tx, pp.position.y - ty),
                     math.degrees(wrap(yaw_from_quat(pp.orientation) - tyaw)))

    def se_cb(self, msg):
        self.se_hist.append((msg, self.get_clock().now().nanoseconds))

    def ideal_cb(self, msg):
        """fake_imu: the noise/bias-free yaw rate of the reading with this stamp (sim only)."""
        key = Time.from_msg(msg.header.stamp).nanoseconds // 1000     # = StateEstimate stamp_us
        self.ideal[key] = msg.vector.z
        self.ideal_keys.append(key)
        while len(self.ideal_keys) > 300:
            self.ideal.pop(self.ideal_keys.popleft(), None)

    def rate_at(self, t_ns):
        """True yaw rate the simulated gyro could have seen at t_ns: the yaw-rate state of the
        newest physics state first published at least STATE_DELIVERY_NS before t_ns."""
        limit = t_ns - STATE_DELIVERY_NS
        for t_state, wz in reversed(self.state_hist):    # newest first; usually 1-2 steps back
            if t_state <= limit:
                return wz
        return float('nan')

    def truth_at(self, t_us):
        """True pose + body velocities INTERPOLATED at the stamp, and the true yaw rate at the
        stamp: the gyro reading's own true rate if fake_imu sent it, else the physics state's."""
        t_ns = t_us * 1000
        got = self.truth_interp(t_ns)
        if got is None:
            self.truth_kind['none'] += 1
            return None, float('nan')
        truth, kind = got
        self.truth_kind[kind] += 1
        if math.hypot(truth[3], truth[4]) < 0.05:
            src, r = 'standing', 0.0                                    # standing car: 0
        elif t_us in self.ideal:
            src, r = 'reading', self.ideal[t_us]
        else:
            src, r = 'state', self.rate_at(t_ns)
        self.r_src[src] += 1
        if math.hypot(truth[3], truth[4]) > 0.5 and self.truth_hist:   # how much the old method was off
            near = self.truth_nearest(t_ns)
            self.near_diff.append(math.hypot(near[0] - truth[0], near[1] - truth[1]))
        return truth, r

    def se_columns(self, limit_ns):
        nan = float('nan')
        item = None
        if limit_ns is not None:
            for it in reversed(self.se_hist):            # newest StateEstimate whose truth is final
                if it[0].stamp_us * 1000 <= limit_ns:
                    item = it
                    break
        if item is None:
            return [nan] * len(SE_COLS)
        m, rx_ns = item
        truth, tr_r = self.truth_at(m.stamp_us)
        if truth is None:
            return [nan] * len(SE_COLS)
        tx, ty, tyaw, tvx, tvy, _ = truth
        tr_s, s_tr, tr_ey, tr_epsi, tr_k = self.track.project(tx, ty, tyaw)
        if self.track_first and s_tr > self.track.L - 2.0:   # same rule as the node:
            self.track.lap = -1                              # started just behind the line
            tr_s -= self.track.L
        self.track_first = False
        age = (rx_ns / 1000.0 - m.stamp_us) / 1000.0
        return [m.s_abs_m, m.e_y_m, m.e_psi_rad, m.vx_mps, m.vy_mps, m.r_radps,
                m.std_e_y_m, m.std_e_psi_rad, float(m.mode), float(m.flags),
                (m.publish_us - m.stamp_us) / 1000.0, age, float(m.sequence_id),
                tr_s, tr_ey, tr_epsi, tvx, tvy, tr_r, tr_k]

    def write_row(self):
        if self.truth is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        self.t0 = self.t0 or now
        limit = self.known_until()
        nan = float('nan')
        tp = self.truth.pose.pose
        tx, ty, tyaw = tp.position.x, tp.position.y, yaw_from_quat(tp.orientation)
        v = math.hypot(self.truth.twist.twist.linear.x, self.truth.twist.twist.linear.y)
        ex = ey = eyaw = esig = esigy = nan
        ekf = None
        if limit is not None:
            for m in reversed(self.ekf_hist):            # newest EKF pose whose truth is final
                if Time.from_msg(m.header.stamp).nanoseconds <= limit:
                    ekf = m
                    break
        if ekf is not None:
            got = self.truth_interp(Time.from_msg(ekf.header.stamp).nanoseconds)
            if got is not None:
                tx, ty, tyaw, tvx, tvy, _ = got[0]
                v = math.hypot(tvx, tvy)
                ep = ekf.pose
                ex, ey, eyaw = ep.pose.position.x, ep.pose.position.y, yaw_from_quat(ep.pose.orientation)
                c = ep.covariance
                esig = math.sqrt(max(c[0] + c[7], 0.0))
                esigy = math.sqrt(max(c[35], 0.0))
        wx = wy = wyaw = nan
        if self.wheel is not None:
            wp = self.wheel.pose.pose
            wx, wy, wyaw = wp.position.x, wp.position.y, yaw_from_quat(wp.orientation)
        row = [now - self.t0, self.s, tx, ty, tyaw,
               ex, ey, eyaw, esig, esigy, wx, wy, wyaw, self.amcl[0], self.amcl[1],
               self.k, self.bias, self.true_bias, v] + self.se_columns(limit)
        self.f.write(','.join(f'{x:.5f}' for x in row) + '\n')
        self.rows += 1
        if self.rows % 50 == 0:
            self.f.flush()

    def truth_summary(self):
        k, d = self.truth_kind, self.near_diff
        return (f'truth at the StateEstimate stamps: interpolated {k["interp"]}, held (standing) {k["hold"]}, '
                f'unknown {k["none"]}; the old nearest-message truth differed (moving rows, n={len(d)}) by '
                f'median {100 * pct(d, 50):.2f} cm, p95 {100 * pct(d, 95):.2f} cm, max {100 * pct(d, 100):.2f} cm')

    def report(self):
        c = self.r_src
        self.get_logger().info(f'{self.rows} rows, {self.s:.1f} m driven -> {self.path} | true yaw rate '
                               f'from the gyro reading {c["reading"]}, from the physics state '
                               f'{c["state"]}, standing {c["standing"]} | {self.truth_summary()}')

    def destroy_node(self):
        self.f.flush()
        self.f.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EvalLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.get_logger().info(f'Saved {node.rows} rows to {node.path} | {node.truth_summary()}')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
