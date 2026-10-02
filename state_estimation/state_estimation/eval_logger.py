#!/usr/bin/env python3
"""Record everything needed to evaluate the localization, 10 rows per second.

One CSV row every 0.1 s with the latest value of each source:
  t, s                      time since start [s], distance driven (truth) [m]
  tx, ty, tyaw              simulator ground truth
  ex, ey, eyaw, e_sig, e_sig_yaw   global EKF (/odom/ekf) and its 1-sigma
                                   (e_sig = sqrt(Pxx + Pyy))
  wx, wy, wyaw              wheel-only odometry (/odom/wheel)
  a_err, a_err_yaw          AMCL error vs truth AT THE SCAN TIME (latest pose)
  k, bias, true_bias        EKF speed scale, EKF gyro bias, simulated true bias
  v                         true speed
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
  tr_r                      true yaw rate from the change of the true heading over +-20 ms
                            (the simulator's twist.angular.z can stay stuck after stops)
File: ~/eval_logs/<run_name>_<YYYYmmdd_HHMMSS>.csv  (plot with plot_eval.py)
"""
import math
import os
import time
from collections import deque

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
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


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


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
        self.truth_hist = deque(maxlen=1500)         # (t_ns, x, y, yaw, vx, vy), a few s
        self.last_xy = None
        self.s = 0.0
        self.ekf = None
        self.wheel = None
        self.amcl = (float('nan'), float('nan'))
        self.k = self.bias = self.true_bias = float('nan')
        self.t0 = None
        self.rows = 0
        self.se = None                               # newest StateEstimate
        self.se_rx_ns = None                         # when it arrived here
        ty = self.get_parameter('track_yaml').value or (
            get_package_share_directory('apex_track')
            + f"/tracks/{self.get_parameter('track').value}/track.yaml")
        self.track = load_track(ty)                  # own lap counter for the truth
        self.track_first = True

        self.create_subscription(Odometry, '/ego_racecar/odom', self.truth_cb, 10)
        self.create_subscription(Odometry, '/odom/ekf', lambda m: setattr(self, 'ekf', m), 10)
        self.create_subscription(Odometry, '/odom/wheel', lambda m: setattr(self, 'wheel', m), 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_cb, 10)
        self.create_subscription(Float64, '/ekf/speed_scale', lambda m: setattr(self, 'k', m.data), 10)
        self.create_subscription(Float64, '/ekf/bias', lambda m: setattr(self, 'bias', m.data), 10)
        self.create_subscription(StateEstimate, '/state_estimate', self.se_cb, 50)
        self.create_subscription(Float64, '/sim/true_gyro_bias',
                                 lambda m: setattr(self, 'true_bias', m.data), 10)
        self.create_timer(1.0 / self.get_parameter('rate_hz').value, self.write_row)
        self.create_timer(10.0, self.report)
        self.get_logger().info(f'Logging to {self.path}')

    def truth_cb(self, msg):
        pp = msg.pose.pose
        x, y = pp.position.x, pp.position.y
        if self.last_xy is not None:
            step = math.hypot(x - self.last_xy[0], y - self.last_xy[1])
            if step < 0.5:                           # ignore teleports
                self.s += step
        self.last_xy = (x, y)
        self.truth = msg
        tw = msg.twist.twist.linear
        self.truth_hist.append((Time.from_msg(msg.header.stamp).nanoseconds,
                                x, y, yaw_from_quat(pp.orientation), tw.x, tw.y))

    def amcl_cb(self, msg):
        if not self.truth_hist:
            return
        t = Time.from_msg(msg.header.stamp).nanoseconds
        _, tx, ty, tyaw, _, _ = min(self.truth_hist, key=lambda e: abs(e[0] - t))
        pp = msg.pose.pose
        self.amcl = (math.hypot(pp.position.x - tx, pp.position.y - ty),
                     math.degrees(wrap(yaw_from_quat(pp.orientation) - tyaw)))

    def se_cb(self, msg):
        self.se = msg
        self.se_rx_ns = self.get_clock().now().nanoseconds

    def truth_at(self, t_ns):
        """Truth sample nearest to t_ns, and the true yaw rate over +-20 ms around it."""
        hist = self.truth_hist
        near = min(hist, key=lambda e: abs(e[0] - t_ns))
        a = min(hist, key=lambda e: abs(e[0] - (t_ns - 20_000_000)))
        b = min(hist, key=lambda e: abs(e[0] - (t_ns + 20_000_000)))
        dt = (b[0] - a[0]) * 1e-9
        r = wrap(b[3] - a[3]) / dt if dt > 0.005 else float('nan')
        return near, r

    def se_columns(self):
        nan = float('nan')
        if self.se is None or not self.truth_hist:
            return [nan] * len(SE_COLS)
        m = self.se
        t_ns = m.stamp_us * 1000
        (_, tx, ty, tyaw, tvx, tvy), tr_r = self.truth_at(t_ns)
        tr_s, s_tr, tr_ey, tr_epsi, tr_k = self.track.project(tx, ty, tyaw)
        if self.track_first and s_tr > self.track.L - 2.0:   # same rule as the node:
            self.track.lap = -1                              # started just behind the line
            tr_s -= self.track.L
        self.track_first = False
        age = (self.se_rx_ns / 1000.0 - m.stamp_us) / 1000.0 if self.se_rx_ns else nan
        return [m.s_abs_m, m.e_y_m, m.e_psi_rad, m.vx_mps, m.vy_mps, m.r_radps,
                m.std_e_y_m, m.std_e_psi_rad, float(m.mode), float(m.flags),
                (m.publish_us - m.stamp_us) / 1000.0, age, float(m.sequence_id),
                tr_s, tr_ey, tr_epsi, tvx, tvy, tr_r, tr_k]

    def write_row(self):
        if self.truth is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        self.t0 = self.t0 or now
        tp = self.truth.pose.pose
        nan = float('nan')
        ex = ey = eyaw = esig = esigy = nan
        if self.ekf is not None:
            ep = self.ekf.pose
            ex, ey, eyaw = ep.pose.position.x, ep.pose.position.y, yaw_from_quat(ep.pose.orientation)
            c = ep.covariance
            esig = math.sqrt(max(c[0] + c[7], 0.0))
            esigy = math.sqrt(max(c[35], 0.0))
        wx = wy = wyaw = nan
        if self.wheel is not None:
            wp = self.wheel.pose.pose
            wx, wy, wyaw = wp.position.x, wp.position.y, yaw_from_quat(wp.orientation)
        v = math.hypot(self.truth.twist.twist.linear.x, self.truth.twist.twist.linear.y)
        row = [now - self.t0, self.s, tp.position.x, tp.position.y, yaw_from_quat(tp.orientation),
               ex, ey, eyaw, esig, esigy, wx, wy, wyaw, self.amcl[0], self.amcl[1],
               self.k, self.bias, self.true_bias, v] + self.se_columns()
        self.f.write(','.join(f'{x:.5f}' for x in row) + '\n')
        self.rows += 1
        if self.rows % 50 == 0:
            self.f.flush()

    def report(self):
        self.get_logger().info(f'{self.rows} rows, {self.s:.1f} m driven -> {self.path}')

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
    node.get_logger().info(f'Saved {node.rows} rows to {node.path}')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
