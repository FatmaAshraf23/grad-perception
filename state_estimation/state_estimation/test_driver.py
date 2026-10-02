#!/usr/bin/env python3
"""SIMULATION TEST HARNESS: drive the car around the track automatically.

Why: driving laps by hand (teleop + a lagging Foxglove) is slow and never the
same twice. For evaluating the localization we want the same laps every run.

How: pure pursuit on the pre-computed centre line of the Levine loop
(config/levine_centerline.csv, 0.2 m spacing, >= 0.74 m from every wall).

IMPORTANT -- this node uses the simulator's TRUE pose (/ego_racecar/odom) ON
PURPOSE. It is a test harness, not part of the car's software: if it steered
with the EKF, a localization error would change the path it drives and the
test would partly measure itself. Planning/MPC on the real car will use
/odom/ekf instead.

Pure pursuit (one control step, 50 Hz):
  1. find the waypoint nearest to the car, then the "goal" waypoint one
     look-ahead distance Ld further along the loop
  2. angle to the goal in the car frame:   alpha
  3. steering for an arc through the goal:  delta = atan(2 L sin(alpha) / Ld)
  4. speed from the curvature of the next ~2 m:  v = min(v_max, sqrt(a_lat / kappa))

Sequence: wait `start_delay` s (still -> ZUPT calibration), drive `laps` laps,
stop and stay stopped (final ZUPTs). If the car stops moving while commanded
to drive (collision), it stops and reports.
Publishes /drive (ackermann_msgs/AckermannDriveStamped). Don't run teleop at
the same time.
"""
import csv
import math

import numpy as np


def load_waypoints(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith('x')]
    return np.array([[float(r[0]), float(r[1])] for r in rows])


def loop_curvature(wp):
    """Curvature at each waypoint from three neighbouring points (1/m)."""
    p0, p1, p2 = np.roll(wp, 2, axis=0), wp, np.roll(wp, -2, axis=0)
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    cross = np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                   - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
    return 2.0 * cross / np.maximum(a * b * c, 1e-9)


def speed_profile(wp, v_max, v_min, a_lat, preview_m, spacing):
    """Max speed at each waypoint, looking `preview_m` ahead (brake before corners)."""
    kappa = loop_curvature(wp)
    v_curve = np.clip(np.sqrt(a_lat / np.maximum(kappa, 1e-6)), v_min, v_max)
    n, k = len(wp), max(1, int(preview_m / spacing))
    return np.array([min(v_curve[(i + j) % n] for j in range(k)) for i in range(n)])


def pure_pursuit(x, y, yaw, wp, idx_hint, lookahead, wheelbase, max_steer, search=40):
    """Returns (steering angle, index of nearest waypoint)."""
    n = len(wp)
    cand = [(idx_hint + j) % n for j in range(-5, search)]
    d = [math.hypot(wp[i, 0] - x, wp[i, 1] - y) for i in cand]
    idx = cand[int(np.argmin(d))]
    g = idx
    while math.hypot(wp[g, 0] - x, wp[g, 1] - y) < lookahead:
        g = (g + 1) % n
        if g == idx:
            break
    dx, dy = wp[g, 0] - x, wp[g, 1] - y
    alpha = math.atan2(dy, dx) - yaw
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))
    ld = max(math.hypot(dx, dy), 1e-3)
    steer = math.atan(2.0 * wheelbase * math.sin(alpha) / ld)
    return max(-max_steer, min(max_steer, steer)), idx


# ----------------------------------------------------------------------------
# ROS 2 node
# ----------------------------------------------------------------------------

try:
    import os

    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from ament_index_python.packages import get_package_share_directory
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
except ImportError:
    Node = object


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class TestDriver(Node):

    def __init__(self):
        super().__init__('test_driver')
        d = self.declare_parameter
        d('waypoints', '')                  # '' -> config/levine_centerline.csv
        d('laps', 3)
        d('v_max', 1.5)                     # m/s on straights
        d('v_min', 0.7)                     # m/s in the tightest corners
        d('a_lat', 1.5)                     # m/s^2 allowed sideways acceleration
        d('lookahead', 0.8)                 # m (+ 0.2 s * speed)
        d('start_delay', 3.0)               # s standing still before driving
        d('wheelbase', 0.3302)
        d('max_steer', 0.4189)
        d('rate_hz', 50.0)

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        path = p('waypoints') or os.path.join(
            get_package_share_directory('state_estimation'), 'config', 'levine_centerline.csv')
        self.wp = load_waypoints(path)
        spacing = float(np.mean(np.linalg.norm(np.diff(self.wp, axis=0), axis=1)))
        self.loop_len = spacing * len(self.wp)
        self.v_ref = speed_profile(self.wp, p('v_max'), p('v_min'), p('a_lat'), 2.0, spacing)

        self.pose = None
        self.speed = 0.0
        self.idx = None
        self.progress = 0.0                 # waypoints passed (unwrapped)
        self.t_start = None
        self.done = False
        self.stuck_since = None

        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_cb, 10)
        self.pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.create_timer(1.0 / p('rate_hz'), self.step)
        self.get_logger().info(
            f'Test driver: {len(self.wp)} waypoints ({self.loop_len:.1f} m loop) from {path}; '
            f'{p("laps")} laps, v_max {p("v_max")} m/s, start in {p("start_delay")} s')

    def odom_cb(self, msg):
        pp = msg.pose.pose
        self.pose = (pp.position.x, pp.position.y, yaw_from_quat(pp.orientation))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def send(self, v, steer):
        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.drive.speed = float(v)
        m.drive.steering_angle = float(steer)
        self.pub.publish(m)

    def step(self):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.pose is None:
            return
        if self.t_start is None:
            self.t_start = now
            x, y, _ = self.pose
            dmin = float(np.min(np.hypot(self.wp[:, 0] - x, self.wp[:, 1] - y)))
            self.idx = int(np.argmin(np.hypot(self.wp[:, 0] - x, self.wp[:, 1] - y)))
            if dmin > 0.5:
                self.get_logger().warn(
                    f'Car is {dmin:.2f} m from the centre line -- wrong map or start pose? '
                    'Check sim.yaml (levine, sx=sy=stheta=0).')
        if self.done or now - self.t_start < p('start_delay'):
            self.send(0.0, 0.0)
            return

        x, y, yaw = self.pose
        v_cmd = float(self.v_ref[self.idx])
        steer, new_idx = pure_pursuit(x, y, yaw, self.wp, self.idx,
                                      p('lookahead') + 0.2 * self.speed,
                                      p('wheelbase'), p('max_steer'))
        n = len(self.wp)
        self.progress += (new_idx - self.idx) % n if (new_idx - self.idx) % n < n // 2 else 0
        self.idx = new_idx
        laps_done = self.progress / n

        # Collision / stuck detection
        if self.speed < 0.05 and now - self.t_start > p('start_delay') + 2.0:
            self.stuck_since = self.stuck_since or now
            if now - self.stuck_since > 1.5:
                self.get_logger().error('Car is not moving although commanded -- collision? Stopping.')
                self.done = True
        else:
            self.stuck_since = None

        if laps_done >= p('laps'):
            self.get_logger().info(f'{p("laps")} laps done ({self.progress * self.loop_len / n:.1f} m). '
                                   'Stopping (car stays still, ZUPTs follow).')
            self.done = True
            self.send(0.0, 0.0)
            return
        self.send(v_cmd, steer)


def main(args=None):
    rclpy.init(args=args)
    node = TestDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
