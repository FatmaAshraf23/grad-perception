#!/usr/bin/env python3
"""ROS-level test of the LOW-SPEED LiDAR-stale rule in StateEstimateNode (lidar_stale_m, 2026-11-15):
  - lidar_stale_m 0.0 = the old time rule: > 0.30 s of driving without an accepted fix -> FLAG_LIDAR_STALE + DEGRADED
  - lidar_stale_m 0.10 (default since 2026-11-15): stale only when ALSO > 0.10 m were driven since the fix
      0.15 m/s: 0.40 s (0.06 m) is still OK; stale once 0.10 m are passed (0.67 s)
      6 m/s: unchanged, stale after 0.30 s
  - a fix resets time and distance; standing still adds neither; LOST still needs stale AND > 1.0 s of driving
Calls the node's callbacks directly, as state_pipeline does. Driving = EKF messages 10 ms apart (their stamps).
Needs rclpy + the built workspace. Run: python3 test_lidar_stale_dist_node.py"""
import math
import sys

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from apex_msgs.msg import StateEstimate as SE
from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

import state_estimation.state_estimate_node as sen
from state_estimation.state_estimate_node import StateEstimateNode

RESULTS = []
DT_NS = 10_000_000


class Capture:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


def check(name, cond, m=None):
    RESULTS.append(bool(cond))
    info = '' if m is None else f' (mode {m.mode}, flags {m.flags})'
    print(f"{'PASS' if cond else 'FAIL'}  {name}{info}")


def main():
    print(f'module under test: {sen.__file__}')
    rclpy.init()
    node = StateEstimateNode(own_inputs=False)
    cap = Capture()
    node.pub = cap
    node.set_parameters([Parameter('lag_check', value=False)])     # the stamps below are synthetic (not "now")
    t_ns = [node.get_clock().now().nanoseconds - 600_000_000_000]

    def step(v):
        t_ns[0] += DT_NS
        o = Odometry()
        o.header.stamp = Time(nanoseconds=t_ns[0]).to_msg()
        o.header.frame_id = 'map'
        o.pose.pose.orientation.w = 1.0
        o.twist.twist.linear.x = float(v)
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.05 ** 2
        cov[35] = math.radians(0.5) ** 2
        o.pose.covariance = cov
        node.imu_cb(Imu())
        node.wheel_cb(AckermannDriveStamped())
        node.odom_cb(o)
        return cap.msgs[-1]

    def drive(v, seconds):
        m = None
        for _ in range(int(round(seconds * 1e9 / DT_NS))):
            m = step(v)
        return m

    def fix():
        node.fix_cb(Float64(data=1.0))

    def setp(name, value):
        node.set_parameters([Parameter(name, value=value)])

    def is_(m, mode, stale):
        return m.mode == mode and bool(m.flags & SE.FLAG_LIDAR_STALE) == stale

    p = lambda n: node.get_parameter(n).value  # noqa: E731
    check('defaults: lidar_stale_s 0.3, lidar_lost_s 1.0, lidar_stale_m 0.10',
          p('lidar_stale_s') == 0.3 and p('lidar_lost_s') == 1.0 and p('lidar_stale_m') == 0.10)
    m = step(0.0)
    check('before the first fix: INIT', m.mode == SE.MODE_INIT, m)
    fix()
    m = step(0.0)
    check('first fix, parked: OK', is_(m, SE.MODE_OK, False), m)

    # --- lidar_stale_m 0.0: the time rule, as before
    setp('lidar_stale_m', 0.0)
    m = drive(0.15, 0.28)
    check('TIME RULE 0.15 m/s, 0.28 s without a fix: OK', is_(m, SE.MODE_OK, False), m)
    m = drive(0.15, 0.05)
    check('TIME RULE 0.15 m/s, 0.33 s (0.05 m): DEGRADED + LIDAR_STALE', is_(m, SE.MODE_DEGRADED, True), m)
    fix()
    m = step(0.15)
    check('a fix: OK again', is_(m, SE.MODE_OK, False), m)

    # --- lidar_stale_m 0.10 at low speed
    setp('lidar_stale_m', 0.10)
    fix()
    m = drive(0.15, 0.43)
    check('RULE C 0.15 m/s, 0.43 s (0.065 m = AMCL needs 3 scans): OK, no flag', is_(m, SE.MODE_OK, False), m)
    m = drive(0.15, 0.20)
    check('RULE C 0.15 m/s, 0.63 s (0.095 m): still OK', is_(m, SE.MODE_OK, False), m)
    m = drive(0.15, 0.07)
    check('RULE C 0.15 m/s, 0.70 s (0.105 m): DEGRADED + LIDAR_STALE', is_(m, SE.MODE_DEGRADED, True), m)
    m = drive(0.15, 0.40)
    check('RULE C 0.15 m/s, 1.10 s (0.165 m): LOST (stale and > 1.0 s)', is_(m, SE.MODE_LOST, True), m)
    fix()
    m = step(0.15)
    check('a fix resets time AND distance: OK', is_(m, SE.MODE_OK, False) and node.dist_since_fix < 0.01, m)
    m = drive(0.30, 0.31)
    check('RULE C 0.30 m/s, 0.32 s (0.095 m = one late scan): OK, no flag', is_(m, SE.MODE_OK, False), m)

    # --- standing still adds neither time nor distance
    fix()
    drive(0.15, 0.20)
    m = drive(0.0, 3.0)
    check('RULE C parked 3 s after 0.20 s of driving: OK', is_(m, SE.MODE_OK, False), m)
    m = drive(0.15, 0.20)
    check('RULE C sets off again, 0.40 s driven in total (0.06 m): OK', is_(m, SE.MODE_OK, False), m)

    # --- racing speed: identical to the time rule
    fix()
    m = drive(6.0, 0.28)
    check('RULE C 6 m/s, 0.28 s without a fix: OK', is_(m, SE.MODE_OK, False), m)
    m = drive(6.0, 0.04)
    check('RULE C 6 m/s, 0.32 s: DEGRADED + LIDAR_STALE (as the time rule)', is_(m, SE.MODE_DEGRADED, True), m)
    m = drive(6.0, 0.72)
    check('RULE C 6 m/s, 1.04 s: LOST', is_(m, SE.MODE_LOST, True), m)
    fix()
    m = step(6.0)
    check('a fix: OK', is_(m, SE.MODE_OK, False), m)
    fix()
    m = drive(0.40, 0.32)
    check('RULE C 0.40 m/s, 0.32 s (0.128 m): DEGRADED + LIDAR_STALE (above 0.33 m/s = the time rule)',
          is_(m, SE.MODE_DEGRADED, True), m)

    # --- back to 0.0
    setp('lidar_stale_m', 0.0)
    fix()
    m = drive(0.15, 0.33)
    check('lidar_stale_m back to 0.0: 0.15 m/s, 0.33 s -> DEGRADED + LIDAR_STALE again', is_(m, SE.MODE_DEGRADED, True), m)
    setp('lidar_stale_m', 0.10)
    fix()
    m = drive(0.15, 0.33)
    check('lidar_stale_m 0.10 again: 0.15 m/s, 0.33 s -> OK', is_(m, SE.MODE_OK, False), m)

    node.destroy_node()
    rclpy.try_shutdown()
    print(f'\n{sum(RESULTS)} of {len(RESULTS)} passed')
    return 0 if all(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
