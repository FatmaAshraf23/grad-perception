#!/usr/bin/env python3
"""ROS-level test of the WHEEL-DATA mode rules in StateEstimateNode (contract v0.4, Fix 4; agreed by control 2026-11-02):
  - no wheel sample for > wheel_stale_s (50 ms)  -> FLAG_WHEEL_STALE + at least DEGRADED
  - no wheel sample for > wheel_lost_s  (1.0 s)  -> LOST
  - no standstill exception: a PARKED car with missing wheel data is DEGRADED / LOST too
  - never any wheel data (after the first fix) -> LOST, not "parked"
  - INIT still wins before the first fix; data back -> OK again
Calls the node's callbacks directly, as state_pipeline does, and captures the published messages.
Wheel silence is mostly simulated by moving last_wheel_t back (fast, exact); one case uses the real clock.
Needs rclpy + the built workspace. Run: python3 test_wheel_stale_node.py"""
import math
import sys
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from apex_msgs.msg import StateEstimate as SE
from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64, Float64MultiArray

import state_estimation.state_estimate_node as sen
from state_estimation.state_estimate_node import StateEstimateNode

WS_FLAG = 4
RESULTS = []


class Capture:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


def check(name, cond, m=None):
    RESULTS.append(bool(cond))
    info = '' if m is None else f' (mode {m.mode}, flags {m.flags})'
    print(f"{'PASS' if cond else 'FAIL'}  {name}{info}")


def odom(node, v=0.0, std=0.05):
    o = Odometry()
    o.header.stamp = Time(nanoseconds=node.get_clock().now().nanoseconds - 2_000_000).to_msg()
    o.header.frame_id = 'map'
    o.pose.pose.orientation.w = 1.0
    o.twist.twist.linear.x = float(v)
    cov = [0.0] * 36
    cov[0] = cov[7] = std ** 2
    cov[35] = math.radians(0.5) ** 2
    o.pose.covariance = cov
    return o


def map_check(alarm):
    m = Float64MultiArray()
    m.data = [0.0, 0.0, 0.0, 0.2, 0.97, 180.0, 1.0 if alarm else 0.0, 0.5, 1.0, 3.0, 2.0 if alarm else 1.0]
    return m


def is_(m, mode, flag):
    return m.mode == mode and bool(m.flags & WS_FLAG) == flag


def main():
    print(f'module under test: {sen.__file__}')
    rclpy.init()
    node = StateEstimateNode(own_inputs=False)
    cap = Capture()
    node.pub = cap

    def step(wheel_age=None, v=0.0, fix=False):
        """wheel_age None = a fresh wheel sample now; a number = the last one was that many s ago; 'never' = none yet."""
        node.imu_cb(Imu())
        if fix:
            node.fix_cb(Float64(data=1.0))
        if wheel_age is None:
            node.wheel_cb(AckermannDriveStamped())
        elif wheel_age == 'never':
            node.last_wheel_t = None
        else:
            node.last_wheel_t = node.now_s() - wheel_age
        node.odom_cb(odom(node, v=v))
        return cap.msgs[-1]

    def setp(name, value):
        node.set_parameters([Parameter(name, value=value)])

    p = lambda n: node.get_parameter(n).value  # noqa: E731
    check('constant FLAG_WHEEL_STALE == 4', SE.FLAG_WHEEL_STALE == WS_FLAG)
    check("defaults: use_wheel_stale true, wheel_stale_s 0.05, wheel_lost_s 1.0",
          p('use_wheel_stale') is True and p('wheel_stale_s') == 0.05 and p('wheel_lost_s') == 1.0)

    # --- start-up
    m = step(wheel_age='never')
    check('before the first fix, no wheel data ever: INIT wins (INIT > LOST)', m.mode == SE.MODE_INIT, m)
    m = step(wheel_age='never', fix=True)
    check('first fix but NEVER any wheel data: LOST + flag (missing is not "parked")', is_(m, SE.MODE_LOST, True), m)
    m = step()
    check('wheel data arrives: OK, no flag', is_(m, SE.MODE_OK, False), m)

    # --- the 50 ms DEGRADED rule
    m = step(wheel_age=0.03)
    check('wheel 30 ms old (< 50 ms): OK, no flag', is_(m, SE.MODE_OK, False), m)
    m = step(wheel_age=0.07)
    check('PARKED, wheel 70 ms old: DEGRADED + flag (no standstill exception)', is_(m, SE.MODE_DEGRADED, True), m)
    m = step(wheel_age=0.07, v=3.0, fix=True)
    check('DRIVING 3 m/s, wheel 70 ms old: DEGRADED + flag', is_(m, SE.MODE_DEGRADED, True), m)

    # --- the 1.0 s LOST rule
    m = step(wheel_age=0.95)
    check('PARKED, wheel 0.95 s old: still DEGRADED, not LOST yet', is_(m, SE.MODE_DEGRADED, True), m)
    m = step(wheel_age=1.05)
    check('PARKED, wheel 1.05 s old: LOST + flag', is_(m, SE.MODE_LOST, True), m)
    m = step(wheel_age=1.05, v=3.0, fix=True)
    check('DRIVING 3 m/s, wheel 1.05 s old: LOST + flag', is_(m, SE.MODE_LOST, True), m)
    m = step()
    check('wheel data back: OK, no flag', is_(m, SE.MODE_OK, False), m)

    # --- the node's own clock (no back-dating): real silence
    node.wheel_cb(AckermannDriveStamped())
    time.sleep(0.08)
    node.imu_cb(Imu())
    node.odom_cb(odom(node))
    m = cap.msgs[-1]
    check('REAL CLOCK: 80 ms of silence -> DEGRADED + flag', is_(m, SE.MODE_DEGRADED, True), m)
    time.sleep(1.0)
    node.imu_cb(Imu())
    node.odom_cb(odom(node))
    m = cap.msgs[-1]
    check('REAL CLOCK: 1.08 s of silence -> LOST + flag', is_(m, SE.MODE_LOST, True), m)
    m = step()
    check('wheel data back: OK', is_(m, SE.MODE_OK, False), m)

    # --- priority + parameters
    node.map_check_cb(map_check(True))
    m = step(wheel_age=1.05)
    check('wheel 1.05 s old + map-mismatch alarm: LOST wins over DEGRADED', is_(m, SE.MODE_LOST, True), m)
    node.map_check_cb(map_check(False))
    setp('wheel_lost_s', 0.5)
    m = step(wheel_age=0.6)
    check('wheel_lost_s 0.5 (car.yaml can tune it): 0.6 s -> LOST', is_(m, SE.MODE_LOST, True), m)
    setp('wheel_lost_s', 1.0)
    setp('wheel_stale_s', 0.03)
    m = step(wheel_age=0.04)
    check('wheel_stale_s 0.03 (real-car value): 40 ms -> DEGRADED', is_(m, SE.MODE_DEGRADED, True), m)
    setp('wheel_stale_s', 0.05)
    setp('use_wheel_stale', False)
    m = step(wheel_age=1.05)
    check('use_wheel_stale false (old behaviour): flag only, mode OK', is_(m, SE.MODE_OK, True), m)
    setp('use_wheel_stale', True)
    m = step()
    check('back to defaults, fresh data: OK', is_(m, SE.MODE_OK, False), m)

    node.destroy_node()
    rclpy.try_shutdown()
    print(f'\n{sum(RESULTS)} of {len(RESULTS)} passed')
    return 0 if all(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
