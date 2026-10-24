#!/usr/bin/env python3
"""ROS-level test of the contract v0.3 health flags in StateEstimateNode (2026-10-24).
Needs rclpy and the built workspace (apex_msgs with the v0.3 flags, apex_track). Calls the node's callbacks
directly, as state_pipeline does, and captures the published messages instead of sending them.
Run: python3 test_v03_node.py"""
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

from state_estimation.state_estimate_node import StateEstimateNode

MM, LAG = 1024, 2048
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


def odom(node, std=0.05, age_s=0.002):
    o = Odometry()
    o.header.stamp = Time(nanoseconds=node.get_clock().now().nanoseconds - int(age_s * 1e9)).to_msg()
    o.header.frame_id = 'map'
    o.pose.pose.orientation.w = 1.0
    cov = [0.0] * 36
    cov[0] = cov[7] = std ** 2
    cov[35] = math.radians(0.5) ** 2
    o.pose.covariance = cov
    return o


def map_check(alarm):
    m = Float64MultiArray()
    m.data = [0.0, 0.0, 0.0, 0.2, 0.97, 180.0, 1.0 if alarm else 0.0, 0.5, 1.0, 3.0, 2.0 if alarm else 1.0]
    return m


def main():
    rclpy.init()
    node = StateEstimateNode(own_inputs=False)
    cap = Capture()
    node.pub = cap

    def step(**kw):
        node.imu_cb(Imu())
        node.wheel_cb(AckermannDriveStamped())
        node.odom_cb(odom(node, **kw))
        return cap.msgs[-1]

    def setp(name, value):
        node.set_parameters([Parameter(name, value=value)])

    check('message constants FLAG_MAP_MISMATCH 1024, FLAG_ESTIMATOR_LAG 2048',
          SE.FLAG_MAP_MISMATCH == MM and SE.FLAG_ESTIMATOR_LAG == LAG)
    node.map_check_cb(map_check(True))
    m = step()
    check('before the first fix: INIT, flag set anyway', m.mode == SE.MODE_INIT and m.flags & MM, m)
    node.map_check_cb(map_check(False))
    node.fix_cb(Float64(data=1.0))
    m = step()
    check('first fix, small sigma: OK, no v0.3 flags', m.mode == SE.MODE_OK and not m.flags & (MM | LAG), m)
    check("schema 'apex-state-estimate/1.1', valid_for_ms 50",
          m.schema_version == 'apex-state-estimate/1.1' and m.valid_for_ms == 50)
    node.map_check_cb(map_check(True))
    m = step()
    check('map-check alarm -> DEGRADED + FLAG_MAP_MISMATCH', m.mode == SE.MODE_DEGRADED and m.flags & MM, m)
    m = step(std=0.5)
    check('alarm + std_xy 0.5 m -> LOST wins, flag still set', m.mode == SE.MODE_LOST and m.flags & MM, m)
    node.map_check_cb(map_check(False))
    m = step()
    check('alarm off -> OK again', m.mode == SE.MODE_OK and not m.flags & MM, m)
    setp('use_map_check', False)
    node.map_check_cb(map_check(True))
    m = step()
    check('use_map_check false (shadow): alarm ignored', m.mode == SE.MODE_OK and not m.flags & MM, m)
    setp('use_map_check', True)
    m = step()
    check('use_map_check true again: DEGRADED + flag', m.mode == SE.MODE_DEGRADED and m.flags & MM, m)
    node.map_check_cb(map_check(False))
    m1 = step(age_s=0.1)
    check('one estimate 100 ms old: no lag flag yet', m1.mode == SE.MODE_OK and not m1.flags & LAG, m1)
    m2 = step(age_s=0.1)
    check('second one in a row: DEGRADED + FLAG_ESTIMATOR_LAG', m2.mode == SE.MODE_DEGRADED and m2.flags & LAG, m2)
    m3 = step()
    check('young again: flag held (needs 1 s below 10 ms)', m3.mode == SE.MODE_DEGRADED and m3.flags & LAG, m3)
    time.sleep(1.05)
    m4 = step()
    check('1 s later: flag off, OK', m4.mode == SE.MODE_OK and not m4.flags & LAG, m4)
    setp('lag_check', False)
    step(age_s=0.1)
    m5 = step(age_s=0.1)
    check('lag_check false: no flag', m5.mode == SE.MODE_OK and not m5.flags & LAG, m5)
    node.destroy_node()
    rclpy.try_shutdown()
    print(f'\n{sum(RESULTS)} of {len(RESULTS)} passed')
    return 0 if all(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
