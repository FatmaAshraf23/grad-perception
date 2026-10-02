#!/usr/bin/env python3
"""Diagnostic: does the simulator's reported yaw rate match its own heading?

Uses ONLY ground truth from /ego_racecar/odom (no noise, no bias):
  * heading A = yaw from the odom quaternion (the true pose)
  * heading B = integral of odom twist.angular.z (the true yaw rate)
If the simulator is self-consistent, A and B stay equal. The fake IMU is
built from twist.angular.z, so any gap between A and B shows up as
"gyro heading error" even with a perfect gyro.

Run (no build needed):  python3 check_sim_yawrate.py
Prints a line every second and a WARNING whenever the gap jumps by more
than 1 deg within one second, with the speed and yaw rate at that moment.
"""
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class YawRateCheck(Node):

    def __init__(self):
        super().__init__('check_sim_yawrate')
        self.integrated = None
        self.last_t = None
        self.last_gap = 0.0
        self.odom = None
        self.min_speed = float('inf')
        self.max_rate = 0.0
        self.create_subscription(Odometry, '/ego_racecar/odom', self.cb, 100)
        self.create_timer(1.0, self.report)
        self.get_logger().info('Comparing odom heading with integrated odom yaw rate ...')

    def cb(self, msg):
        t = Time.from_msg(msg.header.stamp)
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        if self.integrated is None:
            self.integrated, self.last_t = yaw, t
            return
        dt = (t - self.last_t).nanoseconds * 1e-9
        self.last_t = t
        if 0.0 < dt < 0.5:
            self.integrated = wrap(self.integrated + msg.twist.twist.angular.z * dt)
        self.odom = msg
        self.min_speed = min(self.min_speed, abs(msg.twist.twist.linear.x))
        self.max_rate = max(self.max_rate, abs(msg.twist.twist.angular.z))

    def report(self):
        if self.odom is None:
            return
        gap = math.degrees(wrap(self.integrated - yaw_from_quat(self.odom.pose.pose.orientation)))
        line = (f'gap {gap:+7.2f} deg | speed {self.odom.twist.twist.linear.x:5.2f} m/s '
                f'(min {self.min_speed:4.2f}) | max |yaw rate| {self.max_rate:5.2f} rad/s')
        if abs(gap - self.last_gap) > 1.0:
            self.get_logger().warn('JUMP ' + line)
        else:
            self.get_logger().info(line)
        self.last_gap = gap
        self.min_speed = float('inf')
        self.max_rate = 0.0


def main():
    rclpy.init()
    node = YawRateCheck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
