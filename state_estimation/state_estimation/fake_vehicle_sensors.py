#!/usr/bin/env python3
"""SIMULATION ONLY: pretend to be the car's STM32.

On the real car, the STM32 will report the measured wheel speed (encoder) and
the steering angle. The simulator doesn't publish those, so this node makes
them from the simulator's ground truth and adds realistic errors:

  speed    = true_speed * (1 + speed_scale_error) + noise
  steering = commanded_steering + steering_bias + noise

Output: /vehicle/measured (ackermann_msgs/AckermannDriveStamped) at 50 Hz.
On the real car this node is replaced by the STM32 bridge publishing the
same topic, so everything downstream stays the same.
"""
import math
import random

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class FakeVehicleSensors(Node):

    def __init__(self):
        super().__init__('fake_vehicle_sensors')
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('speed_scale_error', 0.03)  # 3 %: wrong wheel radius
        self.declare_parameter('speed_noise', 0.02)        # m/s
        self.declare_parameter('steering_bias', 0.01)      # rad (~0.6 deg): servo trim off
        self.declare_parameter('steering_noise', 0.005)    # rad
        self.declare_parameter('teleop_steer', 0.3)        # rad, what the sim uses for /cmd_vel
        self.declare_parameter('max_steer', 0.4189)        # rad, sim steering limit

        self.true_speed = 0.0
        self.cmd_steer = 0.0

        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_cb, 10)
        self.create_subscription(Twist, '/cmd_vel', self.teleop_cb, 10)
        self.create_subscription(AckermannDriveStamped, '/drive', self.drive_cb, 10)
        self.pub = self.create_publisher(AckermannDriveStamped, '/vehicle/measured', 10)
        self.create_timer(1.0 / self.get_parameter('rate_hz').value, self.publish)
        self.get_logger().info('Publishing simulated wheel speed + steering on /vehicle/measured')

    def odom_cb(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.true_speed = math.copysign(math.hypot(vx, vy), vx)

    def teleop_cb(self, msg):
        # Same rule the simulator bridge uses for keyboard teleop
        s = self.get_parameter('teleop_steer').value
        self.cmd_steer = s if msg.angular.z > 0 else (-s if msg.angular.z < 0 else 0.0)

    def drive_cb(self, msg):
        m = self.get_parameter('max_steer').value
        self.cmd_steer = max(-m, min(m, msg.drive.steering_angle))

    def publish(self):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'ego_racecar/base_link'
        speed = self.true_speed * (1.0 + p('speed_scale_error'))
        if abs(self.true_speed) > 1e-3:          # a stopped wheel reads zero
            speed += random.gauss(0.0, p('speed_noise'))
        out.drive.speed = speed
        out.drive.steering_angle = (self.cmd_steer + p('steering_bias')
                                    + random.gauss(0.0, p('steering_noise')))
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = FakeVehicleSensors()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
