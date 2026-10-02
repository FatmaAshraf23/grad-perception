#!/usr/bin/env python3
"""SIMULATION ONLY: give AMCL a LiDAR scan attached to the ESTIMATED car.

Problem: the simulator publishes the TRUE pose as TF  map -> ego_racecar/base_link,
and the scan is in frame ego_racecar/laser (a child of the true car). AMCL needs
the chain  map -> odom -> <base> -> <laser>  where odom -> <base> comes from our
local EKF. A frame can't have two parents, so the estimate gets its own frames:

    map --(AMCL)--> odom --(ekf_local)--> est/base_link --(static)--> est/laser

This node
  1. reads the LiDAR mounting offset once from TF (ego_racecar/base_link -> laser),
  2. publishes the same offset as a static TF  est/base_link -> est/laser,
  3. republishes every scan from /scan_a1 on /scan_amcl with frame_id est/laser.
The measurements themselves are untouched.

On the real car this node is not needed: the LiDAR driver publishes the scan in
'laser', and a static TF base_link -> laser comes from the car description.
"""
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class SimScanRelay(Node):

    def __init__(self):
        super().__init__('sim_scan_relay')
        self.declare_parameter('in_topic', '/scan_a1')
        self.declare_parameter('out_topic', '/scan_amcl')
        self.declare_parameter('truth_base_frame', 'ego_racecar/base_link')
        self.declare_parameter('est_base_frame', 'est/base_link')
        self.declare_parameter('est_laser_frame', 'est/laser')
        # Fallback if the TF lookup fails (sim.yaml: scan_distance_to_base_link)
        self.declare_parameter('laser_x', 0.275)
        self.declare_parameter('laser_y', 0.0)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_pub = StaticTransformBroadcaster(self)
        self.offset_done = False
        self.tries = 0

        self.pub = self.create_publisher(LaserScan, self.get_parameter('out_topic').value, 10)
        self.create_subscription(LaserScan, self.get_parameter('in_topic').value,
                                 self.scan_cb, qos_profile_sensor_data)
        self.get_logger().info('Relaying scans for AMCL: '
                               f'{self.get_parameter("in_topic").value} -> '
                               f'{self.get_parameter("out_topic").value} '
                               f'(frame {self.get_parameter("est_laser_frame").value})')

    def publish_offset(self, laser_frame):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        try:
            tf = self.tf_buffer.lookup_transform(p('truth_base_frame'), laser_frame, Time())
            x, y = tf.transform.translation.x, tf.transform.translation.y
            yaw = yaw_from_quat(tf.transform.rotation)
            source = 'TF'
        except Exception:
            self.tries += 1
            if self.tries < 20:          # TF may not be there yet; try again next scan
                return False
            x, y, yaw, source = p('laser_x'), p('laser_y'), 0.0, 'parameters'

        st = TransformStamped()
        st.header.stamp = self.get_clock().now().to_msg()
        st.header.frame_id = p('est_base_frame')
        st.child_frame_id = p('est_laser_frame')
        st.transform.translation.x = x
        st.transform.translation.y = y
        st.transform.rotation.z = math.sin(yaw / 2.0)
        st.transform.rotation.w = math.cos(yaw / 2.0)
        self.static_pub.sendTransform(st)
        self.get_logger().info(f'LiDAR offset ({source}): x={x:.3f} m, y={y:.3f} m, '
                               f'yaw={math.degrees(yaw):.1f} deg')
        return True

    def scan_cb(self, msg):
        if not self.offset_done:
            self.offset_done = self.publish_offset(msg.header.frame_id)
            if not self.offset_done:
                return
            # The offset never changes: stop listening to /tf (hundreds of
            # messages per second -- listening costs a whole CPU core in Python)
            try:
                self.tf_listener.unregister()
            except Exception:
                self.destroy_subscription(self.tf_listener.tf_sub)
                self.destroy_subscription(self.tf_listener.tf_static_sub)
            self.tf_listener = None
            self.tf_buffer = None
        msg.header.frame_id = self.get_parameter('est_laser_frame').value
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimScanRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
