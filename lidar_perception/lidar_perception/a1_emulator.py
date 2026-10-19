#!/usr/bin/env python3
"""SIMULATION ONLY: a ROTATING LiDAR (RPLIDAR A1 timing) made from the simulator's instantaneous scans.

Why: the F1TENTH simulator publishes /scan 250 times a second (a new physics state every ~10 ms),
and every one of those scans is INSTANTANEOUS -- all 360 beams from one pose. The old launch took one
of them every 1/7 s (topic_tools throttle). A real RPLIDAR A1 SPINS: one turn takes ~1/7 s and each
beam is measured at its own moment while the car moves (6 m/s: 0.86 m and up to ~17 deg per turn).
So our localization had never seen the distortion ("skew") the real sensor has at speed.

What it does: every turn (1/rate_hz s) it builds ONE scan, beam by beam, exactly like the A1 + the
rplidar_ros ROS 2 driver (Slamtec, ros2 branch, A1 launch defaults) deliver it:
  - the turn starts behind the car (angle +pi) at header.stamp and runs CLOCKWISE seen from above
    (rplidar_ros: ROS angle = pi - device angle, data in reversed order);
  - the ranges array is in increasing angle, so the LAST element is measured FIRST:
    element j is measured at  stamp + (N-1-j) * time_increment    (rotating_lidar.beam_offsets 'reverse')
  - header.stamp = the START of the turn, time_increment = T / (N-1), scan_time = T (as the driver);
  - each element's range comes from the simulator's physics states at that element's moment:
    linear interpolation between the two states around it (smooth in time like a real spinning
    sensor), or the nearer state at a range jump > edge_jump_m (an object edge).
Topics: in_topic (/scan, the simulator) -> out_topic (/scan_a1, or /scan_a1_raw when scan_deskew
runs after it). Repeated messages of the same physics state (published 2-3 times) are skipped.
Note: interpolating two states averages their noise (the simulator adds 2 cm per state) -> ~0.8x
the simulator's range noise.
"""
import math
from collections import deque

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

try:
    from lidar_perception.rotating_lidar import assemble, beam_offsets
except ImportError:                                   # run as a plain script
    from rotating_lidar import assemble, beam_offsets


def stamp_s(stamp):
    return stamp.sec + 1e-9 * stamp.nanosec


def to_stamp(t):
    sec = int(math.floor(t))
    nsec = int(round((t - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec, nsec = sec + 1, nsec - 1_000_000_000
    return TimeMsg(sec=sec, nanosec=nsec)


class A1Emulator(Node):

    def __init__(self):
        super().__init__('a1_emulator')
        d = self.declare_parameter
        d('in_topic', '/scan')
        d('out_topic', '/scan_a1')
        d('rate_hz', 7.0)               # turns per second (real A1: ~5.5-10 Hz; the old throttle: 7)
        d('edge_jump_m', 0.15)          # range jump between two states treated as an edge (no blending)
        d('buffer_s', 0.5)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.period = 1.0 / float(p('rate_hz'))
        self.edge = float(p('edge_jump_m'))
        self.buffer_s = float(p('buffer_s'))
        self.states = deque()           # (t, ranges) of NEW physics states
        self.last = None
        self.t0 = None                  # start of the current turn
        self.n_out = self.n_resync = 0
        self.pub = self.create_publisher(LaserScan, p('out_topic'), 10)
        self.create_subscription(LaserScan, p('in_topic'), self.scan_cb, qos_profile_sensor_data)
        self.create_timer(10.0, self.report)
        self.get_logger().info(
            f"rotating LiDAR (A1 / rplidar_ros timing): {p('in_topic')} -> {p('out_topic')}, "
            f"{p('rate_hz'):.1f} turns/s; turn starts behind (+pi), clockwise; element j measured at "
            f"stamp + (N-1-j) * time_increment; stamp = start of the turn")

    def scan_cb(self, msg):
        r = np.asarray(msg.ranges, dtype=float)
        if self.last is not None and r.shape == self.last.shape and np.array_equal(r, self.last):
            return                                  # the same physics state, published again
        self.last = r
        t = stamp_s(msg.header.stamp)
        if self.states and t <= self.states[-1][0]:
            return
        self.states.append((t, r))
        while self.states and t - self.states[0][0] > self.buffer_s:
            self.states.popleft()
        if self.t0 is None:
            self.t0 = t
            return
        if t - self.t0 > 3.0 * self.period:         # simulator stalled or jumped: start a new turn now
            self.t0 = t
            self.n_resync += 1
            return
        n = len(r)
        if n < 2:
            return
        dt = self.period / (n - 1)
        off = beam_offsets(n, dt, 'reverse')
        if t < self.t0 + off.max():
            return                                  # the turn's last beam is still in the future
        st = np.array([s[0] for s in self.states])
        sr = np.stack([s[1] for s in self.states])
        ranges = assemble(st, sr, self.t0 + off, self.edge)
        out = LaserScan()
        out.header.stamp = to_stamp(self.t0)
        out.header.frame_id = msg.header.frame_id
        out.angle_min, out.angle_max, out.angle_increment = msg.angle_min, msg.angle_max, msg.angle_increment
        out.time_increment = dt
        out.scan_time = self.period
        out.range_min, out.range_max = msg.range_min, msg.range_max
        out.ranges = ranges.astype(np.float32).tolist()
        self.pub.publish(out)
        self.n_out += 1
        self.t0 += self.period

    def report(self):
        self.get_logger().info(f'{self.n_out} rotating scans published (restarted after a pause: {self.n_resync})')


def main(args=None):
    rclpy.init(args=args)
    node = A1Emulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
