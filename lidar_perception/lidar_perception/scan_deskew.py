#!/usr/bin/env python3
"""LiDAR DE-SKEW: undo the rotating LiDAR's motion distortion before AMCL and the obstacle detector.

A spinning LiDAR (RPLIDAR A1: one turn ~1/7 s) measures its beams one after another while the car
moves -- at 6 m/s ~0.86 m and up to ~17 deg of turning during ONE scan. The driver still packs all
beams into one message as if they came from one pose: walls ahead look closer, corners bend, AMCL
puts the car in the wrong place and obstacles land in the wrong place.

For every beam k (range rho_k, angle theta_k, measured at t_k):
    p_k   = rho_k (cos theta_k, sin theta_k)          the point in the LiDAR frame AT ITS MOMENT t_k
    L(t)  = B(t) * M                                   LiDAR pose = car pose (odometry) * mount
    p_ref = L(t_ref)^-1 * L(t_k) * p_k                 the same point seen from the LiDAR at t_ref
t_ref = the moment of the LAST measured beam (the newest view of the world). B(t) comes from the local
EKF's odometry (odom_topic: smooth, no AMCL jumps), interpolated at every t_k (rotating_lidar.py).
The points go back into the same angle bins by "connecting the dots" (rotating_lidar.resample: the line
between two consecutive beams gives every bin direction it crosses a range; the nearer line wins; no line
across an object edge) -- the motion stretches or squeezes the scan, so plain nearest-bin placement left
12-20 % of the bins empty at 2-6 m/s. Published with header.stamp = t_ref and time_increment = 0
(= "already de-skewed, all beams from one moment"), so AMCL looks the odometry up at the right time.

WHEN WAS BEAM k MEASURED? (parameters time_order, stamp_offset_s -- CHECK ON THE REAL CAR)
  'reverse' (default) = rplidar_ros, the Slamtec ROS 2 driver (read in its source, ros2 branch 2025:
            publish_scan fills ranges[N-1-i] = node i, header.stamp = the time it started waiting for
            the scan, time_increment = scan_time / (N-1) > 0): the array is in increasing angle but the
            sensor turns clockwise from behind, so the LAST element is measured FIRST:
            t_k = stamp + stamp_offset_s + (N-1-k) * time_increment
  'forward' = the standard sensor_msgs/LaserScan meaning: t_k = stamp + stamp_offset_s + k * time_increment
  The wrong order is no fix -- turning on the spot it tears the scan's seam (behind the car) open twice
  as far as no de-skew at all (test_rotating_lidar.py) -> real-car spin test.
mode: 'full' (default) | 'restamp' (cheap partial fix: ranges unchanged, stamp = the beams' mean time)
      | 'off' (pass everything through unchanged).
Passed through unchanged: time_increment = 0 (already instantaneous, e.g. the simulator's throttle),
no odometry yet, odometry too old (> max_extrap_s before the last beam) -- counted in the report.
Mount: laser_x / laser_y / laser_yaw = the LiDAR's pose in the odometry's car frame (sim: 0.275, 0, 0;
REAL CAR: the measured mount, same values as ekf_node's laser_x / laser_y / laser_yaw).
"""
import math
import time
from collections import deque

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

try:
    from lidar_perception.rotating_lidar import beam_offsets, deskew, interp_poses, resample
except ImportError:                                   # run as a plain script
    from rotating_lidar import beam_offsets, deskew, interp_poses, resample


def stamp_s(stamp):
    return stamp.sec + 1e-9 * stamp.nanosec


def to_stamp(t):
    sec = int(math.floor(t))
    nsec = int(round((t - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec, nsec = sec + 1, nsec - 1_000_000_000
    return TimeMsg(sec=sec, nanosec=nsec)


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ScanDeskew(Node):

    def __init__(self):
        super().__init__('scan_deskew')
        d = self.declare_parameter
        d('in_topic', '/scan_a1_raw')
        d('out_topic', '/scan_a1')
        d('odom_topic', '/odom/ekf_local')
        d('mode', 'full')               # full | restamp | off
        d('time_order', 'reverse')      # reverse = rplidar_ros | forward = LaserScan definition
        d('stamp_offset_s', 0.0)        # added to header.stamp (real car: measured)
        d('laser_x', 0.275)             # m   LiDAR on the car (odometry child frame -> LiDAR frame)
        d('laser_y', 0.0)
        d('laser_yaw', 0.0)             # rad
        d('max_extrap_s', 0.05)         # newest odometry may be this much older than the last beam
        d('odom_buffer_s', 1.0)
        d('report_s', 10.0)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.mode = str(p('mode')).strip().lower()
        self.order = str(p('time_order')).strip().lower()
        if self.mode not in ('full', 'restamp', 'off'):
            raise ValueError(f"mode '{self.mode}': full | restamp | off")
        beam_offsets(2, 0.0, self.order)            # validates time_order
        self.mount = (float(p('laser_x')), float(p('laser_y')), float(p('laser_yaw')))
        self.odo = deque()                          # (t, x, y, psi unwrapped, v, w)
        self.psi_last = None
        self.cnt = dict(full=0, restamp=0, instant=0, no_odom=0, odom_gap=0, off=0)
        self.shift, self.work, self.holes = [], [], []
        self.warned = set()
        self.pub = self.create_publisher(LaserScan, p('out_topic'), 10)
        self.create_subscription(Odometry, p('odom_topic'), self.odom_cb, 50)
        self.create_subscription(LaserScan, p('in_topic'), self.scan_cb, qos_profile_sensor_data)
        self.create_timer(float(p('report_s')), self.report)
        self.get_logger().info(
            f"scan de-skew ({self.mode}): {p('in_topic')} -> {p('out_topic')}, motion from {p('odom_topic')}; "
            f"beam time order '{self.order}' ({'rplidar_ros' if self.order == 'reverse' else 'LaserScan'}), "
            f"stamp offset {1000 * p('stamp_offset_s'):+.1f} ms; "
            f"LiDAR at ({self.mount[0]:.3f}, {self.mount[1]:.3f}) m, {math.degrees(self.mount[2]):.1f} deg")

    def odom_cb(self, msg):
        t = stamp_s(msg.header.stamp)
        if self.odo and t <= self.odo[-1][0]:
            return
        yaw = yaw_of(msg.pose.pose.orientation)
        psi = yaw if self.psi_last is None else self.psi_last + math.remainder(yaw - self.psi_last, 2 * math.pi)
        self.psi_last = psi
        self.odo.append((t, msg.pose.pose.position.x, msg.pose.pose.position.y, psi,
                         msg.twist.twist.linear.x, msg.twist.twist.angular.z))
        while self.odo and t - self.odo[0][0] > self.get_parameter('odom_buffer_s').value:
            self.odo.popleft()

    def warn_once(self, key, text):
        if key not in self.warned:
            self.warned.add(key)
            self.get_logger().warn(text)

    def scan_cb(self, msg):
        t_start = time.perf_counter()
        n = len(msg.ranges)
        dt = msg.time_increment
        if self.mode == 'off' or dt <= 0.0 or n < 2:
            self.cnt['off' if self.mode == 'off' else 'instant'] += 1
            self.pub.publish(msg)
            return
        off = beam_offsets(n, dt, self.order) + self.get_parameter('stamp_offset_s').value
        t_k = stamp_s(msg.header.stamp) + off
        out = LaserScan()
        out.header.frame_id = msg.header.frame_id
        out.angle_min, out.angle_max, out.angle_increment = msg.angle_min, msg.angle_max, msg.angle_increment
        out.time_increment = 0.0
        out.scan_time = msg.scan_time
        out.range_min, out.range_max = msg.range_min, msg.range_max
        if self.mode == 'restamp':
            out.header.stamp = to_stamp(float(np.mean(t_k)))
            out.ranges = msg.ranges
            self.cnt['restamp'] += 1
            self.pub.publish(out)
            return
        if len(self.odo) < 2:
            self.cnt['no_odom'] += 1
            self.warn_once('no_odom', 'no odometry yet -> scans passed through unchanged until it arrives')
            self.pub.publish(msg)
            return
        o = np.array(self.odo)
        t_ref = float(t_k.max())
        xk, yk, pk, ok = interp_poses(o[:, 0], o[:, 1], o[:, 2], o[:, 3], o[:, 4], o[:, 5], t_k,
                                      self.get_parameter('max_extrap_s').value)
        if not np.all(ok):
            self.cnt['odom_gap'] += 1
            self.warn_once('gap', f'odometry does not cover the scan (newest {1000 * (t_ref - o[-1, 0]):.0f} ms '
                                  'before the last beam) -> passed through unchanged (counted in the report)')
            self.pub.publish(msg)
            return
        j_ref = int(np.argmax(t_k))
        ranges = np.asarray(msg.ranges, dtype=float)
        use = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        angles = msg.angle_min + msg.angle_increment * np.arange(n)
        qx, qy = deskew(ranges, angles, (xk, yk, pk), (xk[j_ref], yk[j_ref], pk[j_ref]), self.mount)
        new = resample(qx, qy, use, msg.angle_min, msg.angle_increment, n)
        out.header.stamp = to_stamp(t_ref)
        out.ranges = new.astype(np.float32).tolist()
        self.pub.publish(out)
        self.cnt['full'] += 1
        if np.any(use):
            px, py = ranges[use] * np.cos(angles[use]), ranges[use] * np.sin(angles[use])
            self.shift.append(float(np.max(np.hypot(qx[use] - px, qy[use] - py))))
            self.holes.append(float(np.mean(~np.isfinite(new) & use)))
        self.work.append(time.perf_counter() - t_start)

    def report(self):
        c = self.cnt
        line = (f"scans: de-skewed {c['full']}, restamped {c['restamp']}, passed through: "
                f"instantaneous {c['instant']}, no odometry {c['no_odom']}, odometry gap {c['odom_gap']}, "
                f"mode off {c['off']}")
        if self.shift:
            s = np.array(self.shift)
            line += (f" | largest point move per scan: median {100 * np.median(s):.1f} cm, max {100 * s.max():.1f} cm"
                     f" | bins left empty {100 * np.mean(self.holes):.1f} %"
                     f" | work {1000 * np.mean(self.work):.2f} ms/scan")
        self.get_logger().info(line)
        self.cnt = {k: 0 for k in self.cnt}
        self.shift, self.work, self.holes = [], [], []


def main(args=None):
    rclpy.init(args=args)
    node = ScanDeskew()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
