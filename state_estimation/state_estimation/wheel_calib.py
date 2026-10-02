#!/usr/bin/env python3
"""Wheel-speed calibration by a measured straight drive (REAL CAR tool).

Finds k in  v_true = k * v_measured  (wrong effective wheel radius, gear
ratio, tyre squash) and writes it to the EKF's calibration file, so the EKF
starts calibrated instead of learning k at the first corner.

HOW IT WORKS
  1. Mark a straight line on the floor with a tape measure (e.g. 10.00 m).
  2. Start this node with that distance:
        ros2 run state_estimation wheel_calib --ros-args -p true_distance:=10.0
  3. Put the car's REAR AXLE exactly on the start mark, wheels straight.
  4. Drive SLOWLY (< 1 m/s, no wheel spin) straight to the end mark and stop
     with the rear axle exactly on it. Hold still 1 s.
  5. The node prints the encoder distance and k for that trial and writes the
     mean k of all trials so far to the calibration file.
  6. Carry the car back to the start (do NOT drive back -- or do, and the
     return also counts: each "moved > min_move, then stopped 1 s" = 1 trial).
     Do 5 trials. Ctrl+C when done. Trials that differ a lot from the others
     usually mean the car was not exactly on a mark: delete the file, redo.

SIMULATION TEST (no tape measure): -p use_sim_truth:=true
  The true distance of each trial is integrated from /ego_racecar/odom instead
  of true_distance; drive straight with teleop. Expected k ~ 0.971.

Input : /vehicle/measured (ackermann_msgs/AckermannDriveStamped) -- on the real
        car published by the STM32 bridge, in simulation by fake_vehicle_sensors
Output: scale_file (default ~/.ros/ekf_speed_scale.yaml), same format the EKF
        reads and writes.
"""
import math
import os
import statistics
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time


class WheelCalib(Node):

    def __init__(self):
        super().__init__('wheel_calib')
        d = self.declare_parameter
        d('true_distance', 0.0)          # m, measured with the tape
        d('tape_error', 0.02)            # m, how exactly the car is put on the marks
        d('scale_file', '~/.ros/ekf_speed_scale.yaml')
        d('write_file', True)
        d('min_move', 1.0)               # m of encoder distance before a stop ends a trial
        d('stopped_speed', 0.01)         # m/s
        d('stopped_time', 1.0)           # s
        d('use_sim_truth', False)
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        self.use_truth = p('use_sim_truth')
        if not self.use_truth and p('true_distance') <= 0.0:
            raise SystemExit('Give the measured distance: --ros-args -p true_distance:=10.0')
        self.scale_file = os.path.expanduser(p('scale_file'))

        self.d_meas = 0.0                # encoder distance of the current trial
        self.d_true = 0.0                # sim truth distance of the current trial
        self.last_t = None
        self.last_truth = None
        self.stopped_since = None
        self.ks = []

        self.create_subscription(AckermannDriveStamped, '/vehicle/measured', self.meas_cb, 50)
        if self.use_truth:
            self.create_subscription(Odometry, '/ego_racecar/odom', self.truth_cb, 50)
        self.create_timer(1.0, self.progress)
        src = 'SIM TRUTH' if self.use_truth else f'tape = {p("true_distance"):.3f} m'
        self.get_logger().info(f'Wheel calibration ({src}). Put the rear axle on the start '
                               'mark and drive slowly straight to the end mark, then stop.')

    def meas_cb(self, msg):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if t == 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9
        v = msg.drive.speed
        if self.last_t is not None:
            dt = t - self.last_t
            if 0.0 < dt < 0.5:
                self.d_meas += abs(v) * dt
        self.last_t = t

        if abs(v) < p('stopped_speed'):
            if self.stopped_since is None:
                self.stopped_since = t
            elif t - self.stopped_since >= p('stopped_time') and self.d_meas >= p('min_move'):
                self.finish_trial()
        else:
            self.stopped_since = None

    def truth_cb(self, msg):
        pos = msg.pose.pose.position
        if self.last_truth is not None:
            self.d_true += math.hypot(pos.x - self.last_truth[0], pos.y - self.last_truth[1])
        self.last_truth = (pos.x, pos.y)

    def finish_trial(self):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        true_d = self.d_true if self.use_truth else p('true_distance')
        k = true_d / self.d_meas
        self.ks.append(k)
        n = len(self.ks)
        k_mean = statistics.mean(self.ks)
        # uncertainty of the mean: spread between trials (if >= 2) and tape error
        spread = statistics.stdev(self.ks) / math.sqrt(n) if n >= 2 else 0.0
        tape = 0.0 if self.use_truth else p('tape_error') / true_d / math.sqrt(n)
        sigma = max(math.hypot(spread, tape), 0.002)
        self.get_logger().info(
            f'TRIAL {n}: true {true_d:.3f} m, encoder {self.d_meas:.3f} m -> k = {k:.4f} | '
            f'mean of {n}: k = {k_mean:.4f} +- {sigma:.4f}')
        if n >= 3 and abs(k - k_mean) > 3 * max(sigma * math.sqrt(n), 0.005):
            self.get_logger().warn('This trial is far from the others -- was the car exactly on '
                                   'the marks? Was there wheel spin?')
        if p('write_file'):
            self.write(k_mean, sigma, n)
        self.d_meas = 0.0
        self.d_true = 0.0
        self.stopped_since = None
        self.get_logger().info('Ready for the next trial (car back on the start mark).')

    def write(self, k, sigma, n):
        tmp = self.scale_file + '.tmp'
        os.makedirs(os.path.dirname(self.scale_file) or '.', exist_ok=True)
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write('# Wheel-speed scale from wheel_calib (straight-line tape drive), '
                    'v_true = k * v_measured\n')
            f.write(f'speed_scale: {k:.6f}\n')
            f.write(f'speed_scale_sigma: {sigma:.6f}\n')
            f.write(f'trials: {n}\n')
            f.write(f"saved: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        os.replace(tmp, self.scale_file)
        self.get_logger().info(f'Written to {self.scale_file}')

    def progress(self):
        if self.d_meas > 0.0:
            extra = f', sim truth {self.d_true:6.3f} m' if self.use_truth else ''
            self.get_logger().info(f'encoder distance {self.d_meas:6.3f} m{extra}')


def main(args=None):
    rclpy.init(args=args)
    node = WheelCalib()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if node.ks:
        print(f'\nDone: {len(node.ks)} trials, k = ' + ', '.join(f'{k:.4f}' for k in node.ks))
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
