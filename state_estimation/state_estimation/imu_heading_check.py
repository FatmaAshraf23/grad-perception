#!/usr/bin/env python3
"""Heading from the gyro alone -- a test before building the EKF.

Integrates the IMU yaw rate into a heading and compares it with the ground
truth, next to the heading error of the wheel odometry. This shows the two
error types the EKF will have to combine:

  * wheel odometry heading: wrong by a steering bias -> error grows with
    DISTANCE driven (and with speed / slip)
  * gyro heading: wrong by the gyro bias -> error grows with TIME, even when
    the car stands still

Gyro bias calibration: while the car is stopped (wheel speed ~ 0) the true
yaw rate is zero, so the average gyro reading IS the bias. The node averages
the first `calib_samples` stationary readings and subtracts that. Set
`calibrate:=false` to see how bad the heading gets without it.

Input : /imu/data, /vehicle/measured (to know the car is stopped),
        /ego_racecar/odom (truth, for the report only), /odom/wheel
Output: log lines every 2 s; /imu/heading (std_msgs/Float64, rad) to plot.
"""
import math

try:
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.time import Time
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Float64
except ImportError:
    Node = object


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ImuHeadingCheck(Node):

    def __init__(self):
        super().__init__('imu_heading_check')
        self.declare_parameter('yaw0', 0.0)             # matches sim.yaml stheta
        self.declare_parameter('calibrate', True)
        self.declare_parameter('calib_samples', 200)    # 2 s at 100 Hz
        self.declare_parameter('stopped_speed', 0.01)   # m/s

        self.yaw = self.get_parameter('yaw0').value
        self.last_time = None
        self.wheel_speed = 0.0
        self.calib_sum = 0.0
        self.calib_n = 0
        self.bias_est = 0.0
        self.calibrated = not self.get_parameter('calibrate').value
        self.truth = None
        self.wheel = None
        self.start = None

        self.create_subscription(Imu, '/imu/data', self.imu_cb, 100)
        self.create_subscription(AckermannDriveStamped, '/vehicle/measured', self.meas_cb, 50)
        self.create_subscription(Odometry, '/ego_racecar/odom', self.truth_cb, 10)
        self.create_subscription(Odometry, '/odom/wheel', self.wheel_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.reset_cb, 10)
        self.heading_pub = self.create_publisher(Float64, '/imu/heading', 10)
        self.create_timer(2.0, self.report)
        if self.calibrated:
            self.get_logger().info('Calibration OFF: integrating the raw gyro')
        else:
            self.get_logger().info('Keep the car STILL -- measuring gyro bias ...')

    def meas_cb(self, msg):
        self.wheel_speed = msg.drive.speed

    def truth_cb(self, msg):
        self.truth = msg

    def wheel_cb(self, msg):
        self.wheel = msg

    def imu_cb(self, msg):
        t = Time.from_msg(msg.header.stamp)
        wz = msg.angular_velocity.z

        # 1) Bias calibration while standing still
        if not self.calibrated:
            stopped = abs(self.wheel_speed) < self.get_parameter('stopped_speed').value
            if stopped:
                self.calib_sum += wz
                self.calib_n += 1
            if self.calib_n >= self.get_parameter('calib_samples').value or not stopped:
                if self.calib_n == 0:
                    self.get_logger().warn('Car moved before calibration -- bias assumed 0')
                self.bias_est = self.calib_sum / max(self.calib_n, 1)
                self.calibrated = True
                self.get_logger().info(
                    f'Gyro bias estimated from {self.calib_n} samples: '
                    f'{self.bias_est:+.4f} rad/s ({math.degrees(self.bias_est):+.2f} deg/s)')
            self.last_time = t
            return

        # 2) Integrate yaw rate -> heading
        if self.last_time is None:
            self.last_time = t
            self.start = t
            return
        dt = (t - self.last_time).nanoseconds * 1e-9
        self.last_time = t
        if self.start is None:
            self.start = t
        if dt <= 0.0 or dt > 0.5:
            return
        self.yaw = wrap(self.yaw + (wz - self.bias_est) * dt)
        self.heading_pub.publish(Float64(data=self.yaw))

    def reset_cb(self, msg):
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.start = self.last_time
        self.get_logger().info(f'Gyro heading reset to {math.degrees(self.yaw):.0f} deg')

    def report(self):
        if self.truth is None or not self.calibrated or self.start is None:
            return
        true_yaw = yaw_from_quat(self.truth.pose.pose.orientation)
        gyro_err = math.degrees(wrap(self.yaw - true_yaw))
        elapsed = (self.last_time - self.start).nanoseconds * 1e-9
        line = f't {elapsed:5.0f} s | gyro heading error {gyro_err:+6.2f} deg'
        if self.wheel is not None:
            wheel_err = math.degrees(wrap(yaw_from_quat(self.wheel.pose.pose.orientation) - true_yaw))
            line += f' | wheel heading error {wheel_err:+6.2f} deg'
        self.get_logger().info(line)


def main(args=None):
    rclpy.init(args=args)
    node = ImuHeadingCheck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
