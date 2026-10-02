#!/usr/bin/env python3
"""SIMULATION ONLY: pretend to be the car's IMU.

The simulator gives the true yaw rate and speed but no IMU topic, so this
node builds one from the ground truth and adds the errors a cheap MEMS IMU
(MPU-6050 / ICM-20948 class) really has:

  gyro_z  = true_yaw_rate + bias(t) + white noise
            (true_yaw_rate = how far the TRUE POSE rotated since the last
             reading / dt. The simulator's own twist.angular.z is NOT used:
             it can stay stuck at a non-zero value after the car stops,
             which a real gyro would never report -- see check_sim_yawrate.py)
  accel_x = dv/dt          + bias    + white noise   (forward acceleration)
  accel_y = v * yaw_rate   + bias    + white noise   (centripetal acceleration)
  accel_z = +9.81          + noise                   (gravity; car is flat)

bias(t) is a random walk: it starts at `gyro_bias` and wanders slowly, the
way a real gyro bias changes with temperature. That is why you can't
calibrate it once and forget it -- the EKF will have to keep tracking it.

Output: /imu/data (sensor_msgs/Imu) at 100 Hz, frame ego_racecar/base_link
(IMU assumed mounted at the car's centre, axes aligned: x forward, y left,
z up -- the REP-103 convention). On the real car this node is replaced by
the IMU driver publishing the same topic.

Orientation is NOT provided (a 6-axis IMU can't measure absolute yaw), so
orientation_covariance[0] = -1, the ROS convention for "no orientation".
"""
import math
import random

# ----------------------------------------------------------------------------
# Pure model (no ROS, testable on its own)
# ----------------------------------------------------------------------------


class RandomWalkBias:
    """Bias that drifts: b_{k+1} = b_k + N(0, sigma_rw * sqrt(dt))."""

    def __init__(self, initial, random_walk):
        self.value = initial
        self.random_walk = random_walk      # units/sqrt(s)

    def step(self, dt):
        self.value += random.gauss(0.0, self.random_walk * math.sqrt(dt))
        return self.value


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def pose_yaw_rate(yaw_now, yaw_before, dt, max_rate):
    """Average yaw rate from two headings. A jump faster than max_rate can't
    be real rotation (e.g. the simulator teleported the car) -> return 0."""
    if dt <= 0.0:
        return 0.0
    rate = math.atan2(math.sin(yaw_now - yaw_before), math.cos(yaw_now - yaw_before)) / dt
    return rate if abs(rate) <= max_rate else 0.0


def ideal_imu(v, v_prev, yaw_rate, dt):
    """Noise-free body-frame IMU readings from speed and yaw rate.
    Returns (ax, ay, az, wz)."""
    ax = (v - v_prev) / dt if dt > 0 else 0.0
    ay = v * yaw_rate
    return ax, ay, 9.81, yaw_rate


# ----------------------------------------------------------------------------
# ROS 2 node
# ----------------------------------------------------------------------------

try:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Float64
except ImportError:
    Node = object


class FakeImu(Node):

    def __init__(self):
        super().__init__('fake_imu')
        self.declare_parameter('rate_hz', 100.0)
        # Gyroscope (rad/s)
        self.declare_parameter('gyro_noise', 0.005)        # white noise std per sample
        self.declare_parameter('gyro_bias', 0.01)          # ~0.6 deg/s starting bias
        self.declare_parameter('gyro_bias_walk', 0.00005)  # rad/s/sqrt(s); ~0.05 deg/s drift in 5 min
        # Accelerometer (m/s^2)
        self.declare_parameter('accel_noise', 0.05)
        self.declare_parameter('accel_bias_x', 0.08)
        self.declare_parameter('accel_bias_y', -0.05)
        self.declare_parameter('accel_smoothing', 0.2)     # low-pass on dv/dt, 0..1
        self.declare_parameter('frame_id', 'ego_racecar/base_link')
        self.declare_parameter('max_yaw_rate', 10.0)       # rad/s; faster = teleport, ignored

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.gyro_bias = RandomWalkBias(p('gyro_bias'), p('gyro_bias_walk'))
        self.true_speed = 0.0
        self.true_yaw = None          # latest true heading from the pose
        self.yaw_at_last_pub = None   # heading at the previous IMU reading
        self.last_pub_time = None
        self.prev_speed = 0.0
        self.ax_filtered = 0.0
        self.dt = 1.0 / p('rate_hz')

        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_cb, 10)
        self.pub = self.create_publisher(Imu, '/imu/data', 50)
        # Simulation truth, ONLY for judging the EKF (a real IMU can't tell you this)
        self.bias_pub = self.create_publisher(Float64, '/sim/true_gyro_bias', 10)
        self.create_timer(self.dt, self.publish)
        self.create_timer(5.0, self.report)
        self.get_logger().info('Publishing simulated IMU on /imu/data '
                               f'(start gyro bias {p("gyro_bias"):.3f} rad/s)')

    def odom_cb(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.true_speed = math.copysign(math.hypot(vx, vy), vx)
        self.true_yaw = yaw_from_quat(msg.pose.pose.orientation)

    def publish(self):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        if self.true_yaw is None:
            return                                  # no simulator data yet
        now = self.get_clock().now()
        if self.last_pub_time is None:
            self.last_pub_time, self.yaw_at_last_pub = now, self.true_yaw
            return
        dt = (now - self.last_pub_time).nanoseconds * 1e-9
        true_rate = pose_yaw_rate(self.true_yaw, self.yaw_at_last_pub, dt, p('max_yaw_rate'))
        self.last_pub_time, self.yaw_at_last_pub = now, self.true_yaw

        ax, ay, az, wz = ideal_imu(self.true_speed, self.prev_speed, true_rate, self.dt)
        self.prev_speed = self.true_speed
        a = p('accel_smoothing')
        self.ax_filtered = (1 - a) * self.ax_filtered + a * ax

        g_noise, a_noise = p('gyro_noise'), p('accel_noise')
        bias = self.gyro_bias.step(self.dt)

        msg = Imu()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = p('frame_id')

        msg.orientation_covariance[0] = -1.0        # no orientation estimate

        msg.angular_velocity.x = random.gauss(0.0, g_noise)
        msg.angular_velocity.y = random.gauss(0.0, g_noise)
        msg.angular_velocity.z = wz + bias + random.gauss(0.0, g_noise)
        gv = g_noise ** 2
        msg.angular_velocity_covariance = [gv, 0.0, 0.0, 0.0, gv, 0.0, 0.0, 0.0, gv]

        msg.linear_acceleration.x = self.ax_filtered + p('accel_bias_x') + random.gauss(0.0, a_noise)
        msg.linear_acceleration.y = ay + p('accel_bias_y') + random.gauss(0.0, a_noise)
        msg.linear_acceleration.z = az + random.gauss(0.0, a_noise)
        av = a_noise ** 2
        msg.linear_acceleration_covariance = [av, 0.0, 0.0, 0.0, av, 0.0, 0.0, 0.0, av]

        self.pub.publish(msg)
        self.bias_pub.publish(Float64(data=bias))

    def report(self):
        b = self.gyro_bias.value
        self.get_logger().info(f'true gyro bias now {b:+.4f} rad/s ({math.degrees(b):+.2f} deg/s)')


def main(args=None):
    rclpy.init(args=args)
    node = FakeImu()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
