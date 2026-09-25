#!/usr/bin/env python3
"""Wheel odometry with a kinematic bicycle model.

Input : /vehicle/measured  (speed from the wheel encoder, steering angle)
Output: /odom/wheel        (nav_msgs/Odometry, x, y, yaw, v, yaw rate)
        /odom/wheel_path, /odom/truth_path (nav_msgs/Path, for Foxglove)

Model (reference point = centre of gravity, wheelbase L = lf + lr):
    beta     = atan( lr / L * tan(steering) )     slip angle of the CoG
    x_dot    = v * cos(yaw + beta)
    y_dot    = v * sin(yaw + beta)
    yaw_rate = v * cos(beta) * tan(steering) / L

The pose starts at the known start position and is integrated forward
("dead reckoning"). Any error in speed or steering accumulates, so the
estimate DRIFTS. The ground truth from the simulator is used only to measure
that drift.
"""
import math


# ----------------------------------------------------------------------------
# Pure model (no ROS, testable on its own)
# ----------------------------------------------------------------------------

def bicycle_step(x, y, yaw, v, steer, dt, lf, lr):
    """Advance the pose by dt seconds. Returns (x, y, yaw, yaw_rate)."""
    L = lf + lr
    beta = math.atan(lr / L * math.tan(steer))
    yaw_rate = v * math.cos(beta) * math.tan(steer) / L
    # Midpoint integration: use the heading halfway through the step
    yaw_mid = yaw + 0.5 * yaw_rate * dt
    x += v * math.cos(yaw_mid + beta) * dt
    y += v * math.sin(yaw_mid + beta) * dt
    yaw = math.atan2(math.sin(yaw + yaw_rate * dt), math.cos(yaw + yaw_rate * dt))
    return x, y, yaw, yaw_rate


def quat_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# ----------------------------------------------------------------------------
# ROS 2 node
# ----------------------------------------------------------------------------

try:
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry, Path
    from rclpy.node import Node
    from rclpy.time import Time
except ImportError:
    Node = object


class WheelOdometry(Node):

    def __init__(self):
        super().__init__('wheel_odometry')
        # Vehicle geometry (F1TENTH values from the simulator; measure your real car!)
        self.declare_parameter('lf', 0.15875)   # m, CoG to front axle
        self.declare_parameter('lr', 0.17145)   # m, CoG to rear axle
        # Known start pose (matches sim.yaml sx, sy, stheta)
        self.declare_parameter('x0', 3.0)
        self.declare_parameter('y0', 0.8)
        self.declare_parameter('yaw0', 0.0)
        self.declare_parameter('frame_id', 'map')

        self.x = self.get_parameter('x0').value
        self.y = self.get_parameter('y0').value
        self.yaw = self.get_parameter('yaw0').value
        self.last_time = None
        self.distance = 0.0
        self.truth = None

        self.path = Path()
        self.truth_path = Path()
        self.path.header.frame_id = self.truth_path.header.frame_id = \
            self.get_parameter('frame_id').value

        self.create_subscription(AckermannDriveStamped, '/vehicle/measured', self.meas_cb, 50)
        self.create_subscription(Odometry, '/ego_racecar/odom', self.truth_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.reset_cb, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom/wheel', 10)
        self.path_pub = self.create_publisher(Path, '/odom/wheel_path', 10)
        self.truth_path_pub = self.create_publisher(Path, '/odom/truth_path', 10)
        self.create_timer(2.0, self.report)
        self.create_timer(0.2, self.publish_paths)
        self.get_logger().info('Wheel odometry running: /vehicle/measured -> /odom/wheel')

    def meas_cb(self, msg):
        t = Time.from_msg(msg.header.stamp)
        if self.last_time is None:
            self.last_time = t
            return
        dt = (t - self.last_time).nanoseconds * 1e-9
        self.last_time = t
        if dt <= 0.0 or dt > 0.5:       # skip bad or huge gaps
            return

        v = msg.drive.speed
        steer = msg.drive.steering_angle
        self.x, self.y, self.yaw, yaw_rate = bicycle_step(
            self.x, self.y, self.yaw, v, steer, dt,
            self.get_parameter('lf').value, self.get_parameter('lr').value)
        self.distance += abs(v) * dt

        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = self.path.header.frame_id
        odom.child_frame_id = 'ego_racecar/base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        (odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
         odom.pose.pose.orientation.z, odom.pose.pose.orientation.w) = quat_from_yaw(self.yaw)
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = yaw_rate
        self.odom_pub.publish(odom)

    def truth_cb(self, msg):
        self.truth = msg

    def reset_cb(self, msg):
        """When the car is reset in the simulator, reset the odometry too."""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.distance = 0.0
        self.path.poses.clear()
        self.truth_path.poses.clear()
        self.get_logger().info(f'Odometry reset to ({self.x:.2f}, {self.y:.2f}, '
                               f'{math.degrees(self.yaw):.0f} deg)')

    def publish_paths(self):
        now = self.get_clock().now().to_msg()
        ps = PoseStamped()
        ps.header.frame_id = self.path.header.frame_id
        ps.header.stamp = now
        ps.pose.position.x, ps.pose.position.y = self.x, self.y
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = quat_from_yaw(self.yaw)
        self.path.poses.append(ps)
        self.path.header.stamp = now
        self.path_pub.publish(self.path)

        if self.truth is not None:
            ts = PoseStamped()
            ts.header.frame_id = self.truth_path.header.frame_id
            ts.header.stamp = now
            ts.pose = self.truth.pose.pose
            self.truth_path.poses.append(ts)
            self.truth_path.header.stamp = now
            self.truth_path_pub.publish(self.truth_path)

    def report(self):
        if self.truth is None:
            return
        tp = self.truth.pose.pose
        pos_err = math.hypot(self.x - tp.position.x, self.y - tp.position.y)
        yaw_err = math.degrees(math.atan2(math.sin(self.yaw - yaw_from_quat(tp.orientation)),
                                          math.cos(self.yaw - yaw_from_quat(tp.orientation))))
        pct = 100.0 * pos_err / self.distance if self.distance > 0.5 else 0.0
        self.get_logger().info(
            f'driven {self.distance:5.1f} m | position error {pos_err:5.2f} m '
            f'({pct:4.1f} % of distance) | heading error {yaw_err:+5.1f} deg')


def main(args=None):
    rclpy.init(args=args)
    node = WheelOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
