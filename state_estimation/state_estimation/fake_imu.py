#!/usr/bin/env python3
"""SIMULATION ONLY: pretend to be the car's IMU.

The simulator gives the true yaw rate and speed but no IMU topic, so this
node builds one from the ground truth and adds the errors a cheap MEMS IMU
(MPU-6050 / ICM-20948 class) really has:

  gyro_z  = true_yaw_rate + bias(t) + white noise
            true_yaw_rate, parameter `model`:
              v1 = how far the TRUE POSE rotated since the last reading / dt.
                   Exact in total, but the 10 ms reading timer slides against the
                   simulator's 10 ms physics timer: in turns ~80 % of the readings
                   see 0 or 2 physics steps (0x / 2x the true rate).
              v3 = pause_proof.PauseProofRate (2026-10-03): the simulator's own
                   yaw-rate state, scaled by how fast the simulator really
                   progresses, plus a slow correction that keeps the reported
                   rotation equal to the pose's rotation -> smooth AND it never
                   reports rotation the simulated car did not make (the simulator
                   falls behind / pauses when the laptop is busy). Offline test on
                   recorded data: rate error rms 0.003 rad/s (v1: 0.27-0.30),
                   heading off by 0.00 deg after injected freezes.
              (The simulator's yaw-rate state alone = "v2": smooth, but it kept
               reporting rotation during simulator pauses -> rejected 2026-10-02.)
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

SIMULATOR TIMING CHECK (every 5 s in the log, both models): this node sees every
simulator message, so it counts the new physics states while the car moves and
the longest gap between two of them; a gap > stall_s = the simulator stalled.
A run with stalls is not a clean real-time run.

EVALUATION SIDE CHANNEL /sim/imu_ideal (geometry_msgs/Vector3Stamped, added 2026-10-03):
for every reading, the SAME stamp and the noise- and bias-free values it was made from
(z = yaw rate, x = forward accel, y = sideways accel). eval_logger uses z as the true yaw
rate of that reading. Why: the simulator's yaw rate jumps at every 10 ms physics step
(its steering actuator flicks the wheels by +-0.032 rad per step), and from the outside
one cannot tell reliably whether a reading taken 0-2 ms after a new state was published
already used that state (job 048) -> any "truth at the stamp" picked from
/ego_racecar/odom made the r error a lottery of timer phases. Simulation only, like
/sim/true_gyro_bias.
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
    from geometry_msgs.msg import Vector3Stamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Float64

    from state_estimation.pause_proof import PauseProofRate
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
        # True yaw rate: 'v1' = pose difference per reading, 'v3' = PauseProofRate (see docstring)
        self.declare_parameter('model', 'v3')
        self.declare_parameter('sim_step_s', 0.01)         # simulator physics step
        self.declare_parameter('stall_s', 0.025)           # no new state for longer = stalled
        self.declare_parameter('track_tau_s', 0.2)         # v3: correction time constant

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.gyro_bias = RandomWalkBias(p('gyro_bias'), p('gyro_bias_walk'))
        self.true_speed = 0.0
        self.true_yaw = None          # latest true heading from the pose
        self.yaw_at_last_pub = None   # heading at the previous IMU reading
        self.last_pub_time = None
        self.prev_speed = 0.0
        self.ax_filtered = 0.0
        self.dt = 1.0 / p('rate_hz')
        self.model = str(p('model')).strip().lower()
        if self.model not in ('v1', 'v3'):
            raise ValueError(f"model must be 'v1' or 'v3', not {self.model!r}")
        self.rate_model = PauseProofRate(step_s=p('sim_step_s'), hold_s=p('stall_s'),
                                         tau_s=p('track_tau_s'), angle=True,
                                         max_jump=0.1 * p('max_yaw_rate'))
        # simulator timing check (new physics states while moving)
        self.last_pose = None
        self.t_last_state = None
        self.speed_last_state = 0.0
        self.win = {'states': 0, 'max_gap': 0.0, 'stalls': 0}
        self.run_stalls, self.run_max_gap = 0, 0.0

        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_cb, 10)
        self.pub = self.create_publisher(Imu, '/imu/data', 50)
        # Simulation truth, ONLY for judging the EKF (a real IMU can't tell you this)
        self.bias_pub = self.create_publisher(Float64, '/sim/true_gyro_bias', 10)
        self.ideal_pub = self.create_publisher(Vector3Stamped, '/sim/imu_ideal', 50)
        self.create_timer(self.dt, self.publish)
        self.create_timer(5.0, self.report)
        self.get_logger().info('Publishing simulated IMU on /imu/data '
                               f'(start gyro bias {p("gyro_bias"):.3f} rad/s, yaw-rate model {self.model})')

    def odom_cb(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.true_speed = math.copysign(math.hypot(vx, vy), vx)
        self.true_yaw = yaw_from_quat(msg.pose.pose.orientation)
        # A NEW physics state? (the simulator re-sends its newest state every 4 ms)
        pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, self.true_yaw)
        if pose == self.last_pose:
            return
        self.last_pose = pose
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t_last_state is not None and abs(self.speed_last_state) > 0.05:
            gap = now - self.t_last_state
            self.win['states'] += 1
            self.win['max_gap'] = max(self.win['max_gap'], gap)
            if gap > self.get_parameter('stall_s').value:
                self.win['stalls'] += 1
        self.t_last_state, self.speed_last_state = now, self.true_speed
        if self.model == 'v3':
            self.rate_model.on_state(now, self.true_yaw, msg.twist.twist.angular.z)

    def publish(self):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        if self.true_yaw is None:
            return                                  # no simulator data yet
        now = self.get_clock().now()
        if self.last_pub_time is None:
            self.last_pub_time, self.yaw_at_last_pub = now, self.true_yaw
            self.rate_model.sample(now.nanoseconds * 1e-9)     # starts its clock
            return
        dt = (now - self.last_pub_time).nanoseconds * 1e-9
        if self.model == 'v3':
            true_rate = self.rate_model.sample(now.nanoseconds * 1e-9)
        else:
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
        ideal = Vector3Stamped()                    # what this reading was made from (evaluation)
        ideal.header = msg.header
        ideal.vector.x, ideal.vector.y, ideal.vector.z = self.ax_filtered, ay, wz
        self.ideal_pub.publish(ideal)

    def report(self):
        b = self.gyro_bias.value
        w, stall = self.win, self.get_parameter('stall_s').value
        self.run_stalls += w['stalls']
        self.run_max_gap = max(self.run_max_gap, w['max_gap'])
        sim = (f"sim while moving: {w['states']} new states in 5 s, longest gap {w['max_gap'] * 1000:.0f} ms, "
               f"stalls > {stall * 1000:.0f} ms: {w['stalls']} (run: {self.run_stalls}, "
               f"longest {self.run_max_gap * 1000:.0f} ms)") if w['states'] else 'sim: car not moving'
        self.get_logger().info(f'true gyro bias now {b:+.4f} rad/s ({math.degrees(b):+.2f} deg/s) | {sim}')
        self.win = {'states': 0, 'max_gap': 0.0, 'stalls': 0}


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
