#!/usr/bin/env python3
"""EKF version 2: wheel speed + gyro + LiDAR cones against a known cone map.

WHAT IS NEW IN v2 (compared with v1)
  1. LiDAR cone update: every cone that scan_cluster_node detects is compared
     with where the known cone map says it should be. The difference corrects
     position AND heading. This stops the drift that v1 still had.
  2. Data association: before a detection can be used, the filter decides
     WHICH mapped cone it is, using the Mahalanobis distance (distance measured
     in units of uncertainty) and rejects detections that match no cone.
  3. Wheel-speed scale state k: v1 could not learn the 3 % wheel-speed error.
     With the LiDAR telling the filter where the car really is, it can.
  4. Consistency check (NIS) and Foxglove markers showing which cone matched.
  5. Position +- in the report now uses both x and y: sqrt(Pxx + Pyy).

STATE (5 numbers)
    x = [ px, py, yaw, b, k ]
        px, py : position in the map frame                         [m]
        yaw    : heading                                           [rad]
        b      : gyro bias (b_gyro)                                [rad/s]
        k      : wheel-speed scale, true speed = k * measured speed  [-]
                 (starts at 1.0; the sim's 3 % error means k -> 1/1.03 = 0.971)

PREDICT (every IMU message, 100 Hz) -- same idea as v1, plus k
    v      = k * v_meas                    corrected wheel speed
    w      = w_meas - b                    corrected yaw rate
    px    += v * cos(yaw + beta) * dt
    py    += v * sin(yaw + beta) * dt
    yaw   += w * dt
    b, k   stay the same, but their uncertainty grows a little (random walk)

    F (Jacobian of the prediction w.r.t. the state), with c/s = cos/sin(yaw+beta):
        px : [ 1  0  -v*s*dt   0    v_meas*c*dt ]
        py : [ 0  1   v*c*dt   0    v_meas*s*dt ]   <- a wrong k becomes a
        yaw: [ 0  0   1       -dt   0           ]      position error
        b  : [ 0  0   0        1    0           ]
        k  : [ 0  0   0        0    1           ]

UPDATE 1 -- zero-velocity update (ZUPT), unchanged from v1
    While stopped: z = w_meas, h(x) = b, H = [0 0 0 1 0].

UPDATE 2 -- LiDAR cone (NEW)
    A detection gives the cone position in the CAR frame (x forward, y left):
        z = [ zx, zy ]   (from /lidar/cones, shifted by the LiDAR mounting offset)
    If the car is at (px, py, yaw) and the map cone is at (mx, my), the filter
    EXPECTS to see it at (rotate the map difference into the car frame):
        dx, dy = mx - px, my - py
        h(x)   = [  cos(yaw)*dx + sin(yaw)*dy ,
                   -sin(yaw)*dx + cos(yaw)*dy ]
    Jacobian H = dh/dx (2 x 5):
        [ -cos(yaw)  -sin(yaw)   h_y   0  0 ]
        [  sin(yaw)  -cos(yaw)  -h_x   0  0 ]
    Measurement noise R: the LiDAR is accurate in RANGE (sigma_r) but only
    ~1 deg in ANGLE (sigma_a), so a far cone is uncertain sideways. R is built
    in polar form and rotated into x/y:  R = J diag(sigma_r^2, sigma_a^2) J^T.

    Standard EKF update:
        nu = z - h(x)                  innovation (what surprised us)
        S  = H P H^T + R               how big nu is expected to be
        K  = P H^T S^-1                Kalman gain
        x  = x + K nu
        P  = (I - K H) P (I - K H)^T + K R K^T   (Joseph form: stays symmetric
                                                   and positive, numerically safe)

DATA ASSOCIATION (NEW) -- which mapped cone is this detection?
    For every detection and every map cone:
        NIS = nu^T S^-1 nu       "normalised innovation squared"
    = squared distance between seen and expected cone, measured in standard
    deviations. The smallest NIS wins, if it is below the gate
    (9.21 = chi-square, 2 degrees of freedom, 99 %). Otherwise the detection is
    rejected (false cone, or a cone not in the map). Each map cone can be used
    once per scan.

CONSISTENCY (NEW)
    If the filter's uncertainty is honest, the average NIS of accepted updates
    is about 2 (the number of measured values). Much larger -> filter is
    overconfident; much smaller -> too cautious. Reported every 2 s.

Topics
  in : /imu/data, /vehicle/measured, /lidar/cones, /initialpose
       /ego_racecar/odom, /sim/true_gyro_bias  (simulation truth, report only)
  out: /odom/ekf (Odometry + covariance), /odom/ekf_path,
       /ekf/bias, /ekf/bias_sigma, /ekf/speed_scale, /ekf/nis (Float64),
       /ekf/matches (Marker: lines from the car to each matched map cone)
"""
import math

import numpy as np

CHI2_2DOF_99 = 9.21


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# ----------------------------------------------------------------------------
# The filter itself: plain numpy, no ROS -> can be tested offline
# ----------------------------------------------------------------------------

class ConeEKF:
    PX, PY, YAW, B, K = range(5)
    N = 5

    def __init__(self, x0, P0_diag, sigma_v, sigma_w, sigma_bw, sigma_kw):
        self.x = np.array(x0, dtype=float)
        self.P = np.diag(np.array(P0_diag, dtype=float) ** 2)
        self.sigma_v = sigma_v      # wheel speed noise              [m/s]
        self.sigma_w = sigma_w      # gyro white noise per sample    [rad/s]
        self.sigma_bw = sigma_bw    # gyro bias random walk          [rad/s/sqrt(s)]
        self.sigma_kw = sigma_kw    # speed-scale random walk        [1/sqrt(s)]

    # --- prediction ----------------------------------------------------------

    def predict(self, v_meas, w_meas, beta, dt):
        px, py, yaw, b, k = self.x
        v = k * v_meas
        heading = yaw + beta
        c, s = math.cos(heading), math.sin(heading)

        self.x[self.PX] = px + v * c * dt
        self.x[self.PY] = py + v * s * dt
        self.x[self.YAW] = wrap(yaw + (w_meas - b) * dt)

        F = np.eye(self.N)
        F[0, 2] = -v * s * dt
        F[1, 2] = v * c * dt
        F[0, 4] = v_meas * c * dt
        F[1, 4] = v_meas * s * dt
        F[2, 3] = -dt
        G = np.zeros((self.N, 2))            # how input noise (v_meas, w_meas) enters
        G[0, 0] = k * c * dt
        G[1, 0] = k * s * dt
        G[2, 1] = dt
        Qu = np.diag([self.sigma_v ** 2, self.sigma_w ** 2])
        self.P = F @ self.P @ F.T + G @ Qu @ G.T
        self.P[self.B, self.B] += self.sigma_bw ** 2 * dt
        self.P[self.K, self.K] += self.sigma_kw ** 2 * dt

    def stationary_predict(self, dt):
        self.P[self.B, self.B] += self.sigma_bw ** 2 * dt
        self.P[self.K, self.K] += self.sigma_kw ** 2 * dt

    # --- generic EKF update --------------------------------------------------

    def _update(self, nu, H, R):
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ nu
        self.x[self.YAW] = wrap(self.x[self.YAW])
        I_KH = np.eye(self.N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return float(nu @ S_inv @ nu)

    # --- update 1: zero velocity ---------------------------------------------

    def zupt(self, w_meas):
        H = np.zeros((1, self.N))
        H[0, self.B] = 1.0
        nu = np.array([w_meas - self.x[self.B]])
        R = np.array([[self.sigma_w ** 2]])
        self._update(nu, H, R)

    # --- update 2: LiDAR cone ------------------------------------------------

    def expected_cone(self, mx, my):
        """Where the map cone (mx, my) should appear in the car frame, and H."""
        px, py, yaw = self.x[0], self.x[1], self.x[2]
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = mx - px, my - py
        hx = c * dx + s * dy
        hy = -s * dx + c * dy
        H = np.zeros((2, self.N))
        H[0, 0], H[0, 1], H[0, 2] = -c, -s, hy
        H[1, 0], H[1, 1], H[1, 2] = s, -c, -hx
        return np.array([hx, hy]), H

    def cone_nis(self, z, mx, my, R):
        """NIS of detection z against map cone (mx, my), without updating."""
        h, H = self.expected_cone(mx, my)
        nu = z - h
        S = H @ self.P @ H.T + R
        return float(nu @ np.linalg.solve(S, nu))

    def cone_update(self, z, mx, my, R):
        h, H = self.expected_cone(mx, my)
        return self._update(z - h, H, R)

    # --- helpers -------------------------------------------------------------

    def reset_pose(self, px, py, yaw, sigma_pos=0.05, sigma_yaw=math.radians(2.0)):
        """New known pose; bias and speed scale are kept (they belong to the sensors)."""
        self.x[:3] = [px, py, yaw]
        self.P[:3, :] = 0.0
        self.P[:, :3] = 0.0
        self.P[0, 0] = self.P[1, 1] = sigma_pos ** 2
        self.P[2, 2] = sigma_yaw ** 2

    def sigma(self, i):
        return math.sqrt(max(self.P[i, i], 0.0))

    def sigma_pos(self):
        return math.sqrt(max(self.P[0, 0] + self.P[1, 1], 0.0))


def cone_noise(zx_laser, zy_laser, sigma_r, sigma_a):
    """Measurement covariance of one cone, from range/angle noise (polar -> x,y)."""
    r = max(math.hypot(zx_laser, zy_laser), 1e-3)
    a = math.atan2(zy_laser, zx_laser)
    J = np.array([[math.cos(a), -r * math.sin(a)],
                  [math.sin(a),  r * math.cos(a)]])
    return J @ np.diag([sigma_r ** 2, sigma_a ** 2]) @ J.T


def associate_and_update(ekf, detections, landmarks, sigma_r, sigma_a, laser_xy, gate):
    """detections: list of (x, y) in the LASER frame. landmarks: list of (mx, my).
    Greedy nearest-neighbour in NIS, each landmark at most once per scan.
    Returns (list of (detection, landmark_index, nis) accepted, rejected count)."""
    lx, ly = laser_xy
    cands = []
    for di, (zx, zy) in enumerate(detections):
        z = np.array([zx + lx, zy + ly])                 # laser frame -> car frame
        R = cone_noise(zx, zy, sigma_r, sigma_a)
        for li, (mx, my) in enumerate(landmarks):
            nis = ekf.cone_nis(z, mx, my, R)
            if nis < gate:
                cands.append((nis, di, li, z, R))
    cands.sort(key=lambda c: c[0])
    used_d, used_l, accepted = set(), set(), []
    for nis, di, li, z, R in cands:
        if di in used_d or li in used_l:
            continue
        used_d.add(di)
        used_l.add(li)
        mx, my = landmarks[li]
        nis_now = ekf.cone_update(z, mx, my, R)      # recomputed with the updated state
        accepted.append((di, li, nis_now))
    return accepted, len(detections) - len(accepted)


# ----------------------------------------------------------------------------
# ROS 2 node
# ----------------------------------------------------------------------------

try:
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import Point, PoseArray, PoseStamped, PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry, Path
    from rclpy.node import Node
    from rclpy.time import Time
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Float64
    from tf2_ros import Buffer, TransformListener
    from visualization_msgs.msg import Marker
except ImportError:
    Node = object


def quat_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class EkfNode(Node):

    def __init__(self):
        super().__init__('ekf_node')
        d = self.declare_parameter
        # Start pose (matches sim.yaml) and its uncertainty
        d('x0', 3.0); d('y0', 0.8); d('yaw0', 0.0)
        d('sigma_pos0', 0.05)               # m
        d('sigma_yaw0_deg', 1.0)            # deg
        d('sigma_bias0', 0.05)              # rad/s: bias unknown at start
        d('sigma_scale0', 0.05)             # 5 %: speed scale unknown at start
        # Process noise (the "tuning knobs")
        d('sigma_v', 0.05)                  # m/s   wheel speed
        d('sigma_w', 0.005)                 # rad/s gyro per sample
        d('sigma_bw', 0.0001)               # rad/s/sqrt(s) bias drift
        d('sigma_kw', 0.0005)               # 1/sqrt(s) speed-scale drift (tyre wear, load)
        # LiDAR cone update
        d('use_lidar', True)
        d('sigma_range', 0.05)              # m   cone range noise
        d('sigma_angle_deg', 1.0)           # deg cone bearing noise (A1: ~1 deg beams)
        d('gate', CHI2_2DOF_99)
        d('max_scan_age', 0.3)              # s   ignore older detections
        # Known cone map (x1, y1, x2, y2, ...) = the cones in make_grad_track.py.
        # On the real track: measure the cones and put their positions here.
        d('landmarks', [7.0, 0.5, 9.5, 1.1, 13.2, 4.5, 8.0, 8.2, 5.0, 7.8, 0.8, 4.5])
        # Frames and LiDAR mounting (TF is tried first; these are the fallback)
        d('base_frame', 'ego_racecar/base_link')
        d('laser_x', 0.275); d('laser_y', 0.0)
        # Vehicle geometry for the slip angle
        d('lf', 0.15875); d('lr', 0.17145)
        # Zero-velocity detection
        d('stopped_speed', 0.01)
        d('stopped_time', 0.3)
        d('frame_id', 'map')

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.ekf = ConeEKF(
            x0=[p('x0'), p('y0'), p('yaw0'), 0.0, 1.0],
            P0_diag=[p('sigma_pos0'), p('sigma_pos0'), math.radians(p('sigma_yaw0_deg')),
                     p('sigma_bias0'), p('sigma_scale0')],
            sigma_v=p('sigma_v'), sigma_w=p('sigma_w'),
            sigma_bw=p('sigma_bw'), sigma_kw=p('sigma_kw'))
        lm = list(p('landmarks'))
        self.landmarks = list(zip(lm[0::2], lm[1::2]))
        self.laser_xy = None

        self.v = 0.0
        self.steer = 0.0
        self.stopped_since = None
        self.last_imu_t = None
        self.zupt_count = 0
        self.distance = 0.0
        self.truth = None
        self.true_bias = None
        self.stats = {'acc': 0, 'rej': 0, 'nis': 0.0}

        self.path = Path()
        self.path.header.frame_id = p('frame_id')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Imu, '/imu/data', self.imu_cb, 100)
        self.create_subscription(AckermannDriveStamped, '/vehicle/measured', self.meas_cb, 50)
        self.create_subscription(PoseArray, '/lidar/cones', self.cones_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.reset_cb, 10)
        self.create_subscription(Odometry, '/ego_racecar/odom', self.truth_cb, 10)
        self.create_subscription(Float64, '/sim/true_gyro_bias', self.true_bias_cb, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom/ekf', 10)
        self.path_pub = self.create_publisher(Path, '/odom/ekf_path', 10)
        self.bias_pub = self.create_publisher(Float64, '/ekf/bias', 10)
        self.bias_sigma_pub = self.create_publisher(Float64, '/ekf/bias_sigma', 10)
        self.scale_pub = self.create_publisher(Float64, '/ekf/speed_scale', 10)
        self.nis_pub = self.create_publisher(Float64, '/ekf/nis', 10)
        self.match_pub = self.create_publisher(Marker, '/ekf/matches', 10)
        self.create_timer(0.2, self.publish_path)
        self.create_timer(2.0, self.report)
        mode = 'wheel + gyro + LiDAR cones' if p('use_lidar') else 'wheel + gyro only (v1 mode)'
        self.get_logger().info(f'EKF v2 running ({mode}), {len(self.landmarks)} map cones')

    # --- inputs --------------------------------------------------------------

    def meas_cb(self, msg):
        self.v = msg.drive.speed
        self.steer = msg.drive.steering_angle
        now = Time.from_msg(msg.header.stamp)
        if abs(self.v) < self.get_parameter('stopped_speed').value:
            if self.stopped_since is None:
                self.stopped_since = now
        else:
            self.stopped_since = None

    def is_stopped(self, now):
        if self.stopped_since is None:
            return False
        waited = (now - self.stopped_since).nanoseconds * 1e-9
        return waited >= self.get_parameter('stopped_time').value

    def imu_cb(self, msg):
        t = Time.from_msg(msg.header.stamp)
        if self.last_imu_t is None:
            self.last_imu_t = t
            return
        dt = (t - self.last_imu_t).nanoseconds * 1e-9
        self.last_imu_t = t
        if dt <= 0.0 or dt > 0.5:
            return
        w_m = msg.angular_velocity.z

        if self.is_stopped(t):
            self.ekf.stationary_predict(dt)
            self.ekf.zupt(w_m)
            self.zupt_count += 1
        else:
            lf, lr = self.get_parameter('lf').value, self.get_parameter('lr').value
            beta = math.atan(lr / (lf + lr) * math.tan(self.steer))
            self.ekf.predict(self.v, w_m, beta, dt)
            self.distance += abs(self.ekf.x[4] * self.v) * dt

        self.publish_odom(msg.header.stamp, w_m)

    def lookup_laser_offset(self, laser_frame):
        """LiDAR position on the car (a fixed mounting offset, not the car's pose)."""
        if self.laser_xy is not None:
            return self.laser_xy
        base = self.get_parameter('base_frame').value
        try:
            tf = self.tf_buffer.lookup_transform(base, laser_frame, Time())
            self.laser_xy = (tf.transform.translation.x, tf.transform.translation.y)
            self.get_logger().info(f'LiDAR offset from TF: x={self.laser_xy[0]:.3f} m, '
                                   f'y={self.laser_xy[1]:.3f} m')
        except Exception:
            self.laser_xy = (self.get_parameter('laser_x').value,
                             self.get_parameter('laser_y').value)
            self.get_logger().warn(f'No TF {base} -> {laser_frame}; using laser_x/laser_y '
                                   f'parameters {self.laser_xy}')
        return self.laser_xy

    def cones_cb(self, msg):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        if not p('use_lidar') or not msg.poses or self.last_imu_t is None:
            return
        age = (self.last_imu_t - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9
        if age > p('max_scan_age'):
            return
        laser_xy = self.lookup_laser_offset(msg.header.frame_id)
        detections = [(pp.position.x, pp.position.y) for pp in msg.poses]
        accepted, rejected = associate_and_update(
            self.ekf, detections, self.landmarks,
            p('sigma_range'), math.radians(p('sigma_angle_deg')), laser_xy, p('gate'))
        self.stats['acc'] += len(accepted)
        self.stats['rej'] += rejected
        for _, _, nis in accepted:
            self.stats['nis'] += nis
            self.nis_pub.publish(Float64(data=nis))
        self.publish_matches(msg.header.stamp, [li for _, li, _ in accepted])

    def reset_cb(self, msg):
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.ekf.reset_pose(msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        self.distance = 0.0
        self.path.poses.clear()
        self.get_logger().info('EKF pose reset (bias and speed scale kept)')

    def truth_cb(self, msg):
        self.truth = msg

    def true_bias_cb(self, msg):
        self.true_bias = msg.data

    # --- outputs -------------------------------------------------------------

    def publish_odom(self, stamp, w_m):
        px, py, yaw, b, k = self.ekf.x
        P = self.ekf.P
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.path.header.frame_id
        odom.child_frame_id = self.get_parameter('base_frame').value
        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        (odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
         odom.pose.pose.orientation.z, odom.pose.pose.orientation.w) = quat_from_yaw(yaw)
        cov = [0.0] * 36                      # (x, y, z, roll, pitch, yaw), row-major
        cov[0], cov[1], cov[5] = P[0, 0], P[0, 1], P[0, 2]
        cov[6], cov[7], cov[11] = P[1, 0], P[1, 1], P[1, 2]
        cov[30], cov[31], cov[35] = P[2, 0], P[2, 1], P[2, 2]
        odom.pose.covariance = [float(c) for c in cov]
        odom.twist.twist.linear.x = float(k * self.v)
        odom.twist.twist.angular.z = float(w_m - b)
        self.odom_pub.publish(odom)
        self.bias_pub.publish(Float64(data=float(b)))
        self.bias_sigma_pub.publish(Float64(data=self.ekf.sigma(3)))
        self.scale_pub.publish(Float64(data=float(k)))

    def publish_matches(self, stamp, landmark_ids):
        m = Marker()
        m.header.frame_id = self.path.header.frame_id
        m.header.stamp = stamp
        m.ns = 'ekf_matches'
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.03
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 1.0, 0.9
        px, py = float(self.ekf.x[0]), float(self.ekf.x[1])
        for li in landmark_ids:
            mx, my = self.landmarks[li]
            m.points.append(Point(x=px, y=py, z=0.1))
            m.points.append(Point(x=float(mx), y=float(my), z=0.1))
        self.match_pub.publish(m)

    def publish_path(self):
        px, py, yaw = (float(v) for v in self.ekf.x[:3])
        ps = PoseStamped()
        ps.header.frame_id = self.path.header.frame_id
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = px, py
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = quat_from_yaw(yaw)
        self.path.poses.append(ps)
        self.path.header.stamp = ps.header.stamp
        self.path_pub.publish(self.path)

    def report(self):
        px, py, yaw, b, k = self.ekf.x
        line = f'driven {self.distance:5.1f} m'
        if self.truth is not None:
            tp = self.truth.pose.pose
            pos_err = math.hypot(px - tp.position.x, py - tp.position.y)
            yaw_err = math.degrees(wrap(yaw - yaw_from_quat(tp.orientation)))
            line += (f' | pos err {pos_err:5.2f} m (+-{2 * self.ekf.sigma_pos():4.2f})'
                     f' | heading err {yaw_err:+6.2f} deg'
                     f' (+-{2 * math.degrees(self.ekf.sigma(2)):4.2f})')
        line += (f' | bias {math.degrees(b):+5.2f} deg/s'
                 f' (+-{2 * math.degrees(self.ekf.sigma(3)):4.2f})')
        if self.true_bias is not None:
            line += f' true {math.degrees(self.true_bias):+5.2f}'
        line += f' | speed scale {k:5.3f} (+-{2 * self.ekf.sigma(4):5.3f})'
        s = self.stats
        if self.get_parameter('use_lidar').value:
            mean_nis = s['nis'] / s['acc'] if s['acc'] else float('nan')
            line += f" | cones used {s['acc']:3d} rejected {s['rej']:2d} mean NIS {mean_nis:4.1f}"
        line += f' | ZUPTs {self.zupt_count}'
        self.stats = {'acc': 0, 'rej': 0, 'nis': 0.0}
        self.get_logger().info(line)


def main(args=None):
    rclpy.init(args=args)
    node = EkfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
