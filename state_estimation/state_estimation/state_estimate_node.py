#!/usr/bin/env python3
"""StateEstimate publisher (APEX Perception -> Control contract v0.2), version 0.

Turns the global EKF output into the message control consumes:

    /odom/ekf (100 Hz, one per IMU sample) ─┐
    /vehicle/measured (steering) ───────────┼─> this node ─> /state_estimate
    /ekf/nis (one per ACCEPTED AMCL fix) ───┤                (apex_msgs/StateEstimate)
    /imu/data, /initialpose (health only) ──┘
                                     + apex_track (s_abs, e_y, e_psi, kappa)

ONE OUTPUT PER /odom/ekf MESSAGE, SAME TIMESTAMP
    ekf_global publishes once per IMU sample, stamped with that sample's time =
    "the time the state is valid". We copy that stamp unchanged and do NOT
    predict forward (contract section 5: control owns latency compensation).
    publish_us - stamp_us is therefore the real estimator latency.

WHAT IS HONEST AND WHAT IS STILL A PLACEHOLDER (v0, 2026-10-02)
    real    x, y, psi, r, s_abs, s_track, lap, e_y, e_psi, std_xy, std_psi,
            std_e_y, std_e_psi (= sqrt(std_psi^2 + (kappa * std along track)^2)),
            mode, flags, timing fields
    model   vx, vy from the no-slip KINEMATIC bicycle model at the CG
            (v = EKF speed, beta = atan(lr/(lf+lr) tan(steering))):
            vx = v cos(beta), vy = v sin(beta).  FLAG_VY_KINEMATIC is always set.
    rough   std_vx, std_vy, std_r: simple parameter models until the EKF
            exports its own velocity uncertainty (racing-speed work)
    zero    ax, ay ("recommended" fields; not estimated yet)
    unset   FLAG_OFF_TRACK (needs track widths in apex_track),
            FLAG_WHEEL_SLIP (needs the a_x slip detector)

REFERENCE POINT
    The EKF propagates position along yaw + beta (the CG velocity direction of
    the kinematic bicycle model), i.e. its point is the CG, like the gym's
    single-track model. `ref_to_cg_m` = distance from the EKF point forward to
    the CG: 0.0 here; on the real car set it if the EKF reference is the rear
    axle (then = lr). Checked against sim truth in the evaluation step.

MODES (contract section 6, thresholds provisional = parameters)
    INIT     until the first accepted AMCL correction and std_xy <= 0.15 m
    OK       std_xy <= 0.15 m and an accepted AMCL fix within the last 0.3 s of DRIVING
    DEGRADED std_xy 0.15-0.30 m, or 0.3-1.0 s of driving without an accepted fix
    LOST     std_xy > 0.30 m, or > 1.0 s of driving without an accepted fix
    LiDAR staleness counts DRIVING time only: AMCL deliberately stops updating
    while the car stands still (update_min_d), which is not a fault -- and after
    a long wait at the start line the car must not be LOST the moment it sets off.

MERGED PIPELINE (2026-10-03): state_pipeline.py runs this node in the same process
as both EKFs (own_inputs=False) and calls odom_cb / wheel_cb / imu_cb / fix_cb
directly -- one IMU and one wheel subscription for all three nodes.

Real car: identical node. Only the inputs change (STM32 bridge, MPU-6050
driver, rplidar), see the implementation guide.
"""
import math
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from apex_msgs.msg import StateEstimate as SE
from apex_track import load_track, wrap
from state_estimation.calibration import load_steering
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

SCHEMA = 'apex-state-estimate/1.0'


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp_us(t):
    return int(t.nanoseconds // 1000)


class StateEstimateNode(Node):

    def __init__(self, own_inputs=True):
        # own_inputs=False: state_pipeline.py (same process as both EKFs) calls odom_cb,
        # wheel_cb, imu_cb and fix_cb directly instead of through subscriptions
        super().__init__('state_estimate')
        d = self.declare_parameter
        d('track_yaml', '')                    # '' = apex_track's tracks/<track>/track.yaml
        d('track', 'levine')
        d('lf', 0.15875); d('lr', 0.17145)     # same values as ekf_node (shared vehicle config later)
        d('ref_to_cg_m', 0.0)                  # EKF point -> CG along body x (see docstring)
        d('valid_for_ms', 30)
        # provisional thresholds (contract section 6)
        d('std_ok_m', 0.15)
        d('std_lost_m', 0.30)
        d('lidar_stale_s', 0.3)
        d('lidar_lost_s', 1.0)
        d('moving_speed', 0.1)                 # m/s; below = standing still
        d('imu_stale_s', 0.025)                # sim IMU 100 Hz (real car 200 Hz -> 0.010)
        d('wheel_stale_s', 0.05)               # sim wheel 50 Hz (real car -> 0.030)
        d('correction_step_m', 0.05)
        d('near_singular', 0.3)                # flag when 1 - kappa*e_y < this
        d('start_zone_m', 2.0)                 # starting this far BEHIND the line -> lap -1
        # rough velocity uncertainty models (until the EKF exports them)
        d('std_vx_rel', 0.01); d('std_vx_min', 0.02)
        d('std_vy_rel', 0.05); d('std_vy_min', 0.02)
        d('std_r', 0.01)
        d('stats_every_s', 5.0)
        # steering calibration (steer_calib) for beta -> vx, vy; 'none' = measured steering as it is
        d('steer_calib_file', '~/.ros/steering_calibration.yaml')
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        path = p('track_yaml') or (get_package_share_directory('apex_track')
                                   + f"/tracks/{p('track')}/track.yaml")
        self.track = load_track(path)
        self.get_logger().info(f'track {self.track.track_id}, hash {self.track.track_hash}, '
                               f'L = {self.track.L:.2f} m ({path})')
        self.steer_cal, problem = load_steering(p('steer_calib_file'))
        if problem:
            self.get_logger().warn(problem)
        else:
            self.get_logger().info(self.steer_cal.describe())

        self.seq = 0
        self.steer = 0.0
        self.last_wheel_t = None               # node clock (s) of the last /vehicle/measured
        self.last_imu_t = None
        self.last_fix_t = None                 # node clock (s) of the last ACCEPTED AMCL fix
        self.driven_since_fix = 0.0            # s of DRIVING since that fix (standing doesn't count)
        self.prev = None                       # (t_s, x, y, v, psi) of the previous EKF message
        self.reset_pending = False
        self.mode = SE.MODE_INIT
        self.first = True
        self.lat_us, self.proc_us, self.mode_count = [], [], [0, 0, 0, 0]
        self.clock_steps = 0                   # messages published BEFORE their own stamp

        self.pub = self.create_publisher(SE, '/state_estimate', 10)
        if own_inputs:
            self.create_subscription(Odometry, '/odom/ekf', self.odom_cb, 50)
            self.create_subscription(AckermannDriveStamped, '/vehicle/measured', self.wheel_cb, 50)
            self.create_subscription(Imu, '/imu/data', self.imu_cb, 100)
            self.create_subscription(Float64, '/ekf/nis', self.fix_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.reset_cb, 10)
        self.create_timer(p('stats_every_s'), self.report)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # --- inputs used only for health ------------------------------------------------
    def wheel_cb(self, msg):
        self.steer = self.steer_cal.correct(msg.drive.steering_angle)
        self.last_wheel_t = self.now_s()

    def imu_cb(self, _msg):
        self.last_imu_t = self.now_s()

    def fix_cb(self, _msg):
        self.last_fix_t = self.now_s()
        self.driven_since_fix = 0.0

    def reset_cb(self, _msg):
        self.reset_pending = True

    # --- main: one StateEstimate per EKF message -----------------------------------
    def odom_cb(self, odom):
        t_wall0 = time.perf_counter()
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        now = self.now_s()
        t_valid = Time.from_msg(odom.header.stamp)

        x_r = odom.pose.pose.position.x
        y_r = odom.pose.pose.position.y
        psi = yaw_from_quat(odom.pose.pose.orientation)
        v = odom.twist.twist.linear.x          # EKF speed k*v_meas (along the CG velocity)
        r = odom.twist.twist.angular.z         # gyro, bias removed by the EKF

        # CG position (no-op in sim, ref_to_cg_m = 0)
        c, s = math.cos(psi), math.sin(psi)
        x = x_r + p('ref_to_cg_m') * c
        y = y_r + p('ref_to_cg_m') * s

        # kinematic body velocities at the CG
        lf, lr = p('lf'), p('lr')
        beta = math.atan(lr / (lf + lr) * math.tan(self.steer))
        vx, vy = v * math.cos(beta), v * math.sin(beta)

        # Frenet (apex_track): continuous s_abs, lap counting, reseed detection
        s_abs, s_tr, e_y, e_psi, kappa = self.track.project(x, y, psi)
        if self.first and s_tr > self.track.L - p('start_zone_m'):
            self.track.lap = -1                # started just behind the line
            s_abs -= self.track.L
        lap = self.track.lap

        # uncertainty
        cov = np.array(odom.pose.covariance).reshape(6, 6)
        Pxy = cov[:2, :2]
        std_xy = math.sqrt(max(float(np.linalg.eigvalsh(Pxy).max()), 0.0))
        std_psi = math.sqrt(max(cov[5, 5], 0.0))
        th = psi - e_psi                       # centre-line tangent heading
        n = np.array([-math.sin(th), math.cos(th)])
        std_e_y = math.sqrt(max(float(n @ Pxy @ n), 0.0))
        # e_psi = psi - theta(s). An error ds in s reads the tangent at the wrong place, and in
        # a corner theta turns kappa rad per metre -> e_psi error ~ -kappa*ds. So sigma(e_psi)
        # also contains the position uncertainty ALONG the track (run se_b, 2026-10-02: 4-5 deg
        # e_psi spikes in Levine's corners while sigma(psi) was ~0.3 deg -> only 67 % in 2 sigma).
        tg = np.array([math.cos(th), math.sin(th)])
        var_s = max(float(tg @ Pxy @ tg), 0.0)
        std_e_psi = math.sqrt(std_psi ** 2 + kappa ** 2 * var_s)

        # health
        moving = abs(v) > p('moving_speed')
        # staleness = DRIVING time since the last accepted fix. AMCL deliberately pauses while
        # the car stands still, so standing time is not a fault (run se_c 2026-10-02: the car
        # waited 2.4 s after its first fix and was reported LOST for 0.26 s when it set off).
        if moving and self.prev is not None:
            dt_drive = (t_valid.nanoseconds * 1e-9) - self.prev[0]
            if 0.0 < dt_drive < 0.5:
                self.driven_since_fix += dt_drive
        fix_age = None if self.last_fix_t is None else self.driven_since_fix
        lidar_stale = moving and (fix_age is None or fix_age > p('lidar_stale_s'))
        flags = SE.FLAG_VY_KINEMATIC
        if lidar_stale:
            flags |= SE.FLAG_LIDAR_STALE
        if self.last_imu_t is not None and now - self.last_imu_t > p('imu_stale_s'):
            flags |= SE.FLAG_IMU_STALE
        if self.last_wheel_t is None or now - self.last_wheel_t > p('wheel_stale_s'):
            flags |= SE.FLAG_WHEEL_STALE
        if self.prev is not None:
            dt = (t_valid.nanoseconds * 1e-9) - self.prev[0]
            if 0.0 < dt < 0.5:                 # position jump beyond what the motion explains
                px = self.prev[1] + self.prev[3] * math.cos(self.prev[4]) * dt
                py = self.prev[2] + self.prev[3] * math.sin(self.prev[4]) * dt
                if math.hypot(x - px, y - py) > p('correction_step_m'):
                    flags |= SE.FLAG_CORRECTION_STEP
        if self.reset_pending:
            flags |= SE.FLAG_FILTER_RESET
            self.reset_pending = False
        if self.track.reseeded:
            flags |= SE.FLAG_PROJECTION_RESEEDED
        if 1.0 - kappa * e_y < p('near_singular'):
            flags |= SE.FLAG_NEAR_SINGULAR

        # mode
        if self.mode == SE.MODE_INIT and (self.last_fix_t is None or std_xy > p('std_ok_m')):
            mode = SE.MODE_INIT                # stays INIT until the first good fix
        elif std_xy > p('std_lost_m') or (lidar_stale and fix_age is not None
                                          and fix_age > p('lidar_lost_s')):
            mode = SE.MODE_LOST
        elif std_xy > p('std_ok_m') or lidar_stale:
            mode = SE.MODE_DEGRADED
        else:
            mode = SE.MODE_OK
        if mode != self.mode:
            self.get_logger().info(f'mode {self.mode} -> {mode} (std_xy {std_xy*100:.1f} cm, '
                                   f'driven since fix {fix_age if fix_age is None else round(fix_age, 2)} s)')
        self.mode = mode

        # message
        m = SE()
        m.header.stamp = odom.header.stamp     # valid time, unchanged
        m.header.frame_id = odom.header.frame_id or 'map'
        m.schema_version = SCHEMA
        m.sequence_id = self.seq
        m.stamp_us = stamp_us(t_valid)
        m.track_id, m.track_hash = self.track.track_id, self.track.track_hash
        m.x_m, m.y_m, m.psi_rad = x, y, wrap(psi)
        m.vx_mps, m.vy_mps, m.r_radps = vx, vy, r
        m.s_abs_m, m.e_y_m, m.e_psi_rad = s_abs, e_y, e_psi
        m.s_track_m, m.lap = s_tr, lap
        m.ax_mps2 = m.ay_mps2 = 0.0
        m.std_xy_m, m.std_psi_rad = std_xy, std_psi
        m.std_vx_mps = max(p('std_vx_min'), p('std_vx_rel') * abs(vx))
        m.std_vy_mps = max(p('std_vy_min'), p('std_vy_rel') * abs(vx))
        m.std_r_radps = p('std_r')
        m.std_e_y_m, m.std_e_psi_rad = std_e_y, std_e_psi
        m.mode, m.flags = mode, flags
        m.valid_for_ms = p('valid_for_ms')
        m.publish_us = stamp_us(self.get_clock().now())   # last thing before sending
        self.pub.publish(m)

        self.seq = (self.seq + 1) % (1 << 32)
        self.first = False
        self.prev = (t_valid.nanoseconds * 1e-9, x, y, v, psi + beta)
        lat = m.publish_us - m.stamp_us
        if lat < 0:
            # only possible if the system clock stepped BACKWARDS while this message was in
            # flight (WSL does this). Not estimator latency -> counted, kept out of the stats.
            self.clock_steps += 1
            if self.clock_steps == 1 or self.clock_steps % 10 == 0:
                self.get_logger().warn(f'published {-lat / 1000:.1f} ms BEFORE its stamp -- the '
                                       f'system clock stepped backwards ({self.clock_steps} so far). '
                                       'Real car: time sync must slew, not step.')
        else:
            self.lat_us.append(lat)
        self.proc_us.append((time.perf_counter() - t_wall0) * 1e6)
        self.mode_count[mode] += 1

    def report(self):
        if not self.lat_us:
            if self.seq == 0:
                self.get_logger().info('waiting for /odom/ekf ...')
            return
        lat, proc = np.array(self.lat_us) / 1000.0, np.array(self.proc_us)
        n = len(lat)
        names = ['INIT', 'OK', 'DEGRADED', 'LOST']
        modes = ' '.join(f'{names[i]} {c}' for i, c in enumerate(self.mode_count) if c)
        self.get_logger().info(
            f'{n} msgs ({n / self.get_parameter("stats_every_s").value:.0f} Hz) | latency stamp->publish '
            f'mean {lat.mean():.1f} p95 {np.percentile(lat, 95):.1f} max {lat.max():.1f} ms | '
            f'node work mean {proc.mean():.0f} us | modes: {modes} | lap {self.track.lap}'
            + (f' | clock steps so far: {self.clock_steps}' if self.clock_steps else ''))
        self.lat_us, self.proc_us, self.mode_count = [], [], [0, 0, 0, 0]


def main():
    rclpy.init()
    node = StateEstimateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
