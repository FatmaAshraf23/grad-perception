#!/usr/bin/env python3
"""Steering calibration from a slow drive (REAL CAR tool, also used in simulation).

Finds   delta_true = gain * (delta_measured - offset)
(servo trim off-centre -> offset; linkage ratio -> gain) and writes it to
~/.ros/steering_calibration.yaml, which ekf_node and state_estimate read (calibration.py).

WHY
    The EKF uses the steering angle only for the slip angle at the CG,
    beta = atan(lr / L * tan(delta)). A steering offset of +0.01 rad makes beta 0.30 deg
    too large; AMCL keeps the direction of MOTION (heading + beta) right, so the EKF's
    heading settles 0.30 deg too low -- most of the constant -0.45 deg heading offset in
    every evaluation run (2026-10-02). StateEstimate's vy = v * sin(beta) is off too.

HOW -- no tape measure needed
    A car rolling without tyre slip (slow, little sideways acceleration) follows the
    kinematic bicycle model:   yaw rate r = v * tan(delta_true) / L
        ->  delta_true = atan(L * r / v)
    r = gyro minus its bias (measured here whenever the car stands still >= 1 s),
    v = wheel speed x the speed-scale anchor of wheel_calib, L = wheelbase.
    Every 0.1 s of STEADY driving (speed min_speed..max_speed, |v r| < max_lat_acc,
    steering and yaw rate not changing within window_s) gives one pair
    (delta_measured, delta_true).
      offset = median of (delta_measured - delta_true) over the nearly straight pairs
               (|delta_true| < straight_band). It does not depend on L, the speed scale
               or tyre slip: they all scale delta_true, which is ~0 there.
      gain   = slope of delta_true over (delta_measured - offset), least squares through
               zero, from the SLOW pairs only (v <= gain_max_speed, default 0.5 m/s) if they
               span >= min_gain_span rad; else 1.0 ("not determined").
               Why only slow pairs (job 045): at 0.8 m/s the simulated car already
               UNDERSTEERS (tyres slip, it turns ~3 % less than r = v tan(delta)/L), and the
               fit read that as a gain of 0.967 although the steering sensor has gain 1.0.
               The calibration describes the steering ACTUATOR (servo trim, linkage); tyre
               behaviour belongs to the vehicle-dynamics model (vy work). The "apparent
               gain" of the faster pairs is reported as information (understeer).

DRIVE (real car): wheel_calib first (speed anchor). Car stands still >= 1 s, then drive
    slowly for 2+ minutes, straights AND curves in both directions, smooth steering. For
    the gain, part of it at walking speed (0.3..0.5 m/s); otherwise only the offset is
    calibrated (gain 1) -- or measure the gain with tape (full-lock circle). Ctrl+C -> file written (also every 10 s once valid). Checks before
    writing: >= min_straight straight pairs, |offset| <= 0.15 rad, gain 0.7..1.3.
    Cross-check with tape (F1TENTH guide): with steering command = -offset the car drives
    straight (< 5 cm drift over 5 m). Give the offset to control / vehicle integration:
    on the real car the SERVO is off-centre, so commands need the same trim (shared
    measured vehicle file).

SIMULATION: -p use_sim_truth:=true also estimates from the simulator's TRUE yaw rate and
    speed (checks the gyro method); fake_vehicle_sensors adds steering_bias +0.01 rad
    (the true offset) and no gain error.

Input : /vehicle/measured (wheel speed, steering), /imu/data (gyro z)
        [/ego_racecar/odom only with use_sim_truth]
Output: out_file (default ~/.ros/steering_calibration.yaml)
"""
import math
import os
import statistics
import time
from collections import deque

try:                                      # installed package
    from state_estimation.calibration import GAIN_LIMITS, OFFSET_LIMIT_RAD, read_numbers
except ImportError:                       # next to calibration.py (offline / staging)
    from calibration import GAIN_LIMITS, OFFSET_LIMIT_RAD, read_numbers


class SteeringCalibrator:
    """Pure Python: steady (delta_measured, delta_true) pairs -> offset and gain."""

    def __init__(self, wheelbase, speed_scale=1.0, min_speed=0.25, max_speed=1.2,
                 max_lat_acc=1.0, window_s=0.5, steer_step=0.01, rate_step=0.03,
                 straight_band=0.03, sample_every_s=0.1, standstill_s=1.0, min_gain_span=0.1,
                 gain_max_speed=0.5):
        self.L, self.k = wheelbase, speed_scale
        self.min_speed, self.max_speed, self.max_lat_acc = min_speed, max_speed, max_lat_acc
        self.window, self.steer_step, self.rate_step = window_s, steer_step, rate_step
        self.straight_band, self.sample_every = straight_band, sample_every_s
        self.standstill, self.min_gain_span = standstill_s, min_gain_span
        self.gain_max_speed = gain_max_speed
        self.hist = deque()           # (t, delta_measured, yaw rate without bias, v)
        self.bias_sum, self.bias_n = 0.0, 0
        self.stand_since = None
        self.last_pair_t = -1e9
        self.pairs = []               # (delta_measured, delta_true, speed)

    def gyro_bias(self):
        return self.bias_sum / self.bias_n if self.bias_n else None

    def add(self, t, v_meas, delta_meas, gyro):
        """One reading (IMU rate): raw wheel speed [m/s], steering [rad], gyro z [rad/s]."""
        if abs(v_meas) < 0.01:                      # standing: the gyro reads its bias
            if self.stand_since is None:
                self.stand_since = t
            if t - self.stand_since >= self.standstill:
                self.bias_sum += gyro
                self.bias_n += 1
            self.hist.clear()
            return
        self.stand_since = None
        b = self.gyro_bias()
        if b is None:
            return                                  # no standstill yet: bias unknown
        self.hist.append((t, delta_meas, gyro - b, self.k * v_meas))
        while t - self.hist[0][0] > self.window:
            self.hist.popleft()
        if t - self.last_pair_t < self.sample_every or t - self.hist[0][0] < 0.9 * self.window:
            return
        n = len(self.hist)
        if n < 10:
            return
        d = [h[1] for h in self.hist]
        r = [h[2] for h in self.hist]
        half = n // 2
        if (abs(statistics.fmean(d[half:]) - statistics.fmean(d[:half])) > self.steer_step or
                abs(statistics.fmean(r[half:]) - statistics.fmean(r[:half])) > self.rate_step):
            return                                  # steering or yaw rate still changing
        v = statistics.fmean(h[3] for h in self.hist)
        if not self.min_speed <= v <= self.max_speed:
            return
        r_m = statistics.fmean(r)
        if abs(v * r_m) > self.max_lat_acc:
            return                                  # tyres slip: kinematic model invalid
        self.pairs.append((statistics.fmean(d), math.atan(self.L * r_m / v), v))
        self.last_pair_t = t

    def _fit_gain(self, pairs, off):
        """(gain, sigma, span, n) through zero, or (None, nan, span, n) if not enough spread."""
        xs = [dm - off for dm, _, _ in pairs]
        ys = [dt for _, dt, _ in pairs]
        span = (max(xs) - min(xs)) if xs else 0.0
        sxx = sum(x * x for x in xs)
        if span < self.min_gain_span or len(xs) < 50 or sxx <= 0.0:
            return None, float('nan'), span, len(xs)
        g = sum(x * y for x, y in zip(xs, ys)) / sxx
        s2 = sum((y - g * x) ** 2 for x, y in zip(xs, ys)) / max(len(xs) - 1, 1)
        return g, math.sqrt(s2 / sxx) * math.sqrt(5.0), span, len(xs)

    def estimate(self):
        """dict with offset, offset_sigma, n_straight, gain, gain_sigma, gain_determined,
        n_pairs, span (of the slow pairs), apparent_gain / apparent_speed (faster pairs,
        information only) -- or None (fewer than 20 straight pairs)."""
        straight = [dm - dt for dm, dt, _ in self.pairs if abs(dt) < self.straight_band]
        if len(straight) < 20:
            return None
        off = statistics.median(straight)
        mad = statistics.median(abs(x - off) for x in straight)
        # neighbouring pairs are correlated (0.5 s window, a pair every 0.1 s): ~1 in 5 independent
        off_sig = 1.4826 * mad / math.sqrt(max(len(straight) / 5.0, 1.0))
        slow = [q for q in self.pairs if q[2] <= self.gain_max_speed]
        fast = [q for q in self.pairs if q[2] > self.gain_max_speed]
        g, g_sig, span, _ = self._fit_gain(slow, off)
        ga, _, _, _ = self._fit_gain(fast, off)
        return {'offset': off, 'offset_sigma': off_sig, 'n_straight': len(straight),
                'gain': g if g is not None else 1.0, 'gain_sigma': g_sig,
                'gain_determined': g is not None, 'n_pairs': len(self.pairs), 'n_slow': len(slow),
                'span': span, 'apparent_gain': ga,
                'apparent_speed': statistics.fmean(q[2] for q in fast) if fast else float('nan')}


def check_estimate(est, min_straight):
    """None if the estimate may be written, else the reason it may not."""
    if est is None:
        return 'not enough steady straight driving yet'
    if est['n_straight'] < min_straight:
        return f"only {est['n_straight']} straight pairs (need {min_straight})"
    if abs(est['offset']) > OFFSET_LIMIT_RAD:
        return f"offset {est['offset']:+.3f} rad is implausible (> {OFFSET_LIMIT_RAD})"
    if not GAIN_LIMITS[0] <= est['gain'] <= GAIN_LIMITS[1]:
        return f"gain {est['gain']:.3f} is implausible (outside {GAIN_LIMITS})"
    return None


def write_calibration(path, est, info):
    tmp = path + '.tmp'
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write('# Steering calibration written by steer_calib:  '
                'delta_true = gain * (delta_measured - offset)\n')
        f.write(f"steering_offset_rad: {est['offset']:.6f}\n")
        f.write(f"steering_offset_sigma_rad: {est['offset_sigma']:.6f}\n")
        f.write(f"steering_gain: {est['gain']:.5f}\n")
        f.write(f"steering_gain_determined: {'true' if est['gain_determined'] else 'false'}\n")
        f.write(f"steering_gain_sigma: {est['gain_sigma']:.5f}\n")
        f.write(f"slow_pairs_for_gain: {est['n_slow']}\n")
        if est['apparent_gain'] is not None:
            f.write(f"apparent_gain_info: {est['apparent_gain']:.4f}   "
                    f"# at {est['apparent_speed']:.2f} m/s, includes tyre slip (understeer) -- not used\n")
        f.write(f"straight_pairs: {est['n_straight']}\n")
        f.write(f"all_pairs: {est['n_pairs']}\n")
        for k, v in info.items():
            f.write(f'{k}: {v}\n')
        f.write(f"saved: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# ROS 2 node
# ----------------------------------------------------------------------------

try:
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import Imu
except ImportError:
    Node = object


class SteerCalib(Node):

    def __init__(self):
        super().__init__('steer_calib')
        d = self.declare_parameter
        d('lf', 0.15875); d('lr', 0.17145)          # wheelbase L = lf + lr (same values as ekf_node)
        d('scale_file', '~/.ros/ekf_speed_scale.yaml')   # speed-scale anchor (wheel_calib)
        d('out_file', '~/.ros/steering_calibration.yaml')
        d('min_speed', 0.25); d('max_speed', 1.2)
        d('gain_max_speed', 0.5)                    # gain only from pairs at or below this speed
        d('max_lat_acc', 1.0)
        d('straight_band', 0.03)
        d('min_straight', 300)
        d('use_sim_truth', False)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.L = p('lf') + p('lr')
        self.out_file = os.path.expanduser(p('out_file'))
        k = 1.0
        sf = os.path.expanduser(str(p('scale_file')))
        try:
            k = float(read_numbers(sf)['speed_scale'])
        except (OSError, KeyError, ValueError):
            self.get_logger().warn(f'no speed scale in {sf} -> k = 1 (the offset is still right; '
                                   'the gain would be off by the wheel-speed error)')
        self.k = k
        kw = dict(min_speed=p('min_speed'), max_speed=p('max_speed'), max_lat_acc=p('max_lat_acc'),
                  straight_band=p('straight_band'), gain_max_speed=p('gain_max_speed'))
        self.cal = SteeringCalibrator(self.L, speed_scale=k, **kw)
        self.truth_cal = SteeringCalibrator(self.L, speed_scale=1.0, **kw) if p('use_sim_truth') else None
        self.v_meas, self.delta = 0.0, 0.0
        self.have_wheel = False
        self.written = None
        self.create_subscription(AckermannDriveStamped, '/vehicle/measured', self.wheel_cb, 50)
        self.create_subscription(Imu, '/imu/data', self.imu_cb, 100)
        if self.truth_cal is not None:
            self.create_subscription(Odometry, '/ego_racecar/odom', self.truth_cb, 10)
        self.create_timer(5.0, self.report)
        self.create_timer(10.0, self.save)
        self.get_logger().info(
            f'steering calibration: L = {self.L:.4f} m, speed scale k = {k:.4f}; stand still >= 1 s, '
            f'then drive slowly ({p("min_speed")}..{p("max_speed")} m/s), straights + gentle curves')

    @staticmethod
    def t_of(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def wheel_cb(self, msg):
        self.v_meas, self.delta = msg.drive.speed, msg.drive.steering_angle
        self.have_wheel = True

    def imu_cb(self, msg):
        if self.have_wheel:
            self.cal.add(self.t_of(msg.header.stamp), self.v_meas, self.delta, msg.angular_velocity.z)

    def truth_cb(self, msg):
        if self.have_wheel:
            tw = msg.twist.twist
            v = math.copysign(math.hypot(tw.linear.x, tw.linear.y), tw.linear.x)
            self.truth_cal.add(self.t_of(msg.header.stamp), v, self.delta, tw.angular.z)

    @staticmethod
    def fmt(est):
        if est is None:
            return 'not enough steady straight driving yet'
        g = (f"gain {est['gain']:.3f} +-{est['gain_sigma']:.3f} ({est['n_slow']} slow pairs)"
             if est['gain_determined'] else f"gain not determined ({est['n_slow']} slow pairs, span {est['span']:.3f} rad) -> 1.0")
        a = ('' if est['apparent_gain'] is None else
             f" | apparent gain {est['apparent_gain']:.3f} at {est['apparent_speed']:.2f} m/s (tyre slip, info)")
        return (f"offset {est['offset']:+.5f} rad ({math.degrees(est['offset']):+.3f} deg) "
                f"+-{est['offset_sigma']:.5f} | {g}{a} | pairs {est['n_straight']} straight / {est['n_pairs']} all")

    def report(self):
        b = self.cal.gyro_bias()
        line = (f"gyro bias {'n/a' if b is None else f'{b:+.5f} rad/s'} | GYRO method: "
                + self.fmt(self.cal.estimate()))
        if self.truth_cal is not None:
            line += ' || SIM TRUTH: ' + self.fmt(self.truth_cal.estimate())
        self.get_logger().info(line)

    def save(self, final=False):
        est = self.cal.estimate()
        why = check_estimate(est, self.get_parameter('min_straight').value)
        if why is not None:
            if final:
                self.get_logger().warn(f'NOT writing {self.out_file}: {why}')
            return
        info = {'speed_scale_used': f'{self.k:.6f}', 'wheelbase_m': f'{self.L:.5f}',
                'gyro_bias_rad_s': f'{self.cal.gyro_bias():.6f}'}
        write_calibration(self.out_file, est, info)
        if final or self.written is None:
            self.get_logger().info(f'wrote {self.out_file}: ' + self.fmt(est))
        self.written = est


def main(args=None):
    rclpy.init(args=args)
    node = SteerCalib()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.report()
        node.save(final=True)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
