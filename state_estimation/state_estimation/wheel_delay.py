"""Wheel-speed DELAY compensation with the IMU's forward acceleration (2026-10-19, AMCL-sliding work).

WHY
    The wheel speed reaches the EKF late, for two reasons:
      1. the sensor's own delay: an encoder speed is "counts over a time window", so it describes the
         MIDDLE of that window; add the link to the computer.  Simulator (fake_vehicle_sensors):
         10-12 ms (job-151 recordings).  Real car: half the STM32's counting window + the serial link
         -- measure it (README, "wheel_delay_s").
      2. the EKF predicts at every IMU message (100 Hz) with the NEWEST wheel speed and holds it until
         the next one arrives (50 Hz in the simulator): another 0..20 ms.
    While the speed is constant that is harmless.  While the car speeds up, the integrated distance
    falls behind:   position deficit ~ (speed gained) x (total delay)
    e.g. 2.1 -> 6 m/s with ~15 ms -> about 6 cm, and nothing on a featureless straight can see it: AMCL
    moves its particles with this same odometry (ekf_local) and its scans show no along-track feature.

WHAT
    The IMU measures the forward acceleration a_x, and acceleration integrates to a speed change.  So the
    speed NOW = the last wheel speed + what the car gained since that value was true:
        v_now = v_wheel + integral of (a_x - b_ax) dt   over [t_wheel - tau_w, now]
    tau_w = the wheel speed's own delay [s] (ekf_node parameter wheel_delay_s), b_ax = the
    accelerometer's bias (+ mounting tilt x g), learned while the car stands still.  The integral only
    spans tau_w + the gap to the next wheel message (10-30 ms) and restarts with every wheel message, so
    accelerometer noise and bias cannot pile up (0.1 m/s^2 of bias x 30 ms = 3 mm/s).  This is the
    usual complementary idea: the wheel gives the speed, the IMU only fills the short time between
    "when it was true" and now.

RESULTS (closed1, job-154 recordings, offline AMCL replays on the same scans, tau_w 10 ms)
    AMCL fed with the compensated odometry: along-track mean / p95 at 6 m/s -8.2 / 17.8 -> -4.8 / 12.8 cm,
    and the error no longer grows with speed (5-7 m/s band -13.7 -> -5.8 cm); 4.5 m/s p95 18.5 -> 15.1 cm;
    <= 3 m/s unchanged (little acceleration there).  The remaining ~-5 cm was AMCL's own (amcl_sim.yaml).

UNITS
    on_imu() returns the speed GAINED in true m/s (a_x is a true acceleration).  The EKF's wheel input is
    the RAW measured speed (true = k * raw), so the EKF adds gained / k to the raw value.
"""


class WheelSpeedCompensator:

    def __init__(self, tau_w=0.010, max_age=0.1, bias_tau=2.0):
        self.tau_w = float(tau_w)          # the wheel speed's own delay [s]; 0 = only bridge the hold
        self.max_age = float(max_age)      # s: never extrapolate a wheel value further than this
        self.bias_tau = float(bias_tau)    # s: averaging time of the bias estimate while standing
        self.b_ax = 0.0                    # accelerometer bias (+ tilt * g) [m/s^2]
        self.bias_ready = False
        self.t_w = None                    # stamp of the newest wheel speed [s]
        self.t_imu = None                  # stamp of the newest IMU message [s]
        self.dv = 0.0                      # speed gained since that value was true [m/s]
        self.last_ax = 0.0

    def on_wheel(self, t):
        """a new wheel speed with stamp t [s]: its value was true tau_w before t, so the gain starts with
        the newest acceleration x (tau_w + the IMU time already processed after t, if the wheel message
        came late)"""
        self.t_w = t
        ahead = max(0.0, self.t_imu - t) if self.t_imu is not None else 0.0
        self.dv = (self.last_ax - self.b_ax) * (self.tau_w + ahead)

    def on_imu(self, t, ax, dt, stationary):
        """an IMU message (stamp t [s], forward acceleration ax [m/s^2], dt [s] since the previous IMU
        message); returns the speed gained since the newest wheel value was true [true m/s]"""
        self.last_ax = ax
        self.t_imu = t
        if stationary:                     # standing: a_x is pure bias (+ tilt * g)
            a = min(1.0, dt / self.bias_tau) if self.bias_ready else 1.0
            self.b_ax += a * (ax - self.b_ax)
            self.bias_ready = True
            self.dv = 0.0
            return 0.0
        if self.t_w is None:
            return 0.0
        span = min(dt, t - self.t_w)       # only the part of this IMU interval AFTER the wheel stamp
        if span > 0.0 and t - self.t_w + self.tau_w <= self.max_age:
            self.dv += (ax - self.b_ax) * span
        return self.dv

    def describe(self):
        return (f'wheel-speed delay compensation ON: tau_w {1000 * self.tau_w:.1f} ms + the time to the next '
                'wheel message, bridged with the IMU forward acceleration (bias learned while standing)')
