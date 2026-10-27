#!/usr/bin/env python3
"""imu_gap.py -- IMU DROPOUT: detect it, predict the yaw rate from wheel + steering, validate the returning gyro
(Fix 2, 2026-10-27; contract Option A, approved by control on 26 Oct for isolated development and synthetic evaluation).
No ROS, no numpy: unit-tested in test/test_imu_gap.py.

WHY
    ekf_node predicted ONLY when an IMU sample arrived. During a dropout the estimate stood still; when the IMU came
    back after > 0.5 s the whole gap was skipped (`dt > 0.5: return`) -- fault F04 (1 s, 6 m/s): ~6 m of driving lost in
    BOTH EKFs, AMCL (fed by ekf_local) lost too, the estimate never recovered (5.9 m / 86 deg off).

HOW (ImuGap, one per EKF)
    1. DETECT at every wheel sample: no IMU sample for more than gap_s (25 ms = 2.5 periods at 100 Hz; the recorded
       runs: 149 087 IMU intervals, max 19.7 ms) -> state GAP. Detection = gap_s + up to one wheel period after the
       last IMU sample.
       ...AND, when the caller passes receive times (rx = the time the callback ran, on the SAME clock as the stamps;
       ekf_node passes its ROS clock), this wheel message must be FRESH (rx - t <= gap_s) and the IMU silent in real
       time too (rx - rx_imu > gap_s). Why (job 207): with a processing backlog (start-up, a busy CPU) the executor
       handles one message per topic per cycle, the 100 Hz IMU queue falls behind the 50 Hz wheel queue, and the
       STAMPS alone said "dropout" 27 times in 0.35 s while the IMU never stopped -> 30 s of false DEGRADED/LOST at the
       start. In a backlog or after a whole-process stall the wheel messages are old too (handled in a burst), so they
       cannot prove that the IMU is silent; in a real dropout they arrive on time. Price: a real dropout during a
       backlog is detected only once the backlog is gone (the estimator-lag flag is set meanwhile).
    2. PREDICT while GAP (the EKF calls yaw_rate() at each wheel sample): kinematic bicycle from the wheel speed and
       the (calibrated) steering,  r = v tan(delta) / (L + kus v^2)  (kus = understeer, 0 in the simulator: the fit
       gave -0.0019 s^2/m), plus the model's error at the start of the gap (last trusted gyro - model) fading with
       tau_b (0.2 s). The EKF uses sigma_w_gap (0.4 rad/s per 20 ms step) as the yaw-rate noise: with it 97-100 % of
       the offline errors were inside 2 sigma (validation set, gaps 0.05-1 s).
    3. RECOVER: the first IMU sample after a gap -> state RECOVERING. The returning gyro is NOT trusted yet (control:
       "a returning sample must not automatically clear critical faults"): the EKF keeps predicting with the model
       and compares; rec_n samples in a row with |gyro - model| <= rec_tol -> OK, the gyro is used again.
       A disagreeing sample restarts the count; IMU silent again -> back to GAP (no offset: the gyro was not trusted).
    The estimator's gyro bias is not learned from untrusted samples (the EKF skips ZUPT then).

OFFLINE EVIDENCE (fix2_update/analysis/gap_eval.py, validation set, p95 at the end of the gap, heading / lateral)
    gap      today's code       this model (kinematic + fading gyro offset)
    0.05 s   0.37 deg / 0.4 cm  0.31 deg / 0.3 cm
    0.2 s    4.5 deg / 6.5 cm   1.7 deg / 1.1 cm
    0.5 s    23 deg / 44 cm     4.0 deg / 4.8 cm
    1.0 s    68 deg / 189 cm    6.5 deg / 17 cm   (today: + 4.5 m lost along the track)
    worst situation: steering transitions (0.5 s: ~5 deg / 6 cm); straights ~1.3 deg / 2 cm.

KNOWN LIMITS
    - a process stall whose queued messages are handled all-wheel-first (not the order of a single-threaded executor,
      which takes one message per topic per cycle) ends in one short false dropout (~0.1 s until the gyro is
      validated): the newest queued wheel message is fresh and no IMU was handled for the whole stall.
    - a real dropout during a processing backlog is detected only after the backlog (the lag flag is set meanwhile).

REAL CAR
    measure the IMU sample spacing at its real rate (gap_s ~ 2.5 periods); fit kus on real laps with the gyro (a real
    car understeers); the steering is the COMMANDED angle unless the servo reports its position (servo lag adds error
    in transitions); check the sigma_w_gap coverage on real data.
"""
import math

OK, GAP, RECOVERING = 0, 1, 2
STATE_NAMES = {OK: 'OK', GAP: 'GAP', RECOVERING: 'RECOVERING'}


class ImuGap:
    def __init__(self, gap_s=0.025, wheelbase=0.3302, kus=0.0, tau_b=0.2, rec_n=10, rec_tol=0.5, max_steer=0.6):
        if gap_s <= 0.0 or wheelbase <= 0.0 or tau_b <= 0.0 or rec_n < 1:
            raise ValueError('ImuGap needs gap_s > 0, wheelbase > 0, tau_b > 0, rec_n >= 1')
        self.gap_s, self.L, self.kus, self.tau_b = float(gap_s), float(wheelbase), float(kus), float(tau_b)
        self.rec_n, self.rec_tol, self.max_steer = int(rec_n), float(rec_tol), float(max_steer)
        self.state = OK
        self.t_imu = None          # newest IMU sample (any)
        self.rx_imu = None         # receive time of the newest IMU sample (same clock as the stamps), if given
        self.t_trusted = None      # newest TRUSTED gyro sample (the gap starts here)
        self.r_trusted = 0.0       # its yaw rate (bias removed)
        self.v, self.steer = 0.0, 0.0
        self.off0 = 0.0            # model error at the start of the gap (fades with tau_b)
        self.t_gap0 = None         # start of the current gap (= t_trusted)
        self.t_back = None         # first IMU sample after the gap
        self.n_ok = 0
        self.events = []           # (kind, t, a, b): ('gap', t_detect, t_last_trusted, 0), ('back', t, gap_s, 0),
                                   # ('valid', t, recovery_s, gap_s), ('reject', t, gyro - model, 0)
        self.episodes = 0
        self.longest = 0.0
        self.rejects = 0

    # --- model ---------------------------------------------------------------------------------------------
    def model_r(self, v, steer):
        d = max(-self.max_steer, min(self.max_steer, float(steer)))
        return float(v) * math.tan(d) / (self.L + self.kus * float(v) * float(v))

    def yaw_rate(self, t, v=None, steer=None):
        """Yaw rate to predict with [rad/s] at time t (while not OK): the model + the fading start offset."""
        v = self.v if v is None else v
        steer = self.steer if steer is None else steer
        r = self.model_r(v, steer)
        if self.t_gap0 is not None:
            r += self.off0 * math.exp(-max(0.0, t - self.t_gap0) / self.tau_b)
        return r

    @property
    def trusted(self):
        return self.state == OK

    def gap_len(self, t):
        """Time since the last trusted gyro sample [s] (0 while OK)."""
        return 0.0 if self.state == OK or self.t_gap0 is None else max(0.0, t - self.t_gap0)

    def pop_events(self):
        ev, self.events = self.events, []
        return ev

    # --- inputs --------------------------------------------------------------------------------------------
    def silent(self, t, rx=None):
        """Is the IMU silent at this wheel sample? Stamps: > gap_s since the newest IMU sample. Receive times (if
        given): this wheel sample is fresh AND no IMU sample was received for > gap_s (not a backlog / a stall)."""
        if self.t_imu is None or t - self.t_imu <= self.gap_s:
            return False
        if rx is None or self.rx_imu is None:
            return True
        return rx - t <= self.gap_s and rx - self.rx_imu > self.gap_s

    def wheel(self, t, v, steer, rx=None):
        """Every wheel sample: speed [m/s, true], steering [rad, calibrated], rx = receive time (same clock as the
        stamps, optional). Returns the state after it."""
        self.v, self.steer = float(v), float(steer)
        silent = self.silent(t, rx)
        if self.state == OK and silent:
            self.state = GAP
            self.t_gap0 = self.t_trusted if self.t_trusted is not None else self.t_imu
            self.off0 = self.r_trusted - self.model_r(v, steer)
            self.episodes += 1
            self.events.append(('gap', float(t), float(self.t_gap0), 0.0))
        elif self.state == RECOVERING and silent:
            self.state = GAP                   # silent again before the gyro was validated: no offset (not trusted)
            self.off0 = 0.0
            self.events.append(('gap', float(t), float(self.t_gap0), 0.0))
        return self.state

    def imu(self, t, r_gyro, rx=None):
        """Every IMU sample: time [s], yaw rate with the bias removed [rad/s], rx = receive time (optional).
        Returns True if the gyro is trusted."""
        t = float(t)
        self.t_imu = t if self.t_imu is None else max(self.t_imu, t)
        if rx is not None:
            self.rx_imu = float(rx)
        if self.state == GAP:
            self.state = RECOVERING
            self.t_back = t
            self.n_ok = 0
            g = t - self.t_gap0
            self.longest = max(self.longest, g)
            self.events.append(('back', t, g, 0.0))
        if self.state == RECOVERING:
            err = float(r_gyro) - self.yaw_rate(t)
            if abs(err) <= self.rec_tol:
                self.n_ok += 1
            else:
                if self.n_ok > 0 or not any(e[0] == 'reject' for e in self.events):
                    self.events.append(('reject', t, err, 0.0))
                self.rejects += 1
                self.n_ok = 0
            if self.n_ok >= self.rec_n:
                self.state = OK
                self.events.append(('valid', t, t - self.t_back, t - self.t_gap0))
                self.t_gap0 = None
            else:
                return False
        self.t_trusted, self.r_trusted = t, float(r_gyro)
        return True
