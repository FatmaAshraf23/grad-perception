#!/usr/bin/env python3
"""wheel_gap.py -- WHEEL-DATA DROPOUT: detect it and say how uncertain the IMU speed is meanwhile (Fix 4, 2026-10-30).
No ROS, no numpy: unit-tested in test/test_wheel_gap.py.

WHY
    ekf_node took the speed for its prediction from the LAST wheel message. When /vehicle/measured stops (a loose cable,
    the ESP32 resetting on a voltage dip, the bridge node crashing) the EKFs kept moving the car at that speed forever,
    and a car that was parked stayed "parked" (ZUPTs) while it drove off. Fault F11 (job 221, 6 m/s, all other fixes on):
    1 s while braking 6 -> 2 m/s: 1.56 m error; 3 s: 13.7 m; dropped while parked: 52.6 m, never recovered -- with the
    estimate's own uncertainty at 3-5 cm and the mode OK for up to 2.2 s.

HOW (WheelGap, one per EKF)
    1. DETECT at every IMU sample: no wheel sample for more than gap_s (50 ms = 2.5 periods at 50 Hz) by the stamps
       ...AND, when receive times are given (the ROS clock when the callbacks run), this IMU message is FRESH
       (rx - t <= gap_s) and no wheel message was HANDLED for > gap_s in real time -- a processing backlog or a
       stall is not a dropout (the same rule as imu_gap.ImuGap, job 207).
    2. PREDICT (ekf_node): with the slip check's IMU speed (slip_check.SlipCheck observer: a_x - bias integrated; its
       "parked" memory and bias learning are switched off for the dropout -- SlipCheck.wheel_lost), never "parked",
       no ZUPT. Offline (wheeldrop_update/analysis/wheelgap_eval.py, 17 validation runs, p95 distance error at the end
       of the gap): 0.5 s 7 cm, 1 s 12 cm, 2 s 13 cm (holding the last wheel speed: 46 cm, 1.44 m, 3.17 m);
       accelerating / braking 1 s: 14 cm vs 1.65 m; setting off from parked 1 s: 18 cm vs 2.97 m.
    3. UNCERTAINTY: the distance driven during a gap of length L is uncertain by
           sigma_d(L) = sv0 * L + 0.5 * sa * L^2
       sv0 = 0.14 m/s fitted offline (96-100 % of the errors inside 2 sigma, design AND validation runs); sa = 0.1
       m/s^2 for an accelerometer-bias change the observer has not learned yet (the simulator has none; on the real
       car the bias drifts while the IMU warms up). sigma_v_step(t, dt) is the per-step speed noise that makes the
       EKF's own covariance grow exactly like sigma_d -- so the reported std_xy says how unsure the estimate is.
    4. BACK: the first wheel sample ends the gap (ekf_node re-syncs the slip observer to it: no false WHEEL SLIP --
       job 221 showed one in F11c / F11d).
    state_estimate decides the MODE from its own clock (FLAG_WHEEL_STALE after wheel_stale_s -> DEGRADED, LOST after
    wheel_lost_s); this class feeds the EKF and the diagnostics topic <diag_prefix>/wheel_gap.

REAL CAR
    gap_s ~ 2.5 periods of the ESP32's real wheel rate; check sv0 / sa on real recordings (vibration, bias drift);
    the IMU's own filter delay (MPU-6050 DLPF) is much shorter than the simulator's 40 ms, which is most of sv0 here.
"""
import math

OK, GAP = 0, 1
STATE_NAMES = {OK: 'OK', GAP: 'GAP'}


class WheelGap:
    def __init__(self, gap_s=0.05, sv0=0.14, sa=0.1):
        if gap_s <= 0.0 or sv0 < 0.0 or sa < 0.0 or (sv0 == 0.0 and sa == 0.0):
            raise ValueError('WheelGap needs gap_s > 0, sv0 >= 0, sa >= 0, not both 0')
        self.gap_s, self.sv0, self.sa = float(gap_s), float(sv0), float(sa)
        self.state = OK
        self.t_wheel = None        # stamp of the newest wheel sample
        self.rx_wheel = None       # receive time of the newest wheel sample (same clock as the stamps), if given
        self.events = []           # ('lost', t_detect, t_last_wheel), ('back', t_wheel, gap_len)
        self.episodes = 0
        self.longest = 0.0

    # --- model ---------------------------------------------------------------------------------------------
    def sigma_d(self, length):
        """Uncertainty [m] of the distance driven in a gap of this length [s]."""
        L = max(0.0, float(length))
        return self.sv0 * L + 0.5 * self.sa * L * L

    def gap_len(self, t):
        """Time since the newest wheel sample [s] (0 while OK)."""
        return 0.0 if self.state == OK or self.t_wheel is None else max(0.0, float(t) - self.t_wheel)

    def sigma_v_step(self, t, dt):
        """Speed noise [m/s] for ONE EKF step of dt ending at t: the EKF adds (sigma * dt)^2 to the position variance
        along the heading, so sigma = sqrt(sigma_d(L)^2 - sigma_d(L - dt)^2) / dt makes its growth equal sigma_d."""
        if dt <= 0.0:
            return 0.0
        L = self.gap_len(t)
        a, b = self.sigma_d(L), self.sigma_d(max(0.0, L - dt))
        return math.sqrt(max(a * a - b * b, 0.0)) / dt

    def pop_events(self):
        ev, self.events = self.events, []
        return ev

    @property
    def lost(self):
        return self.state == GAP

    # --- inputs --------------------------------------------------------------------------------------------
    def silent(self, t, rx=None):
        """Is the wheel silent at this IMU sample? Stamps: > gap_s since the newest wheel sample. Receive times (if
        given): this IMU sample is fresh AND no wheel sample was received for > gap_s (not a backlog / a stall)."""
        if self.t_wheel is None or t - self.t_wheel <= self.gap_s:
            return False
        if rx is None or self.rx_wheel is None:
            return True
        return rx - t <= self.gap_s and rx - self.rx_wheel > self.gap_s

    def imu(self, t, rx=None):
        """Every IMU sample: time [s], rx = receive time (optional). Returns True when a gap STARTS here."""
        if self.state == OK and self.silent(float(t), rx):
            self.state = GAP
            self.episodes += 1
            self.events.append(('lost', float(t), float(self.t_wheel)))
            return True
        return False

    def wheel(self, t, rx=None):
        """Every wheel sample: time [s], rx = receive time (optional). Returns True when a gap ENDS here."""
        t = float(t)
        ended = False
        if self.state == GAP:
            g = t - self.t_wheel
            self.longest = max(self.longest, g)
            self.events.append(('back', t, g))
            self.state = OK
            ended = True
        self.t_wheel = t if self.t_wheel is None else max(self.t_wheel, t)
        if rx is not None:
            self.rx_wheel = float(rx)
        return ended
