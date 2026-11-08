"""wheel_stuck.py -- F12 "wheel stuck at 0 while driving": a DEAD encoder's readings are treated as MISSING wheel data
(2026-11-08, decision Fatma). numpy-free, no ROS; tests: test/test_wheel_stuck.py.

WHY (job 252, BEFORE): a dead encoder (loose magnet, broken wire) keeps SENDING wheel messages that read 0 while the car
drives. The slip check (slip_check.py) catches it after ~0.1 s (2 samples compared 30 ms later) -- but at 6 m/s that is
~36 cm lost at once, its IMU speed then ran ~0.23 m/s low, and after max_on (3 s) it gives up and re-syncs to the wheel
(= 0) -> wheel 0 for 0.3 s = "PARKED" -> ZUPTs while driving: 16 m off in the 10-s case. Results: 1 s 55 cm, 3 s 96 cm,
10 s 16 m, all with the uncertainty claiming 2-4 cm.

WHAT: a stuck reading is not a measurement, so it is DROPPED before the EKFs / state_estimate see it. Then Fix 4's tested
path for missing wheel data (wheel_gap.py, F11) takes over: the IMU speed, never "parked", the along-track uncertainty
grows honestly, FLAG_WHEEL_STALE -> DEGRADED -> LOST after wheel_lost_s (contract v0.4: wheel data missing 1 s -> LOST).
  STUCK  on the FIRST sample with |v| < v_zero while the last good wheel speed or the IMU speed is > v_min: going from
         0.5 m/s to 0 within one wheel period (20 ms) would take 25 m/s^2 ~ 2.5 g -- this car cannot brake like that, so
         such a 0 is a dead sensor (or wheels locked in a skid: also not a speed to trust). Nothing is lost: the bad
         sample never reaches the EKF.
  STAYS  while the wheel reads ~0 -- even if the car then really stops (a dead sensor must not produce ZUPTs).
  BACK   after k_back samples in a row reading |v| >= v_zero with physically possible changes between them
         (|dv| <= a_max * dt + dv_tol); the k_back-th is delivered, the EKF re-syncs as after a dropout.
  A real stop is never "stuck": the wheel slows down through v_min (the last good speed is small) and the IMU speed with it.
Inputs in the wheel message's raw units (true = k * raw, k ~ 0.97: the thresholds are coarse on purpose).
REAL CAR: the ESP32's wheel rate decides the "one period" argument (100 Hz: even more margin); embedded item B
(flag zero counts while the ESC is driven) would confirm it directly.
"""


class StuckWheel:

    def __init__(self, v_zero=0.05, v_min=0.5, k_back=3, a_max=12.0, dv_tol=0.3):
        if not (0.0 < v_zero < v_min) or k_back < 1 or a_max <= 0.0 or dv_tol < 0.0:
            raise ValueError('StuckWheel needs 0 < v_zero < v_min, k_back >= 1, a_max > 0, dv_tol >= 0')
        self.v_zero, self.v_min, self.k_back = float(v_zero), float(v_min), int(k_back)
        self.a_max, self.dv_tol = float(a_max), float(dv_tol)
        self.stuck = False
        self.last_good = None          # (t, v) of the newest delivered sample
        self.t_start = None
        self.back = []                 # [(t, v)] candidate samples of a return
        self.events = []               # ('stuck', t, last_good_v, v_imu) | ('back', t, duration_s, n_dropped)
        self.n_dropped = 0
        self.dropped_total = 0

    def check(self, t, v, v_imu=None):
        """a wheel sample (stamp t [s], speed v); v_imu = the IMU (observer) speed or None.
        -> True = DROP it (the wheel is stuck), False = deliver it"""
        av = abs(float(v))
        if not self.stuck:
            ref = max(abs(self.last_good[1]) if self.last_good else 0.0, abs(v_imu) if v_imu is not None else 0.0)
            if av < self.v_zero and ref > self.v_min:
                self.stuck, self.t_start, self.back, self.n_dropped = True, float(t), [], 1
                self.dropped_total += 1
                self.events.append(('stuck', float(t), self.last_good[1] if self.last_good else None, v_imu))
                return True
            self.last_good = (float(t), float(v))
            return False
        if av >= self.v_zero:
            if self.back:
                tp, vp = self.back[-1]
                if abs(float(v) - vp) > self.a_max * max(float(t) - tp, 0.0) + self.dv_tol:
                    self.back = []                     # a jump: not a working encoder yet
            self.back.append((float(t), float(v)))
            if len(self.back) >= self.k_back:
                self.stuck = False
                self.events.append(('back', float(t), float(t) - self.t_start, self.n_dropped))
                self.last_good, self.back = (float(t), float(v)), []
                return False
        else:
            self.back = []
        self.n_dropped += 1
        self.dropped_total += 1
        return True

    def pop_events(self):
        ev, self.events = self.events, []
        return ev
