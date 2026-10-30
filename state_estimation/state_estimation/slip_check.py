#!/usr/bin/env python3
"""slip_check.py -- WHEEL-SLIP CHECK: wheel speed vs the IMU's forward acceleration (Fix 3, 2026-10-26).
numpy only, no ROS: unit-tested in test/test_slip_check.py. Contract flag FLAG_WHEEL_SLIP (8) -> DEGRADED.

WHY
    The EKF moves the car with the wheel speed. Fault F05 (wheel x 1.15 for 1 s = wheelspin) gave 90 cm of
    error, F06 (x 0.85 for 0.6 s = skid) 52 cm, and AMCL cannot see it on a straight: it moves its particles
    with the same wheel odometry. The IMU's accelerometer measures how fast the speed CHANGES, independently
    of the wheels -> integrating it gives a second speed that a spinning or skidding wheel cannot fool.

HOW (SlipCheck)
    1. SPEED OBSERVER, every IMU sample:  v_obs += (a_x - bias) * dt
       While the wheel is trusted, every wheel sample pulls v_obs gently towards it (time constant tau,
       a complementary filter): the IMU gives the fast changes, the wheel stops the slow drift.
    2. RESIDUAL, every wheel sample:  r = v_wheel - v_obs
       The IMU in the simulator lags the true acceleration by ~35-40 ms (fake_imu smooths it) and the
       wheel by ~10 ms -> each wheel sample is compared with v_obs `lag` (30 ms) LATER. So a decision
       waits until the IMU has reached that moment (the sample sits in a queue meanwhile).
    3. DECISION with hysteresis (like map_check.MismatchAlarm, health.LagMonitor):
       ON   |r| > r0 + c_a * |a|  for k_on wheel samples in a row   (c_a * |a|: margin for lag mismatch
                                                                       in hard acceleration / braking)
       OFF  |r| < r_off for k_off samples in a row AND on for >= min_on
       MAX  on for > max_on -> give up: re-sync v_obs to the wheel, OFF (an IMU-only speed drifts with
            the bias error; after 3 s it is no better than a wheel that may have stopped slipping)
       While ON the observer runs on the IMU alone (no pull) -> `observer_speed` is the speed the EKF
       should use instead of the wheel.
    4. ACCELEROMETER BIAS: learned while the wheels stand still (|v| < 0.01 m/s for >= 0.3 s; then the
       true acceleration is 0, so a_x = bias; time constant 2 s). Also v_obs = 0 there.
       AND WHILE DRIVING (ki_bias, added 2026-10-26): every trusted wheel sample also moves the bias by
       -ki_bias * r * dt (an observer that keeps running ahead of the wheel, r < 0, integrates too much
       acceleration -> bias up). The complementary pull alone only corrects the SPEED; the bias error
       stayed and made the IMU-only speed drift as soon as a slip started. Why it matters: in a race the
       car never stops, but the accelerometer bias changes as the IMU warms up. Too large a ki_bias makes
       the bias follow the lag mismatch of hard acceleration (ki_bias 1.0: F05 flag stuck 1.2 s).
    OFF THRESHOLD GROWS WHILE ON (r_off_rate, added 2026-10-26): an IMU-only speed drifts by the bias error
       per second, so after t seconds of slip "agree again" means |r| < r_off + r_off_rate * t
       (capped at 0.9 r0 so a gap to the ON threshold always remains).
       Without the two additions (ki_bias 0, r_off_rate 0 = the prototype): a bias change of 0.1 m/s^2
       while driving left every F05/F06 flag stuck on until max_on and gave false alarms (table below).
    5. IMU GAP: no IMU sample for > gap (50 ms) -> the observer has nothing to integrate: re-sync to the
       wheel, OFF.
    6. START (2026-10-27, job 208): the observer starts from the first wheel sample whose stamp is within sync_tol
       (50 ms) of the newest IMU sample. In one simulator run the estimator got no wheel messages for the first
       ~3 s (the subscription connected late) and the first ones then arrived ~90 ms stale just as the car launched
       at 9.5 m/s^2: the observer started 0.87 m/s too low -> a false 3 s 'slip' -> the estimate was lost. Known limit: a slip that is already on when the IMU drops out is forgotten (the
       observer then starts again from the slipping wheel speed). That is Fix 2's territory (IMU gap).
    7. WHEEL DATA LOST (2026-10-30, Fix 4, wheel_gap.py): wheel_lost(t) when /vehicle/measured goes silent. The
       observer then runs on the IMU alone and is the speed the EKF predicts with: its "parked" memory is cleared
       (stale wheel data must not mean parked -- otherwise it would also stay at 0 when the car sets off), no bias
       learning (stopped is never true), a slip in progress ends ('wheel data lost'), an IMU gap no longer re-syncs
       to the stale wheel speed. The next wheel sample re-syncs the observer to the wheel like a start (sync_tol):
       no decision on it -- job 221 showed a false WHEEL SLIP right after the data came back.

KNOWN LIMITS
    - a wheel error below r0 (0.20 m/s, e.g. 5 % at 4 m/s) is not detected; the pull absorbs it.
    - a slip within the first ~3 s after a LARGE sudden bias change (0.1 m/s^2) may stay on until max_on
      (the driving bias learning needs ~6 s). Warm-up drift is slow, so this needs an unusual event.
    - very violent speed oscillation (+-9.5 m/s^2, the simulator's maximum, reversing every 0.15 s) gives
      false alarms even with the lag compensation: the IMU smoothing flattens such fast changes. The
      recorded racing runs never contain it. Without the lag compensation the recorded clean runs give
      false alarms in 17 of 30 runs -> `lag` is essential (and must be measured on the real car).

SIMULATOR ARTIFACT (SimGate)
    With fake_vehicle_sensors model v3 the wheel follows the distance per WALL-CLOCK second. When the
    simulator runs slower than real time the simulated wheel dips (down to 0.59x) while the IMU (from the
    simulator's speed state) shows nothing -- exactly like a skid, 2-14 % of the clean driving time (job
    204). fake_vehicle_sensors publishes /sim/wheel_ideal (error-free wheel + speed state); SimGate marks
    the moments where the error-free wheel itself reads < ratio x the speed state (+ hold_s after), and
    SlipCheck makes no decision and no correction for wheel samples in those moments.
    The REAL CAR has no such topic -> no gate (the EKF parameter sim_wheel_ideal_topic is empty).

DATA (offline prototype fix3_update/analysis/proto2.py, simulator, jobs 203-204)
    design set 15 runs, validation set 21 runs (not looked at while choosing the thresholds):
    clean runs 0 false slip episodes (13 + 17 runs); F05/F06 detected after 40-53 ms, 93-98 % of the fault
    flagged, flag off 82-236 ms after the fault ends. Distance error over the fault + 0.5 s: wheel only
    F05 +81..+88 cm, F06 -34..-38 cm; with the observer speed while flagged -0.3..+9.9 cm / -0.6..+3.7 cm.
    THIS CLASS (fix3_update/analysis/eval_slipcheck.py, 2026-10-26): with ki_bias 0 and r_off_rate 0 it
    reproduces proto2 exactly on all 36 runs. Validation set (21 runs), with a bias change while driving
    (added to the recorded a_x from 2 s after setting off):
        bias change   prototype rule                          defaults (ki_bias 0.3, r_off_rate 0.05)
        0             0 false; off 82-236 ms                  0 false; off 82-99 ms
        +0.1 m/s^2    2 false episodes; 4/4 faults stuck 3 s  0 false; off 82-139 ms
        -0.1 m/s^2    0 false; 2/4 faults stuck 3 s           0 false; off 82-99 ms
    Detection 40-53 ms and flagged share 93-98 % in all cases; observer error while flagged <= 0.10 m/s.

UNITS
    wheel speeds in TRUE m/s: the EKF passes k * v_raw (k = its learned wheel scale), so this class
    needs no scale of its own (the prototype hard-coded k = 1/1.029, the simulator's 3 % scale error).

REAL CAR (implementation guide)
    - measure the wheel-vs-IMU lag (step test like wheel_delay_calib) and set `lag`; the real IMU's filter
      is much faster than the simulator's 40 ms.
    - IMU near the car's centre, or correct the lever arm (an IMU l_x ahead of the centre also reads
      -r^2 * l_x in turns: 0.45 m/s^2 at 3 rad/s and 5 cm).
    - body pitch under hard braking / acceleration tilts the accelerometer (1 deg ~ 0.17 m/s^2): part of
      the c_a margin -- check on the real car.
    - normal tyre slip is a few % under hard acceleration: r0 must sit above it (measure on the track);
      encoder resolution at low speed sets a speed floor.

REFERENCES
    G. Reina, L. Ojeda, A. Milella, J. Borenstein, "Wheel slippage and sinkage detection for planetary
    rovers", IEEE/ASME Trans. Mechatronics 11(2), 2006.
    U. Kiencke, L. Nielsen, Automotive Control Systems, Springer 2005 (reference speed from wheels +
    accelerometer, ABS/TCS).
    R. Isermann, Fault-Diagnosis Systems, Springer 2006 (observer residuals, thresholds, hysteresis).
"""
import math
from collections import deque

import numpy as np


class SimGate:
    """SIMULATOR ONLY: when is a wheel sample a simulator-timing artifact (see the module docstring)?
    update(t, ideal, state) for every /sim/wheel_ideal message; gated(t) for a wheel sample's time."""

    def __init__(self, ratio=0.97, min_speed=0.5, hold_s=0.3):
        self.ratio, self.min_speed, self.hold_s = float(ratio), float(min_speed), float(hold_s)
        self.dips = deque(maxlen=200)          # times at which the error-free wheel dipped
        self.n_msgs = 0
        self.n_dips = 0

    def update(self, t, ideal, state):
        self.n_msgs += 1
        if abs(state) > self.min_speed and abs(ideal) < self.ratio * abs(state):
            self.dips.append(float(t))
            self.n_dips += 1

    def gated(self, t):
        """True if a dip happened in [t - hold_s, t] (the prototype: the last 15 wheel samples = 0.3 s)."""
        t = float(t)
        return any(t - self.hold_s <= d <= t + 1e-6 for d in reversed(self.dips))


class SlipCheck:
    """Wheel-slip check (module docstring). imu(t, a_x) per IMU sample, wheel(t, v) per wheel sample
    (v in true m/s); `on`, `observer_speed`; pop_events() for the log."""

    def __init__(self, lag=0.03, tau=0.5, r0=0.20, c_a=0.03, r_off=0.10, k_on=2, k_off=5,
                 min_on=0.2, max_on=3.0, gap=0.05, stop_speed=0.01, stop_s=0.3, sim_gate=None,
                 ki_bias=0.3, r_off_rate=0.05, b_max=0.5, sync_tol=0.05):
        if not r_off < r0:
            raise ValueError('SlipCheck needs r_off < r0 (hysteresis gap)')
        self.lag, self.tau, self.r0, self.c_a, self.r_off = float(lag), float(tau), float(r0), float(c_a), float(r_off)
        self.ki_bias = float(ki_bias)          # bias learning WHILE DRIVING from the residual [1/s^2] (0 = off)
        self.r_off_rate = float(r_off_rate)    # OFF threshold grows by this per s of slip [m/s/s] (0 = off)
        self.b_max = float(b_max)
        self.sync_tol = float(sync_tol)        # start the observer only from a wheel sample this close to the IMU time
        self.k_on, self.k_off = int(k_on), int(k_off)
        self.min_on, self.max_on, self.gap = float(min_on), float(max_on), float(gap)
        self.stop_speed, self.stop_s = float(stop_speed), float(stop_s)
        self.sim_gate = sim_gate               # SimGate or None (real car)
        self.reset()

    def reset(self):
        self.v = 0.0                           # observer speed [m/s]
        self.b = 0.0                           # accelerometer bias [m/s^2]
        self.a_f = 0.0                         # filtered acceleration (bias removed), for the margin
        self.t_imu = None                      # time of the newest IMU sample
        self.hist = deque(maxlen=100)          # (t, v_obs) of the last ~1 s of IMU samples
        self.pend = deque()                    # wheel samples waiting for the IMU to pass t + lag
        self.vw = 0.0                          # newest wheel speed
        self.synced = False                    # observer started from a wheel sample yet?
        self.t_last_w = None                   # time of the last decided wheel sample
        self.stop_since = None                 # wheel at ~0 since
        self._on = False
        self.n_on = 0                          # bad samples in a row (while off)
        self.n_off = 0                         # good samples in a row (while on)
        self.t_on = None
        self.last = None                       # (t, z, v_obs, r, thr, on) of the last decided sample
        self.events = deque(maxlen=50)         # (t, on, why, residual) not yet popped
        self.episodes = 0                      # statistics for the node's report
        self.on_s = 0.0
        self.gated_n = 0
        self.decided_n = 0
        self.resyncs = 0
        self.sync_waits = 0                    # stale wheel samples skipped while waiting for a current one
        self.lost = False                      # wheel data lost (wheel_lost, Fix 4): IMU-only until the next sample
        self.lost_n = 0

    # --- outputs ---------------------------------------------------------------------------------------

    @property
    def on(self):
        """True while wheel slip is detected (FLAG_WHEEL_SLIP)."""
        return self._on

    @property
    def observer_speed(self):
        """IMU-based speed [m/s] at the newest IMU sample: use it instead of the wheel while `on`."""
        return self.v

    @property
    def bias(self):
        return self.b

    def pop_events(self):
        out = list(self.events)
        self.events.clear()
        return out

    # --- inputs ----------------------------------------------------------------------------------------

    def imu(self, t, ax):
        """One IMU sample: time [s], forward acceleration in base_link [m/s^2]."""
        t = float(t)
        if self.t_imu is not None:
            dt = t - self.t_imu
            # a wheel at 0 WHILE SLIPPING is a locked wheel (hard braking), not a parked car: the car may
            # still slide at several m/s -- no v = 0, no bias learning from the braking deceleration
            stopped = (not self._on and self.stop_since is not None and t - self.stop_since >= self.stop_s)
            if dt > self.gap or dt <= 0.0:             # IMU gap (or time went back): re-sync to the wheel ...
                if not self.lost:                       # ... unless the wheel data is lost too (stale, Fix 4)
                    self.v = self.vw
                self.hist.clear()
                self.pend.clear()
                self.n_on = 0
                self.resyncs += 1
                if self._on:
                    self._set(False, t, 'IMU gap %.0f ms' % (dt * 1000.0), 0.0)
            elif stopped:                               # true acceleration 0: learn the bias
                self.b += min(1.0, dt / 2.0) * (ax - self.b)
                self.v = 0.0
            else:
                self.v += (ax - self.b) * dt
            self.a_f += 0.1 * ((ax - self.b) - self.a_f)
        self.t_imu = t
        self.hist.append((t, self.v))

    def wheel_lost(self, t):
        """The wheel data stopped (docstring 7): IMU-only observer, not parked, no bias learning, no slip decision."""
        if self.lost:
            return
        self.lost = True
        self.lost_n += 1
        self.stop_since = None
        self.pend.clear()
        self.n_on = 0
        if self._on:
            self._set(False, float(t), 'wheel data lost', 0.0)

    def wheel(self, t, v, gated=False):
        """One wheel sample: time [s], speed in TRUE m/s (k * v_raw). gated=True: a known simulator artifact
        (also checked with sim_gate at decision time). Decides every queued sample the IMU has passed by
        `lag`; returns their (t, z, v_obs, r, thr, on) rows (diagnostics, may be empty)."""
        t, v = float(t), float(v)
        if self.lost:                                  # first sample after a dropout: re-sync like a start
            self.lost = False
            self.synced = False
        self.vw = v
        if not self.synced:                            # first wheel sample (node started while moving?):
            if self.t_imu is not None and abs(t - self.t_imu) > self.sync_tol:
                self.sync_waits += 1                   # STALE (or the IMU far behind): wait for a CURRENT sample --
                return []                              # job 208: started from a 90 ms old one, 0.87 m/s low at launch
            self.v = v                                 # start the observer at the wheel speed, not at 0
            self.hist.clear()
            self.pend.clear()
            self.synced = True
        if abs(v) < self.stop_speed:
            if self.stop_since is None:
                self.stop_since = t
        else:
            self.stop_since = None
        self.pend.append((t, v, bool(gated)))
        out = []
        while self.pend and self.t_imu is not None and self.t_imu >= self.pend[0][0] + self.lag:
            tw, z, g = self.pend.popleft()
            if len(self.hist) < 2:
                continue
            ts = np.fromiter((h[0] for h in self.hist), float, len(self.hist))
            vs = np.fromiter((h[1] for h in self.hist), float, len(self.hist))
            vr = float(np.interp(tw + self.lag, ts, vs))
            r = z - vr
            thr = self.r0 + self.c_a * abs(self.a_f)
            dtw = 0.02 if self.t_last_w is None else min(max(tw - self.t_last_w, 0.0), 0.1)
            if self._on:
                self.on_s += dtw
            self.t_last_w = tw
            if g or (self.sim_gate is not None and self.sim_gate.gated(tw)):
                self.n_on = 0                          # simulator artifact: no decision, no correction
                self.gated_n += 1
            elif not self._on:
                self.decided_n += 1
                self.n_on = self.n_on + 1 if abs(r) > thr else 0
                if self.n_on >= self.k_on:
                    self._set(True, tw, 'wheel %+.2f m/s vs IMU' % r, r)
                else:
                    self.v += (1.0 - math.exp(-dtw / self.tau)) * r     # complementary pull towards the wheel
                    if self.ki_bias > 0.0:                               # observer too fast (r < 0) -> bias up
                        self.b = min(self.b_max, max(-self.b_max, self.b - self.ki_bias * r * dtw))
            else:
                self.decided_n += 1
                r_off = min(self.r_off + self.r_off_rate * (tw - self.t_on), 0.9 * self.r0)   # keep a gap to r0
                self.n_off = self.n_off + 1 if abs(r) < r_off else 0
                if tw - self.t_on > self.max_on:
                    self._set(False, tw, 'longer than max_on %.1f s: re-sync to the wheel' % self.max_on, r)
                    self.v = z
                    self.resyncs += 1
                elif self.n_off >= self.k_off and tw - self.t_on >= self.min_on:
                    self._set(False, tw, 'wheel and IMU agree again', r)
            self.last = (tw, z, vr, r, thr, self._on)
            out.append(self.last)
        return out

    def _set(self, on, t, why, r):
        self._on = on
        self.n_on = 0
        self.n_off = 0
        if on:
            self.t_on = t
            self.episodes += 1
        self.events.append((t, on, why, r))
