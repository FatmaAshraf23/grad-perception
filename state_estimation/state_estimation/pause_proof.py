#!/usr/bin/env python3
"""SIMULATION ONLY: pause-proof simulated rate sensors (fake_imu / fake_vehicle_sensors v3).

Used by fake_imu (yaw rate from the heading) and fake_vehicle_sensors (wheel speed
from the distance travelled). Pure Python, no ROS: tested offline on recorded
simulator data with imu_model_offline.py (2026-10-03).

Real car: not used -- the MPU-6050 and the wheel encoder measure the real thing.
"""
import math
from collections import deque


class PauseProofRate:
    """fake_imu v3: a simulated RATE sensor that is smooth AND pause-proof.

    The simulator makes a new state every 10 ms of SIMULATOR time, but our sensors
    run on WALL time, and when the laptop is busy the simulator falls behind or
    pauses (2026-10-02 findings):
      v1  rate = how far the pose moved since my last reading / dt
          -> exact in total, but the 10 ms reading timer slides against the 10 ms
             physics timer: in turns ~80 % of the readings saw 0 or 2 steps (0x / 2x).
      v2  rate = the simulator's own rate state
          -> smooth, but in a pause it keeps reporting motion that never happened
             (phantom rotation, +12..+56 deg in sharp corners).
    v3 = both ideas together:
      feed-forward  the simulator's rate state x rho, rho = how fast the simulator
                    really progresses (1 = real time; 0 = stalled or standing: no
                    new state for hold_s)
      feedback      everything reported so far (its integral) is pulled towards what
                    the car really did according to its poses (extrapolated by at
                    most extrap_s), with time constant tau_s
    -> smooth like v2, and like v1 it cannot invent motion: in a pause the
       feed-forward switches off and the feedback removes any leftover.
    angle=True: the value is a heading (unwrapped here). max_jump: a change larger
    than this between two states is a teleport/reset, not motion -> not reported.
    """

    def __init__(self, step_s=0.01, hold_s=0.025, extrap_s=0.01, tau_s=0.2,
                 window_s=0.25, angle=True, max_jump=None):
        self.step, self.hold, self.extrap, self.tau = step_s, hold_s, extrap_s, tau_s
        self.window, self.angle, self.max_jump = window_s, angle, max_jump
        self.value = None            # true value at the latest state (unwrapped if angle)
        self.raw = None              # latest raw value (for unwrapping)
        self.rate = 0.0              # the simulator's rate at the latest state
        self.t_state = None          # arrival time of the latest state
        self.arrivals = deque()      # arrival times of recent states (for rho)
        self.reported = None         # integral of everything reported so far
        self.t_sample = None

    def on_state(self, t, value, rate):
        """A NEW simulator state arrived at time t (call it only when the pose changed)."""
        if self.value is None:
            self.value = self.raw = self.reported = value
        else:
            d = value - self.raw
            if self.angle:
                d = math.atan2(math.sin(d), math.cos(d))
            self.raw = value
            if self.max_jump is not None and abs(d) > self.max_jump:
                self.reported += d   # teleport/reset: move along, report nothing
            self.value += d
        if self.t_state is not None and t - self.t_state > self.hold:
            self.arrivals.clear()    # after a stall: measure the progress afresh
        self.rate = rate
        self.t_state = t
        self.arrivals.append(t)
        while t - self.arrivals[0] > self.window:   # keep only the last window_s
            self.arrivals.popleft()

    def progress(self, t):
        """rho = simulator seconds per wall second over the last window_s, 0..1."""
        while self.arrivals and t - self.arrivals[0] > self.window:
            self.arrivals.popleft()
        if self.t_state is None or t - self.t_state > self.hold:
            return 0.0               # stalled, or standing still (no new states)
        n = len(self.arrivals)
        if self.window <= 0.0 or n < 3:
            return 1.0
        span = self.arrivals[-1] - self.arrivals[0]
        return min(1.0, self.step * (n - 1) / span) if span > 0.0 else 1.0

    def sample(self, t):
        """The rate to report for the interval since the previous sample."""
        if self.value is None:
            return 0.0
        if self.t_sample is None:
            self.t_sample = t
            return 0.0
        dt = t - self.t_sample
        self.t_sample = t
        if dt <= 0.0:
            return 0.0
        ff = self.rate * self.progress(t)
        ref = self.value + ff * min(t - self.t_state, self.extrap)
        out = ff + (ref - self.reported - ff * dt) / max(self.tau, dt)
        self.reported += out * dt
        return out
