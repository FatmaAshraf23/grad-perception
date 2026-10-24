#!/usr/bin/env python3
"""health.py -- self-checks of the state estimator that are not tied to one sensor (2026-10-24, contract v0.3).
numpy-free, no ROS: unit-tested in test/test_health.py.

LagMonitor -- ESTIMATOR LAG (FLAG_ESTIMATOR_LAG = 2048)
    age of an estimate = the moment it is published - its stamp (the time it is valid, = the newest IMU
    sample fused). Normally a few ms: simulator, 12 clean runs, 75 123 messages: median 2.3 ms, p99 4-8 ms;
    after start-up never 2 messages in a row above 25 ms (worst single message 29 ms).
    When the CPU is overloaded the input queues fill up and EVERY estimate leaves late -- fault F07
    (12 ms of extra work per IMU message for 3 s): up to 1.0 s late while the mode said OK, so at the
    moment control received it the estimate was 3.3 m behind the car.
    ON   age > on_ms in k_on messages in a row         (F07: 0.09 s after the overload started)
    OFF  age < off_ms continuously for clear_s
    between the two thresholds the flag keeps its state (hysteresis). A negative age (the system clock
    stepped back) counts as 0 -- it is no lag.
    Start-up: the first messages after launch queue up while the nodes load (0.1-0.5 s, drains in
    < 0.3 s) -> the flag is honestly set for ~1 s while the car is still parked.
    This check does NOT replace control's own age check (now - stamp_us <= valid_for_ms): a process
    that is completely stuck publishes nothing, so only the receiver can notice that.
"""


class LagMonitor:
    def __init__(self, on_ms=25.0, k_on=2, off_ms=10.0, clear_s=1.0):
        if not off_ms < on_ms:
            raise ValueError('LagMonitor needs off_ms < on_ms (hysteresis gap)')
        self.on_ms, self.k_on, self.off_ms, self.clear_s = float(on_ms), int(k_on), float(off_ms), float(clear_s)
        self.lag = False
        self.n_over = 0                        # messages in a row above on_ms
        self.below_since = None                # time of the first message of the current run below off_ms
        self.episodes = 0                      # how often the flag went on
        self.worst_ms = 0.0                    # largest age while the flag was on (this episode)

    def update(self, age_ms, t_s):
        """age of this estimate [ms], t_s = now [s] -> True while the estimator is lagging."""
        age = max(float(age_ms), 0.0)
        self.n_over = self.n_over + 1 if age > self.on_ms else 0
        if not self.lag:
            if self.n_over >= self.k_on:
                self.lag, self.below_since, self.worst_ms = True, None, age
                self.episodes += 1
        else:
            self.worst_ms = max(self.worst_ms, age)
            if age < self.off_ms:
                if self.below_since is None:
                    self.below_since = float(t_s)
                if float(t_s) - self.below_since >= self.clear_s:
                    self.lag, self.below_since = False, None
            else:
                self.below_since = None
        return self.lag
