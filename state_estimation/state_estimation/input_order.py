"""Stamp-ordered inputs for state_pipeline (estimator-stall fix A, 2026-11-07).

WHY
    state_pipeline runs ONE single-threaded executor with two queues: /imu/data (100 Hz, depth 100)
    and /vehicle/measured (50 Hz sim / 100 Hz car, depth 50). When the process falls behind
    (busy CPU), the executor takes ONE message from EACH queue per round. At 100 Hz IMU + 50 Hz
    wheel, the wheel queue moves twice as fast in TIME, empties first, and from then on every new
    wheel message is used at once while the IMU being processed is still 0.3-1 s in the PAST.
    ekf_node.meas_cb only stores the newest speed and imu_cb predicts with it, so the EKF drove on a
    wheel speed FROM THE FUTURE: F07 (lag 12 ms for 3 s, sim healthy): speed error -3.5 .. +2.3 m/s,
    position 2.2 m off -> LOST. Test on the 245 lag rows: |est - true speed now| 1.61 m/s,
    |est - true speed one latency later| 0.14 m/s (log 2026-11-07).

WHAT
    A wheel message is handed on only once the IMU has been processed up to its stamp:
      - wheel stamp <= the newest processed IMU stamp  -> at once (the normal case: the wheel is late)
      - otherwise it WAITS; the next IMU message first releases every waiting wheel message with
        stamp <= its own stamp, in stamp order, then is processed itself.
    The IMU never waits (it drives the prediction and /state_estimate's freshness).
    REAL IMU DROPOUT vs BACKLOG: in a backlog the IMU callbacks keep running (they are queued), only
    their stamps are old; in a dropout no IMU callback runs at all. So waiting wheel messages are
    released anyway when no IMU message has been PROCESSED for wait_s of wall time (checked at every
    wheel arrival) -> the IMU-dropout prediction (ekf_node UPDATE 4) still starts, at most one wheel
    period later. max_held bounds the buffer (oldest released first).

    No ROS here (tests: test/test_input_order.py). state_pipeline parameter order_inputs (default
    OFF = the old arrival-order behaviour) and order_wait_s.

REAL CAR
    The ESP32 sends IMU + wheel in ONE 100 Hz frame: equal rates keep the queues nearly in step, so
    the error is much smaller there -- but the wheel queue (50) holds only 0.5 s at 100 Hz and any
    rate difference brings it back. Keep this ON on the car once tested; the bridge must stamp both
    messages of a frame with the same (measurement) time.
"""
import bisect


class InputOrder:

    def __init__(self, wait_s=0.025, max_held=200):
        if wait_s <= 0.0:
            raise ValueError('wait_s must be > 0')
        if max_held < 1:
            raise ValueError('max_held must be >= 1')
        self.wait_s = float(wait_s)
        self.max_held = int(max_held)
        self.held = []                 # [(stamp_s, seq, item, arrived_wall_s)], sorted by stamp
        self._seq = 0
        self.last_imu_stamp = None     # newest IMU stamp processed [s]
        self.last_imu_wall = None      # wall time of that processing [s]
        # statistics (state_pipeline logs them)
        self.n_wheel = 0
        self.n_held = 0                # wheel messages that had to wait
        self.n_timeout = 0             # released because no IMU was processed for wait_s
        self.n_overflow = 0
        self.max_waiting = 0
        self.max_hold_s = 0.0          # longest wall-time wait of one message
        self.max_lead_s = 0.0          # largest (wheel stamp - IMU stamp) seen on arrival

    def _push(self, stamp, item, now):
        self._seq += 1
        bisect.insort(self.held, (stamp, self._seq, item, now))
        self.n_held += 1
        self.max_waiting = max(self.max_waiting, len(self.held))

    def _pop_upto(self, stamp, now):
        out = []
        while self.held and (stamp is None or self.held[0][0] <= stamp):
            s, _, item, t_in = self.held.pop(0)
            self.max_hold_s = max(self.max_hold_s, now - t_in)
            out.append(item)
        return out

    def wheel(self, stamp, item, now):
        """a wheel message (stamp [s], any item, wall time now [s]) arrived; returns the items to
        process NOW, in stamp order (possibly empty: it waits for the IMU)"""
        self.n_wheel += 1
        if self.last_imu_stamp is None:           # no IMU yet (start-up): nothing to wait for
            return self._pop_upto(None, now) + [item]
        lead = stamp - self.last_imu_stamp
        self.max_lead_s = max(self.max_lead_s, lead)
        if lead <= 0.0 and not self.held:         # normal: the wheel is older than the IMU already done
            return [item]
        self._push(stamp, item, now)
        if now - self.last_imu_wall > self.wait_s:      # no IMU processed for wait_s: a dropout, not a backlog
            self.n_timeout += 1
            return self._pop_upto(None, now)
        out = self._pop_upto(self.last_imu_stamp, now)  # an older one that arrived late
        while len(self.held) > self.max_held:
            self.n_overflow += 1
            s, _, it, t_in = self.held.pop(0)
            self.max_hold_s = max(self.max_hold_s, now - t_in)
            out.append(it)
        return out

    def imu(self, stamp, now):
        """an IMU message (stamp [s]) is about to be processed at wall time now; returns the waiting wheel
        items to process BEFORE it (stamp <= the IMU stamp), in stamp order"""
        out = self._pop_upto(stamp, now)
        if self.last_imu_stamp is None or stamp > self.last_imu_stamp:
            self.last_imu_stamp = stamp
        self.last_imu_wall = now
        return out

    def report(self):
        """stats since the last report (reset), for the 5-s log line"""
        r = (f'input order: wheel {self.n_wheel}, waited {self.n_held} (most at once {self.max_waiting}, '
             f'longest {1000 * self.max_hold_s:.0f} ms, wheel ahead of IMU by up to {1000 * self.max_lead_s:.0f} ms), '
             f'released by IMU silence {self.n_timeout}, overflow {self.n_overflow}')
        self.n_wheel = self.n_held = self.n_timeout = self.n_overflow = self.max_waiting = 0
        self.max_hold_s = self.max_lead_s = 0.0
        return r
