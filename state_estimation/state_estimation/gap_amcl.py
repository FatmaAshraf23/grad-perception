#!/usr/bin/env python3
"""gap_amcl.py -- AMCL fixes after an IMU dropout are NOT independent of the EKF: weight them down (2026-11-02).
No ROS, no numpy: unit-tested in test/test_gap_amcl.py.

Why (job 230, cornerunc_update/analysis/amcl_gap_diag.py): AMCL moves its particles with ekf_local's odometry. In
an IMU dropout ekf_local predicts its heading from the steering and drifts -- the SAME drift as ekf_global. So for a
while AMCL's fixes copy the EKF's own error (corner dropouts: correlation of the two errors ~0.9, clean driving
~0.3). The EKF fuses ~7 of them per second as if each were new information -> std_xy stays 3-6 cm while the error
reaches 10-30 cm (NEES up to 14, honest ~2), worst 0-2 s AFTER the dropout (G1, braking corner entry). Then, once
AMCL has recovered (2-5 cm), the over-small P gives its good fixes too little weight -> slow recovery.

What (GapFixWeight): a CONTAMINATED WINDOW per dropout, by the fix's own stamp (= its scan time):
    from the dropout start (the last trusted gyro sample)  to  hold_s after the gyro is trusted again.
  classify(t_fix) ->  'normal'                outside every window
                      'skip'                  in the gap part (gyro not trusted yet) and skip_in_gap
                      'weak'                  otherwise in the window: R is multiplied by `scale`
  reopen(t_fix)  -> (sigma_pos [m], sigma_yaw [rad]) ONCE, at the first fix after a window closed ((0, 0) otherwise):
                    the caller adds sigma_pos^2 to the x and y variance and sigma_yaw^2 to the heading variance, so the
                    EKF listens to AMCL's (by now clean) fixes again -- for the position AND the heading.
  Why the heading too (job 232): after a braking corner-entry dropout (G1) the HEADING error was 4-9 deg while the
  EKF's heading std was 1.0-1.6 deg; re-opening x / y alone left the wrong heading trusted, and it kept pushing the
  position off.
Defaults (scale 1, no skip, hold 0, reopen 0 / 0) = no effect at all (enabled False)."""

import math

NORMAL, WEAK, SKIP = 'normal', 'weak', 'skip'


class GapFixWeight:
    def __init__(self, scale=1.0, hold_s=0.0, reopen_m=0.0, skip_in_gap=False, keep=8, reopen_deg=0.0):
        if scale < 1.0 or hold_s < 0.0 or reopen_m < 0.0 or reopen_deg < 0.0:
            raise ValueError('GapFixWeight needs scale >= 1, hold_s >= 0, reopen_m >= 0, reopen_deg >= 0')
        self.scale, self.hold_s, self.reopen_m = float(scale), float(hold_s), float(reopen_m)
        self.reopen_yaw = math.radians(float(reopen_deg))
        self.skip_in_gap, self.keep = bool(skip_in_gap), int(keep)
        self.windows = []          # [t_start, t_valid or None]  (seconds, the stamps' clock)
        self.reopen_due = None     # t_end of the newest closed window whose reopen is still to be done
        self.counts = {NORMAL: 0, WEAK: 0, SKIP: 0, 'reopen': 0}

    @property
    def enabled(self):
        return self.scale > 1.0 or self.skip_in_gap or self.reopen_m > 0.0 or self.reopen_yaw > 0.0 or self.hold_s > 0.0

    # --- dropout events (from ImuGap: 'gap' -> start, 'valid' -> end) ----------------------------------------
    def start(self, t_start):
        """A dropout began; t_start = the last trusted gyro sample. A second 'gap' event of the same dropout
        (silent again while RECOVERING) keeps the open window."""
        if self.windows and self.windows[-1][1] is None:
            return
        self.windows.append([float(t_start), None])
        del self.windows[:-self.keep]

    def end(self, t_valid):
        """The gyro is trusted again at t_valid; the window stays open hold_s longer."""
        if self.windows and self.windows[-1][1] is None:
            self.windows[-1][1] = float(t_valid)
            self.reopen_due = float(t_valid) + self.hold_s

    def open_now(self):
        return bool(self.windows) and self.windows[-1][1] is None

    # --- per AMCL fix -----------------------------------------------------------------------------------------
    def classify(self, t_fix):
        t = float(t_fix)
        for t0, tv in reversed(self.windows):
            if t < t0:
                continue
            if tv is None or t <= tv:                       # the gap part: gyro not trusted yet
                kind = SKIP if self.skip_in_gap else WEAK
            elif t <= tv + self.hold_s:                     # the hold part
                kind = WEAK
            else:
                continue
            if kind == WEAK and self.scale <= 1.0:
                kind = NORMAL                               # nothing to weaken with (hold without scale)
            self.counts[kind] += 1
            return kind
        self.counts[NORMAL] += 1
        return NORMAL

    def factor(self, kind):
        return self.scale if kind == WEAK else 1.0

    def reopen(self, t_fix):
        """(sigma_pos [m], sigma_yaw [rad]) to add (as variances) ONCE at the first fix after the newest window
        closed, else (0, 0)."""
        if (self.reopen_due is None or (self.reopen_m <= 0.0 and self.reopen_yaw <= 0.0)
                or float(t_fix) <= self.reopen_due or self.open_now()):
            return 0.0, 0.0
        self.reopen_due = None
        self.counts['reopen'] += 1
        return self.reopen_m, self.reopen_yaw
