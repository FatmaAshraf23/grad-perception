#!/usr/bin/env python3
"""Tests for gap_amcl.py (GapFixWeight: AMCL fixes after an IMU dropout, 2026-11-02). No ROS. Run: python3 test_gap_amcl.py"""
import math
import sys

try:
    from state_estimation.gap_amcl import NORMAL, SKIP, WEAK, GapFixWeight
except ImportError:                                    # run from the staging folder
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from gap_amcl import NORMAL, SKIP, WEAK, GapFixWeight

RESULTS = []


def check(name, cond):
    RESULTS.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}")


def raises(f):
    try:
        f()
    except ValueError:
        return True
    return False


def main():
    g = GapFixWeight()
    g.start(10.0); g.end(11.2)
    check('defaults: disabled, every fix normal, no reopen',
          not g.enabled and all(g.classify(t) == NORMAL for t in (9.0, 10.5, 11.5, 13.0)) and g.reopen(13.0) == (0.0, 0.0))
    check('bad parameters refused', raises(lambda: GapFixWeight(scale=0.5)) and raises(lambda: GapFixWeight(hold_s=-1))
          and raises(lambda: GapFixWeight(reopen_m=-0.1)) and raises(lambda: GapFixWeight(reopen_deg=-1.0)))

    g = GapFixWeight(scale=25.0, hold_s=1.0, reopen_m=0.10, skip_in_gap=True)
    check('enabled', g.enabled)
    check('no dropout yet: normal', g.classify(5.0) == NORMAL)
    g.start(10.0)
    check('window open: fix scanned during the gap -> skip', g.open_now() and g.classify(10.4) == SKIP)
    check('fix scanned BEFORE the dropout began -> normal (its odometry was fine)', g.classify(9.95) == NORMAL)
    g.start(10.0)
    check('second gap event of the same dropout keeps ONE window', len(g.windows) == 1)
    check('no reopen while the window is open', g.reopen(10.5) == (0.0, 0.0))
    g.end(11.2)
    check('scanned before the gyro was trusted -> skip (arrives late, still the gap part)', g.classify(11.1) == SKIP)
    check('hold part (gyro trusted, < hold_s) -> weak, factor 25', g.classify(11.6) == WEAK and g.factor(WEAK) == 25.0)
    check('end of hold exactly -> still weak', g.classify(12.2) == WEAK)
    check('no reopen before the window has passed', g.reopen(12.1) == (0.0, 0.0))
    check('after the hold -> normal', g.classify(12.3) == NORMAL and g.factor(NORMAL) == 1.0)
    check('first fix after the window -> reopen 0.10 m (no heading) ONCE', g.reopen(12.3) == (0.10, 0.0) and g.reopen(12.4) == (0.0, 0.0))
    g.start(20.0); g.end(20.5)
    check('a second dropout: its own window', len(g.windows) == 2 and g.classify(20.2) == SKIP
          and g.classify(20.9) == WEAK and g.classify(15.0) == NORMAL)
    check('...and its own reopen', g.reopen(21.6) == (0.10, 0.0))

    g = GapFixWeight(scale=25.0, hold_s=1.0)
    g.start(10.0)
    check('skip_in_gap false: gap part -> weak', g.classify(10.5) == WEAK)
    g.end(11.0)
    check('reopen_m 0: never a reopen', g.reopen(13.0) == (0.0, 0.0))

    g = GapFixWeight(skip_in_gap=True)
    g.start(10.0); g.end(11.0)
    check('skip only (scale 1, hold 0): gap skipped, after the gyro is back normal',
          g.classify(10.5) == SKIP and g.classify(11.05) == NORMAL)

    g = GapFixWeight(scale=4.0, hold_s=0.5, keep=3)
    for k in range(6):
        g.start(10.0 * k); g.end(10.0 * k + 1)
    check('only the newest `keep` windows are kept', len(g.windows) == 3 and g.windows[0][0] == 30.0)
    g = GapFixWeight(scale=4.0, hold_s=0.5)
    g.end(5.0)
    check('end without start: ignored', g.windows == [] and g.reopen_due is None)
    check('counts', GapFixWeight().counts == {NORMAL: 0, WEAK: 0, SKIP: 0, 'reopen': 0})

    g = GapFixWeight(scale=4.0, hold_s=0.5, reopen_m=0.10, reopen_deg=3.0)
    g.start(10.0); g.end(11.0)
    sp, sy = g.reopen(11.6)
    check('heading re-open: (0.10 m, 3 deg in rad) once', sp == 0.10 and abs(sy - math.radians(3.0)) < 1e-12
          and g.reopen(11.7) == (0.0, 0.0))
    g = GapFixWeight(reopen_deg=3.0)
    check('heading re-open alone enables it', g.enabled)
    g.start(10.0); g.end(11.0)
    check('heading re-open alone: (0, 3 deg) after the window', g.reopen(11.1) == (0.0, math.radians(3.0)))

    print(f'\n{sum(RESULTS)} of {len(RESULTS)} passed')
    return 0 if all(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
