#!/usr/bin/env python3
"""Calibration files read by the estimation nodes (real car and simulation). Pure Python.

steering_calibration.yaml -- written ONLY by steer_calib (or by hand):
    steering_offset_rad, steering_gain:   delta_true = gain * (delta_measured - offset)
Read by ekf_node (slip angle beta) and state_estimate (vx, vy), so the correction is
the same everywhere and lives in ONE file -- like the speed-scale anchor (wheel_calib).
"""
import math
import os

OFFSET_LIMIT_RAD = 0.15        # a larger offset is a broken servo mount, not a calibration
GAIN_LIMITS = (0.7, 1.3)


def read_numbers(path):
    """'key: value' lines -> {key: float}; comments and non-numbers are skipped."""
    vals = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.split('#', 1)[0]
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            try:
                vals[key.strip()] = float(val.strip())
            except ValueError:
                pass
    return vals


class SteeringCalibration:
    """delta_true = gain * (delta_measured - offset); the identity when there is no file."""

    def __init__(self, offset=0.0, gain=1.0, source='none'):
        self.offset, self.gain, self.source = offset, gain, source

    def correct(self, delta_measured):
        return self.gain * (delta_measured - self.offset)

    def describe(self):
        if self.source == 'none':
            return 'no steering calibration (offset 0, gain 1)'
        return (f'steering calibration from {self.source}: offset {self.offset:+.4f} rad '
                f'({math.degrees(self.offset):+.2f} deg), gain {self.gain:.3f}')


def load_steering(path):
    """(SteeringCalibration, problem): problem is None, or why the file was not used."""
    p = str(path or '').strip()
    if not p or p.lower() == 'none':
        return SteeringCalibration(), None
    p = os.path.expanduser(p)
    try:
        vals = read_numbers(p)
    except FileNotFoundError:
        return SteeringCalibration(), (f'no steering calibration file {p} -> offset 0, gain 1 '
                                       '(create it with steer_calib)')
    off = vals.get('steering_offset_rad')
    gain = vals.get('steering_gain', 1.0)
    if off is None:
        return SteeringCalibration(), f'{p} has no steering_offset_rad -> not used'
    if abs(off) > OFFSET_LIMIT_RAD or not GAIN_LIMITS[0] <= gain <= GAIN_LIMITS[1]:
        return SteeringCalibration(), (f'{p}: offset {off:+.3f} rad / gain {gain:.3f} outside the '
                                       'plausible range -> not used')
    return SteeringCalibration(off, gain, p), None
