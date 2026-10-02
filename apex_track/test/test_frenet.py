"""Test vectors for apex_track's Frenet math (known answers).
Run from the package folder:  python3 test/test_frenet.py [path/to/track.yaml]"""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # use the package next to this folder
from apex_track import FrenetTrack, load_track

ok = True
def check(name, got, want, tol):
    global ok
    good = abs(got - want) <= tol
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}: got {got:+.4f}, want {want:+.4f} (tol {tol})")

# 1) circle of radius 5 m, counter-clockwise, start at (5, 0) heading +y
R = 5.0
a = np.linspace(0, 2 * math.pi, 400, endpoint=False)
tr = FrenetTrack(np.column_stack([R * np.cos(a), R * np.sin(a)]), smooth_m=0.2, ds=0.02)
check('circle length 2*pi*R (smoothing shrinks R by R*sigma^2/2R^2)', tr.L, 2 * math.pi * R * math.exp(-0.2**2 / (2 * R**2)), 0.01)
Re = tr.L / (2 * math.pi)                     # radius of the SMOOTHED circle = the reference
s_abs, s, e_y, e_psi, k = tr.project(4.7, 0.0, math.pi / 2)        # 0.3 m INSIDE = left
check('e_y inside (left) = Re - 4.7', e_y, Re - 4.7, 0.002)
check('e_psi aligned = 0', e_psi, 0.0, 0.01)
check('kappa = +1/R (left turn)', k, 1 / R, 0.002)
tr.reset(); _, _, e_y, e_psi, _ = tr.project(5.4, 0.0, math.pi / 2 + math.radians(10))
check('e_y outside (right) = Re - 5.4', e_y, Re - 5.4, 0.002)
check('e_psi +10 deg', math.degrees(e_psi), 10.0, 0.6)
tr.reset(); s_abs, *_ = tr.project(0.0, 5.0, math.pi)                 # quarter lap
check('s at quarter lap', s_abs, tr.L / 4, 0.02)

# 2) lap counting: drive 2.5 laps forward, then 1 lap backward
tr.reset()
for ang in np.linspace(0, 5 * math.pi, 2000):
    s_abs, *_ = tr.project(R * math.cos(ang), R * math.sin(ang), ang + math.pi / 2)
check('s_abs after 2.5 laps', s_abs, 2.5 * tr.L, 0.05)
check('lap counter = 2', tr.lap, 2, 0)
for ang in np.linspace(5 * math.pi, 3 * math.pi, 800):
    s_abs, *_ = tr.project(R * math.cos(ang), R * math.sin(ang), ang + math.pi / 2)
check('s_abs after reversing 1 lap', s_abs, 1.5 * tr.L, 0.05)

# 3) straight part of the real track (bottom corridor, y ~ -0.175, heading +x)
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), 'tracks', 'levine', 'track.yaml')
lv = load_track(path)
_, s, e_y, e_psi, k = lv.project(3.0, -0.175 + 0.25, math.radians(-5))
check('levine straight: e_y = +0.25', e_y, 0.25, 0.02)
check('levine straight: e_psi = -5 deg', math.degrees(e_psi), -5.0, 0.5)
check('levine straight: kappa ~ 0', k, 0.0, 0.02)
check('levine straight: s ~ 3.03 m', s, 3.03, 0.05)
print(f'levine smoothed length {lv.L:.2f} m (raw polyline 63.59 m)')
print('ALL PASS' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
