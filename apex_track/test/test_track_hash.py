"""Track identity tests. Run from the package folder:  python3 test/test_track_hash.py

What each test proves:
  1. same file loaded twice           -> same hash (it is deterministic)
  2. same points, different text      -> same hash (CRLF / extra decimals / 0.50 vs 0.5
                                          don't matter, only the geometry does)
  3. one point moved by 1 cm          -> different hash (a real geometry change is caught)
  4. different smooth_m               -> different hash (smoothing changes s, e_y, e_psi)
  5. hash does NOT depend on window_m / max_e_y (search settings, not geometry)
"""
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # use the package next to this folder

from apex_track import compute_track_hash, load_track, load_xy_csv  # noqa: E402

LEVINE = os.path.join(os.path.dirname(HERE), 'tracks', 'levine', 'track.yaml')
ok = True


def check(name, cond, info=''):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name} {info}")


a = load_track(LEVINE)
b = load_track(LEVINE)
print(f'levine: track_id = {a.track_id}, track_hash = {a.track_hash}, L = {a.L:.2f} m')
check('1 deterministic', a.track_hash == b.track_hash)

xy = load_xy_csv(os.path.join(os.path.dirname(LEVINE), 'centerline.csv'))
tmp = tempfile.mkdtemp()
try:
    # 2) rewrite the same points with CRLF line ends and 6 decimals
    with open(os.path.join(tmp, 'centerline.csv'), 'w', newline='') as f:
        f.write('x,y\r\n' + ''.join(f'{x:.6f},{y:.6f}\r\n' for x, y in xy))
    with open(os.path.join(tmp, 'track.yaml'), 'w') as f:
        f.write('track_id: levine\ncenterline: centerline.csv\nsmooth_m: 0.50\nds: 0.050\n')
    c = load_track(os.path.join(tmp, 'track.yaml'))
    check('2 same geometry, different text -> same hash', c.track_hash == a.track_hash,
          f'({c.track_hash})')
finally:
    shutil.rmtree(tmp)

xy2 = xy.copy()
xy2[100, 1] += 0.01
check('3 one point moved 1 cm -> new hash',
      compute_track_hash(xy2, 0.5, 0.05) != a.track_hash)
check('4 smooth_m 1.0 -> new hash',
      compute_track_hash(xy, 1.0, 0.05) != a.track_hash)
d = load_track(LEVINE, window_m=5.0, max_e_y=3.0)
check('5 search settings do not change the hash', d.track_hash == a.track_hash)

print('ALL PASS' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
