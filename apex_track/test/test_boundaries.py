"""Boundary tests. Run from the package folder:  python3 test/test_boundaries.py

What each test proves:
  1. RING map with known walls (inner R 3.0 m, outer R 4.6 m), circular centre line
     -> w_left / w_right equal the true distances within half a pixel diagonal (3.5 cm)
  2. same ring with a 1.2 m DOOR in the outer wall -> those points are flagged
     right_virtual and closed by a straight line across the door (chord distance)
  3. Levine: track_hash UNCHANGED by adding boundaries (control's check keeps working),
     boundary_hash set, width_at = file values at the grid points, array/scalar/s_abs,
     the Frenet limit d_min keeps D = 1 - kappa*e_y >= d_min everywhere
  4. a boundaries file made for another centre line is REFUSED
  5. Pillow is only needed by the tool: loading a track never imports PIL
  6. with_boundaries=False -> no widths, width_at() raises
"""
import math
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # use the package next to this folder

from apex_track import load_track  # noqa: E402
from apex_track.boundaries import compute_boundaries  # noqa: E402

LEVINE = os.path.join(os.path.dirname(HERE), 'tracks', 'levine', 'track.yaml')
ok = True


def check(name, cond, info=''):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name} {info}")


def ring_track(tmp, door_deg=None):
    """Map: free annulus 3.0 < r < 4.6 (black elsewhere), optional door in the outer wall."""
    from PIL import Image
    res, size = 0.05, 240                          # 12 m x 12 m, origin (-6, -6)
    c = (np.arange(size) + 0.5) * res - 6.0
    X, Y = np.meshgrid(c, c[::-1])                 # image row 0 = top = largest y
    r, a = np.hypot(X, Y), np.degrees(np.arctan2(Y, X))
    free = (r > 3.0) & (r < 4.6)
    if door_deg is not None:                       # door: free from r 3.0 out to the map edge
        free |= (r > 3.0) & (np.abs(a - door_deg[0]) < door_deg[1] / 2)
    Image.fromarray(np.where(free, 254, 0).astype(np.uint8)).save(os.path.join(tmp, 'ring.png'))
    with open(os.path.join(tmp, 'ring.yaml'), 'w') as f:
        f.write('image: ring.png\nresolution: 0.05\norigin: [-6.0, -6.0, 0.0]\n'
                'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    t = np.linspace(0, 2 * math.pi, 240, endpoint=False)      # CCW circle R 3.8 -> left = inside
    np.savetxt(os.path.join(tmp, 'centerline.csv'), np.column_stack([3.8 * np.cos(t), 3.8 * np.sin(t)]),
               delimiter=',', fmt='%.4f', header='x,y', comments='')
    with open(os.path.join(tmp, 'track.yaml'), 'w') as f:
        f.write('track_id: ring\ncenterline: centerline.csv\nsmooth_m: 0.2\nds: 0.05\n')
    return load_track(os.path.join(tmp, 'track.yaml')), os.path.join(tmp, 'ring.yaml')


tmp = tempfile.mkdtemp()
try:
    # 1) ring, no door
    tr, mp = ring_track(tmp)
    b = compute_boundaries(tr, mp)
    rc = np.hypot(tr.p[:, 0], tr.p[:, 1])           # smoothed centre-line radius
    el, er = b['w_left'] - (rc - 3.0), b['w_right'] - (4.6 - rc)
    check('1 ring: w_left = r - 3.0 within 3.5 cm', np.max(np.abs(el)) <= 0.036,
          f'(max error {np.max(np.abs(el)) * 100:.1f} cm)')
    check('1 ring: w_right = 4.6 - r within 3.5 cm', np.max(np.abs(er)) <= 0.036,
          f'(max error {np.max(np.abs(er)) * 100:.1f} cm)')
    check('1 ring: no virtual points', not b['left_virtual'].any() and not b['right_virtual'].any())

    # 2) ring with a 1.2 m door (about 15 deg at r 4.6) at +45 deg
    tmp2 = tempfile.mkdtemp()
    try:
        door = (45.0, math.degrees(1.2 / 4.6))
        tr2, mp2 = ring_track(tmp2, door_deg=door)
        b2 = compute_boundaries(tr2, mp2)
        ang = np.degrees(np.arctan2(tr2.p[:, 1], tr2.p[:, 0]))
        inside_door = np.abs(ang - door[0]) < door[1] / 2 - 1.0      # clearly inside the door
        outside = np.abs(ang - door[0]) > door[1] / 2 + 1.0
        check('2 door: points in the door flagged right_virtual',
              b2['right_virtual'][inside_door].all() and not b2['right_virtual'][outside].any(),
              f'({b2["right_virtual"].sum()} virtual points)')
        # straight line between the door posts: distance from the centre = 4.6 cos(dphi)
        rc2 = np.hypot(tr2.p[:, 0], tr2.p[:, 1])
        dphi = np.radians(ang - door[0])
        chord = 4.6 * math.cos(math.radians(door[1] / 2)) / np.cos(dphi) - rc2
        err = np.abs(b2['w_right'][inside_door] - chord[inside_door])
        check('2 door: closed by a straight line across the door within 5 cm', err.max() <= 0.05,
              f'(max error {err.max() * 100:.1f} cm)')
        check('2 door: left (inner) wall unaffected', not b2['left_virtual'].any())
    finally:
        shutil.rmtree(tmp2)
finally:
    shutil.rmtree(tmp)

# 3) Levine
sys.modules.pop('PIL', None)
for m in [k for k in sys.modules if k.startswith('PIL.')]:
    sys.modules.pop(m)
lv = load_track(LEVINE)
print(f'levine: track_hash = {lv.track_hash}, boundary_hash = {lv.boundary_hash}')
check('3 track_hash unchanged by the boundaries', lv.track_hash == 'a44e46d238cf50c2')
check('3 boundary_hash set', len(lv.boundary_hash) == 16)
wl, wr = lv.width_at(lv.s0, d_min=None)
check('3 width_at(grid, raw) = file values', np.allclose(wl, lv.w_left) and np.allclose(wr, lv.w_right))
s_test = 23.456
a1 = lv.width_at(s_test)
a2 = lv.width_at(np.array([s_test]))
a3 = lv.width_at(s_test + 2 * lv.L)
check('3 scalar / array / s_abs give the same widths',
      isinstance(a1[0], float) and abs(a1[0] - a2[0][0]) < 1e-12 and abs(a1[0] - a3[0]) < 1e-9)
k = lv.kappa_at(lv.s0)
wl_c, wr_c = lv.width_at(lv.s0)                    # default d_min 0.1
D = np.minimum(1 - k * wl_c, 1 + k * wr_c)
check('3 default width_at keeps D >= 0.1 everywhere', D.min() >= 0.1 - 1e-9, f'(min D {D.min():.3f})')
cut = (wl_c < wl - 1e-9) | (wr_c < wr - 1e-9)
D_raw = np.minimum(1 - k * wl, 1 + k * wr)
check('3 the limit cuts ONLY where the raw wall has D < 0.1', np.array_equal(cut, D_raw < 0.1 - 1e-9),
      f'({cut.sum()} of {lv.n} points cut, raw min D {D_raw.min():.3f})')
left, right = lv.boundaries_xy()
check('3 boundaries_xy shapes', left.shape == (lv.n, 2) and right.shape == (lv.n, 2))

# 5) runtime does not need Pillow
check('5 load_track did not import PIL', 'PIL' not in sys.modules)

# 4) boundaries made for another centre line are refused
tmp = tempfile.mkdtemp()
try:
    src = os.path.dirname(LEVINE)
    for fn in ('track.yaml', 'centerline.csv', 'boundaries.csv'):
        shutil.copy(os.path.join(src, fn), tmp)
    xy = np.loadtxt(os.path.join(tmp, 'centerline.csv'), delimiter=',', skiprows=1)
    xy[100, 1] += 0.01
    np.savetxt(os.path.join(tmp, 'centerline.csv'), xy, delimiter=',', fmt='%.4f', header='x,y', comments='')
    try:
        load_track(os.path.join(tmp, 'track.yaml'))
        refused = False
    except ValueError as e:
        refused = 'remake it' in str(e)
    check('4 old boundaries + moved centre line -> refused', refused)
finally:
    shutil.rmtree(tmp)

# 6) without boundaries
nb = load_track(LEVINE, with_boundaries=False)
try:
    nb.width_at(1.0)
    raised = False
except RuntimeError:
    raised = True
check('6 with_boundaries=False -> width_at raises', not nb.has_boundaries and raised
      and nb.track_hash == lv.track_hash)

print('ALL PASS' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
