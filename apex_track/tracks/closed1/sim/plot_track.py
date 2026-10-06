#!/usr/bin/env python3
"""plot_track.py -- figure of the closed1 test track: the map, the apex_track centre line and the
walls from boundaries.csv. Writes closed1_track.png next to this file.

Run from anywhere:  python3 plot_track.py
Needs numpy, matplotlib, pyyaml, Pillow and apex_track (installed, or this file inside
apex_track/tracks/closed1/sim/ of the source tree).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
try:
    from apex_track import load_track
except ImportError:                     # source tree: apex_track/tracks/closed1/sim -> apex_track/
    sys.path.insert(0, str(HERE.parents[2]))
    from apex_track import load_track

INK, INK2, SURF, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#e4e3df'
C1, C2 = '#2a78d6', '#eb6834'

t = load_track(str(HERE.parent / 'track.yaml'))
m = yaml.safe_load(open(HERE / 'closed1.yaml'))
img = np.array(Image.open(HERE / m['image']))
h, w = img.shape
r = m['resolution']
ox, oy = m['origin'][:2]

fig, ax = plt.subplots(figsize=(12, 7.2), facecolor=SURF)
ax.set_facecolor(SURF)
# map: lane white, wall band dark, unknown light grey (pixel edges at origin + i * resolution)
shade = np.where(img > 250, 252, np.where(img < 50, 70, 225))
ax.imshow(shade, cmap='gray', vmin=0, vmax=255, extent=[ox, ox + w * r, oy, oy + h * r])
ax.plot(*t.p.T, '-', color=C1, lw=2, label='centre line (apex_track, smoothed)')
left, right = t.boundaries_xy()
ax.plot(*np.vstack([left, left[:1]]).T, '-', color=C2, lw=1.2, label='walls from boundaries.csv')
ax.plot(*np.vstack([right, right[:1]]).T, '-', color=C2, lw=1.2)
ax.annotate('', xy=(1.6, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=INK, lw=2))
ax.plot([0, 0], [-0.8, 0.8], color=INK, lw=3)
ax.text(0.15, -1.25, 'start/finish (0, 0)\ncar starts here, drives +x', fontsize=9, color=INK)

# corners: |kappa| >= 0.3 cores, labelled at their middle, 1.5 m to the right of the driving direction
s = np.arange(0.0, t.L, 0.05)
k = t.kappa_at(s)
core = np.abs(k) >= 0.3
starts = np.flatnonzero(core & ~np.roll(core, 1))
ends = np.flatnonzero(core & ~np.roll(core, -1))
for j, (a, b) in enumerate(zip(starts, ends)):
    if b < a:
        b += len(s)
    idx = np.arange(a, b + 1) % len(s)
    kk = k[idx][np.argmax(np.abs(k[idx]))]
    x, y, th = t.point(s[idx[len(idx) // 2]])
    txt = f'C{j + 1}  R {1 / abs(kk):.1f} m' + ('\n(right turn)' if kk < 0 else '')
    ax.text(x + 1.5 * np.sin(th), y - 1.5 * np.cos(th), txt, fontsize=9, ha='center', va='center', color=INK,
            bbox=dict(boxstyle='round,pad=0.25', fc=SURF, ec='none', alpha=0.9))
ax.text(6.2, -1.25, '14 m featureless straight', fontsize=9, color=INK2, ha='center')
ax.set_aspect('equal')
ax.set_xlabel('x [m]', color=INK2)
ax.set_ylabel('y [m]', color=INK2)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.tick_params(colors=INK2)
ax.set_title(f'closed1: simulator test track  (L = {t.L:.1f} m, 1.6 m lane, walls all around, '
             f'track_hash {t.track_hash})', loc='left', color=INK, fontsize=11)
ax.legend(loc='upper right', fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig(HERE / 'closed1_track.png', dpi=100, facecolor=SURF)
print('wrote', HERE / 'closed1_track.png')
