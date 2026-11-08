#!/usr/bin/env python3
"""camcal_pick.py -- camera calibration step 2, OFFLINE: pick the calibration pictures from a RECORDING.

WHY
  The car's camera (Raspberry Pi Camera V2) sits on the Pi, reached over SSH: no live window as camcal_capture.py has.
  So you RECORD while you move the checkerboard, copy the recording to the laptop, and this script picks the
  pictures. It uses camcal_capture.py's own rules (imported, not copied):
    1. all inner corners found (the whole board visible),
    2. the board held STILL for STILL_S (corners moved < ~1 px between looks) -- a moving board is blurred,
    3. a NEW view (position, size or tilt differs from every picked one by >= NEW_VIEW).
  Offline it can do one thing better: of all frames of one still moment it keeps the SHARPEST (Laplacian variance
  on the board), not the first one. Time = the RECORDING's timestamps (rpicam --save-pts, or the video's own), so the
  result does not depend on how fast this computer is.

INPUT (any one)
  a folder of frames (jpg / png, sorted by name)    rpicam-vid --codec mjpeg -q 95 --segment 1 -o frames/f_%05d.jpg
                                                     --save-pts frames/pts.txt     <- recommended: no video compression
  a video file (mp4 / mkv / mjpeg ...)               rpicam-vid --codec mjpeg -q 95 -o board.mjpeg --save-pts pts.txt
  --pts FILE   rpicam's timestamp file ("# timecode format v2" + one time in ms per frame); default: <folder>/pts.txt
               if it exists, else the video's own times, else --fps.
OUTPUT  <out>/img_001.png ... (lossless) + captures.csv + summary.txt -- exactly what camcal_calibrate.py reads:
        python camcal_calibrate.py <out> --fix-k3 --square <across/10 m> --square-y <down/7 m>

Usage: python camcal_pick.py <folder or video> --out captures_pi [--pts pts.txt] [--fps 30] [--cols 9 --rows 6]
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camcal_capture import (GRID, MAX_SAVED, MIN_GAP_S, NEW_VIEW, STILL_S, find_board,   # noqa: E402
                            sharpness, view_distance, view_of)

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def read_pts(path):
    """rpicam --save-pts file -> times [s] (lines starting with '#' are comments)"""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                out.append(float(line.split()[0]) / 1000.0)
    return out


def frames(source, pts_path=None, fps=30.0):
    """yield (time [s], index, name, BGR frame) in recording order"""
    if os.path.isdir(source):
        files = sorted(f for f in glob.glob(os.path.join(source, '*')) if f.lower().endswith(IMG_EXT))
        pts_path = pts_path or (os.path.join(source, 'pts.txt') if os.path.isfile(os.path.join(source, 'pts.txt'))
                                else None)
        pts = read_pts(pts_path) if pts_path else None
        if pts is not None and len(pts) < len(files):
            print(f'  warning: {len(pts)} timestamps for {len(files)} frames -- the frames after the last time use --fps')
        for k, f in enumerate(files):
            img = cv2.imread(f)
            if img is None:
                continue
            t = pts[k] if pts is not None and k < len(pts) else (pts[-1] + (k - len(pts) + 1) / fps if pts else k / fps)
            yield t, k, os.path.basename(f), img
        return
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f'cannot open {source}')
    pts = read_pts(pts_path) if pts_path else None
    k = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if pts is not None and k < len(pts):
            t = pts[k]
        else:
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = ms / 1000.0 if ms and ms > 0 else k / fps
        yield t, k, f'frame {k}', img
        k += 1
    cap.release()


def pick(source, out, pattern=(9, 6), pts_path=None, fps=30.0, max_saved=MAX_SAVED, log=print):
    """-> list of picked rows (dicts); writes <out>/img_NNN.png, captures.csv, summary.txt"""
    os.makedirs(out, exist_ok=True)
    views, rows, covered = [], [], None
    prev, prev_t, still_since = None, None, None
    best = None                                    # the sharpest frame of the current still moment
    last_save_t = -1e9
    n = n_found = 0
    w = h = None

    def save(b):
        nonlocal last_save_t, covered
        fname = f'img_{len(rows) + 1:03d}.png'
        cv2.imwrite(os.path.join(out, fname), b['img'], [cv2.IMWRITE_PNG_COMPRESSION, 1])
        c2 = b['corners'].reshape(-1, 2)
        for cx, cy in c2:
            covered[min(GRID[1] - 1, int(cy / h * GRID[1])), min(GRID[0] - 1, int(cx / w * GRID[0]))] = True
        v = b['view']
        views.append(v)
        last_save_t = b['t']
        rows.append(dict(time_s=round(b['t'] - t0, 3), file=fname, source=b['name'], x=round(v[0], 3), y=round(v[1], 3),
                         size=round(v[2], 3), tilt_lr=round(v[3], 3), tilt_tb=round(v[4], 3),
                         sharpness=round(b['sharp'], 1), still_frames=b['n_still']))
        log(f"  picked {fname} <- {b['name']} at {b['t'] - t0:6.2f} s: centre ({v[0]:.2f}, {v[1]:.2f}), size "
            f"{v[2]:.2f}, tilt {v[3]:+.2f} / {v[4]:+.2f}, sharpest of {b['n_still']} still frames, covered "
            f"{100.0 * covered.mean():.0f} %")

    def close_still():
        """the still moment ended: save its sharpest frame if it is a new view"""
        nonlocal best
        if best is not None and best['held'] >= STILL_S and len(rows) < max_saved:
            nearest = min((view_distance(best['view'], s) for s in views), default=9.0)
            if nearest >= NEW_VIEW and best['t'] - last_save_t >= MIN_GAP_S:
                save(best)
        best = None

    t0 = None
    for t, k, name, img in frames(source, pts_path, fps):
        n += 1
        if t0 is None:
            t0 = t
            h, w = img.shape[:2]
            covered = np.zeros((GRID[1], GRID[0]), bool)
            still_px = max(1.0, 0.002 * w)
            log(f'  {w} x {h} pixels')
        elif img.shape[:2] != (h, w):
            log(f'  {name}: size {img.shape[1]} x {img.shape[0]} differs from the first frame -- skipped')
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = find_board(gray, pattern)
        if corners is None:
            prev, still_since = None, None
            close_still()
            continue
        n_found += 1
        moved = None if prev is None else float(np.abs(corners - prev).reshape(-1, 2).mean())
        if prev is None or t - prev_t >= 0.08:
            prev, prev_t = corners, t
        if moved is None or moved > still_px:
            still_since = None
            close_still()
            continue
        if still_since is None:
            still_since = t
        held = t - still_since
        s = sharpness(gray, corners)
        if best is None:
            best = dict(t=t, name=name, img=img, corners=corners, view=view_of(corners, pattern, w, h), sharp=s,
                        held=held, n_still=1)
        else:
            best['held'], best['n_still'] = held, best['n_still'] + 1
            if s > best['sharp']:
                best.update(t=t, name=name, img=img, corners=corners, view=view_of(corners, pattern, w, h), sharp=s)
    close_still()
    if n == 0:
        raise SystemExit(f'no frames in {source}')

    with open(os.path.join(out, 'captures.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['time_s', 'file', 'source', 'x', 'y', 'size', 'tilt_lr', 'tilt_tb',
                                           'sharpness', 'still_frames'])
        wr.writeheader()
        wr.writerows(rows)
    dur = t - t0
    lines = [f'source {source}: {n} frames over {dur:.1f} s ({n / max(dur, 1e-9):.1f} per s), {w} x {h}; board found in '
             f'{n_found} ({100.0 * n_found / n:.0f} %)',
             f'picked {len(rows)} pictures (still >= {STILL_S} s, new view >= {NEW_VIEW}, sharpest frame of each still '
             f'moment); image covered by corners {100.0 * covered.mean():.0f} % of a {GRID[0]} x {GRID[1]} grid']
    if rows:
        V = np.array([[r['x'], r['y'], r['size'], r['tilt_lr'], r['tilt_tb']] for r in rows])
        lines.append('centre x %.2f-%.2f, y %.2f-%.2f; size %.2f-%.2f; tilt left/right up to %.2f, top/bottom up to %.2f'
                     % (V[:, 0].min(), V[:, 0].max(), V[:, 1].min(), V[:, 1].max(), V[:, 2].min(), V[:, 2].max(),
                        np.abs(V[:, 3]).max(), np.abs(V[:, 4]).max()))
        advice = []
        if len(rows) < 20:
            advice.append(f'only {len(rows)} pictures: record longer, stop at more places (aim for 25-40)')
        if covered.mean() < 0.7:
            missing = [f'{"top" if r == 0 else "bottom" if r == GRID[1] - 1 else "middle"}-'
                       f'{"left" if c == 0 else "right" if c == GRID[0] - 1 else "centre"}'
                       for r in range(GRID[1]) for c in range(GRID[0]) if not covered[r, c]]
            advice.append('the corners never reached: ' + ', '.join(sorted(set(missing))) + ' -- bring the board there '
                          '(the lens distortion is measured where the corners are)')
        if max(np.abs(V[:, 3]).max(), np.abs(V[:, 4]).max()) < 0.25:
            advice.append('little tilt: turn the board 20-40 degrees, left/right AND up/down (it pins down the focal length)')
        lines += ['ADVICE: ' + a for a in advice] or ['looks complete -- next: camcal_calibrate.py']
    else:
        lines.append('ADVICE: nothing picked -- is the whole board in the picture, and held still for ~1 s at each place?')
    with open(os.path.join(out, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    for line in lines:
        log(line)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('source', help='folder of frames or a video file')
    ap.add_argument('--out', required=True)
    ap.add_argument('--pts', default=None, help="rpicam --save-pts file (default: <folder>/pts.txt if it exists)")
    ap.add_argument('--fps', type=float, default=30.0, help='only when there are no timestamps')
    ap.add_argument('--cols', type=int, default=9, help='INNER corners across')
    ap.add_argument('--rows', type=int, default=6, help='INNER corners down')
    ap.add_argument('--max', type=int, default=MAX_SAVED, help='stop after this many pictures')
    a = ap.parse_args()
    pick(a.source, a.out, (a.cols, a.rows), a.pts, a.fps, a.max)


if __name__ == '__main__':
    main()
