#!/usr/bin/env python3
"""camcal_capture.py -- camera calibration step 2: capture checkerboard pictures for the intrinsic calibration.

WHY THIS WAY
  Sources are tried in the order given: the Windows Flask streamer (.../video, exactly what the ROS 2 camera_node
  and YOLO see) or the camera itself (cam:1 = OpenCV camera index 1, Iriun on this laptop -- its FULL picture,
  as Iriun sends it, nothing resized). A calibration made on the full picture can be converted exactly to any
  RESIZED copy of it (e.g. 2560 x 1440 -> 640 x 480): the focal lengths and the centre scale with the image, the
  distortion numbers stay the same. It can NOT be converted to a cropped picture or another lens / zoom / Iriun
  resolution mode -- then it must be redone.

WHAT YOU SEE
  A live window with the detected corners (coloured lines), a grid over the image (green = some corner of a saved
  picture was there, red = not yet), the number of saved pictures and a hint what to do next.

WHEN A PICTURE IS SAVED (automatically)
  1. all 9 x 6 inner corners are found (the whole board is visible),
  2. the board is held STILL for STILL_S (0.3 s: the corners moved less than ~1 pixel between looks) --
     a moving board is blurred, and blurred corners are measured wrong,
  3. the view is NEW: its position, size or tilt differs enough from every picture saved so far -- 30 pictures of
     the same view teach the calibration nothing more than one.
  The frame is saved exactly as received (lossless PNG); the corners are measured again later, more carefully.

KEYS (click the window first)
  q or Esc  finish          s  save this view anyway (board must be found)          u  undo the last save
  n         NEXT CAMERA (only before the first save): Windows numbers its cameras differently for each driver, so
            the first one tried may be the laptop's own webcam -- press n until the window shows the iPhone.
It stops by itself after MAX_SAVED pictures or MAX_MINUTES.

OUTPUT  <out>/img_001.png ...  (+ <out>/undone/ for undone ones), <out>/captures.csv, <out>/summary.txt

Usage:  python camcal_capture.py --source http://127.0.0.1:5000/video --source cam:1 --out captures_X
        [--cols 9 --rows 6] [--headless]   (a video file as --source plays at its own speed -- for testing)
"""
import argparse
import csv
import math
import os
import threading
import time

import cv2
import numpy as np

MAX_SAVED = 45            # pictures; ~25-40 good, varied ones are plenty
MAX_MINUTES = 8.0         # the capture stops by itself after this
WAIT_STREAM_MIN = 10.0    # waiting this long for the stream to appear
NEW_VIEW = 0.10           # how different a view must be (position/size in image fractions, tilt as log ratio)
MIN_GAP_S = 0.6           # at least this long between two saves
STILL_S = 0.3             # the board must stay still this long before a picture is taken (no motion blur)
GRID = (6, 4)             # coverage grid (columns, rows)
WIN = 'camera calibration - capture  (q = finish, s = save, u = undo, n = next camera)'


def open_source(src, wait_s=5.0):
    """A Stream that really delivers pictures, or None."""
    s = Stream(src)
    if s.ok:
        t0 = time.time()
        while s.latest()[0] is None and time.time() - t0 < wait_s:
            time.sleep(0.05)
        if s.latest()[0] is not None:
            f = s.latest()[0]
            print(f'  {src}: {f.shape[1]} x {f.shape[0]} pixels', flush=True)
            return s
        print(f'  {src}: opened, but no picture in {wait_s:.0f} s', flush=True)
    else:
        print(f'  {src}: could not be opened', flush=True)
    s.close()
    return None


class Stream:
    """Keeps only the NEWEST frame of the stream (a reader thread), so the window never lags behind."""

    def __init__(self, source):
        self.source, self.frame, self.n, self.t = source, None, 0, 0.0
        if source.startswith('cam:'):                 # cam:<index>[:dshow|msmf][:mjpg][:<W>x<H>], e.g. cam:1:dshow:mjpg
            parts = source.split(':')[1:]
            api = {'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF}.get(next((q for q in parts if q in ('dshow', 'msmf')),
                                                                         ''), cv2.CAP_ANY)
            self.cap = cv2.VideoCapture(int(parts[0]), api)
            if 'mjpg' in parts and self.cap.isOpened():   # job 100: Iriun via DirectShow gives BLACK frames at 1/s
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))   # unless MJPG is asked for
            size = next((q for q in parts if 'x' in q), None)
            if size and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(size.split('x')[0]))
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(size.split('x')[1]))
        else:
            self.cap = cv2.VideoCapture(source)
        self.is_file = os.path.isfile(source)
        self.fps_file = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.ok = self.cap.isOpened()
        self.lock, self.running = threading.Lock(), True
        if self.ok:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                if self.is_file:
                    self.running = False
                    break
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame, self.n, self.t = f, self.n + 1, time.time()
            if self.is_file:
                time.sleep(1.0 / self.fps_file)

    def latest(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy()), self.n, self.t

    def close(self):
        self.running = False
        self.cap.release()


def view_of(corners, pattern, w, h):
    """Where the board is and how it is held: centre x/y and size as image fractions, horizontal / vertical tilt
    (log of the ratio of opposite board edges: 0 = facing the camera)."""
    c = corners.reshape(-1, 2)
    cols, rows = pattern
    tl, tr, bl, br = c[0], c[cols - 1], c[-cols], c[-1]
    cross = lambda u, v: u[0] * v[1] - u[1] * v[0]          # noqa: E731 (2-D cross product)
    area = 0.5 * abs(cross(tr - tl, bl - tl)) + 0.5 * abs(cross(tr - br, bl - br))
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    return np.array([c[:, 0].mean() / w, c[:, 1].mean() / h, math.sqrt(area / (w * h)),
                     math.log(left / right), math.log(top / bottom)])


def view_distance(a, b):
    """How different two views are: position (image fractions), distance (size ratio) and tilt. Tilt counts double:
    tilted views are what pins down the focal length. ~0.10 = a new view: the board moved ~1/10 of the image,
    came ~40 % closer / farther, or turned ~15 degrees."""
    d = a - b
    dsize = 0.3 * math.log(a[2] / b[2])
    return math.sqrt(d[0] ** 2 + d[1] ** 2 + dsize ** 2 + 4.0 * (d[3] ** 2 + d[4] ** 2))


def find_board(gray, pattern):
    """Fast detection for the live view (on a smaller copy when the image is large), refined to sub-pixel."""
    h, w = gray.shape
    s = min(1.0, 960.0 / w)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(small, pattern, flags=flags)
    if not ok:
        return None
    corners = corners / s
    win = max(3, int(round(5 * w / 640)))
    return cv2.cornerSubPix(gray, corners.astype(np.float32), (win, win), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))


def sharpness(gray, corners):
    x, y, bw, bh = cv2.boundingRect(corners.astype(np.float32))
    roi = gray[max(0, y):y + bh, max(0, x):x + bw]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0


def draw(frame, corners, pattern, covered, saved, views, hint, flash, w, h, source=''):
    vis = frame.copy()
    gx, gy = GRID
    over = vis.copy()
    for i in range(gx):
        for j in range(gy):
            x0, y0, x1, y1 = int(i * w / gx), int(j * h / gy), int((i + 1) * w / gx), int((j + 1) * h / gy)
            cv2.rectangle(over, (x0, y0), (x1, y1), (60, 200, 60) if covered[j, i] else (60, 60, 220), -1)
    vis = cv2.addWeighted(over, 0.18, vis, 0.82, 0)
    for i in range(1, gx):
        cv2.line(vis, (int(i * w / gx), 0), (int(i * w / gx), h), (200, 200, 200), 1)
    for j in range(1, gy):
        cv2.line(vis, (0, int(j * h / gy)), (w, int(j * h / gy)), (200, 200, 200), 1)
    if corners is not None:
        cv2.drawChessboardCorners(vis, pattern, corners, True)
    if flash:
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 255, 0), max(6, w // 80))
    scale = max(0.5, w / 1280.0)
    lines = [f'saved {saved} / {MAX_SAVED}   covered {100.0 * covered.mean():.0f} %   ({source}, {w} x {h})', hint]
    if views:
        v = np.array(views)
        lines.append('position x %.2f-%.2f  y %.2f-%.2f   size %.2f-%.2f   tilt %.2f / %.2f' % (
            v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max(), v[:, 2].min(), v[:, 2].max(),
            np.abs(v[:, 3]).max(), np.abs(v[:, 4]).max()))
    for k, text in enumerate(lines):
        y = int((30 + 32 * k) * scale)
        cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * scale, (0, 0, 0), int(5 * scale) + 1, cv2.LINE_AA)
        cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * scale, (255, 255, 255), max(1, int(2 * scale)),
                    cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', action='append', required=True, help='stream URL (or a video file); may repeat')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cols', type=int, default=9, help='INNER corners across')
    ap.add_argument('--rows', type=int, default=6, help='INNER corners down')
    ap.add_argument('--headless', action='store_true', help='no window (tests)')
    a = ap.parse_args()
    pattern = (a.cols, a.rows)
    os.makedirs(a.out, exist_ok=True)
    show = not a.headless
    if show:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass

    # 1. wait for the stream
    stream, src_i, t_wait = None, 0, time.time()
    print(f'waiting for a camera ({", ".join(a.source)}) ...', flush=True)
    while stream is None:
        for i, src in enumerate(a.source):
            stream = open_source(src)
            if stream is not None:
                src_i = i
                break
        if stream is None:
            if time.time() - t_wait > WAIT_STREAM_MIN * 60:
                print('NO CAMERA: start Iriun (phone + PC), then run me again', flush=True)
                return
            if show:
                msg = np.full((360, 900, 3), 40, np.uint8)
                for k, text in enumerate(['Waiting for the camera ...', 'Start Iriun on the phone and on the PC.',
                                          'Trying: ' + ', '.join(a.source)]):
                    cv2.putText(msg, text, (20, 60 + 50 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(WIN, msg)
                if (cv2.waitKey(1500) & 0xFF) in (ord('q'), 27):
                    return
            else:
                time.sleep(1.5)
    frame, _, _ = stream.latest()
    h, w = frame.shape[:2]
    print(f'stream: {stream.source}, {w} x {h} pixels', flush=True)
    if show:
        dw = min(w, 1280)
        cv2.resizeWindow(WIN, dw, int(h * dw / w))

    # 2. capture
    still_px = max(1.0, 0.002 * w)
    covered = np.zeros((GRID[1], GRID[0]), bool)
    views, rows_out, saved_files = [], [], []
    prev, prev_t, last_n, last_save, flash_until, next_id = None, 0.0, -1, 0.0, 0.0, 1
    still_since = None                                # time the board stopped moving (None = moving / not seen)
    t_start, n_frames, n_found, t_last_frame = time.time(), 0, 0, time.time()
    hint = 'Is this the iPhone? If not, press n (next camera). Then show the WHOLE board.'
    pending_key = 255                                 # a key pressed while waiting for the next frame
    while True:
        frame, n, t_frame = stream.latest()
        if n == last_n:                               # no new frame yet
            if time.time() - t_last_frame > 10.0:
                print('stream lost for 10 s -- stopping', flush=True)
                break
            if show:
                k = cv2.waitKey(5) & 0xFF
                if k in (ord('q'), 27):
                    break
                if k != 255:
                    pending_key = k
            if not stream.running and stream.is_file:
                break
            time.sleep(0.005)
            continue
        last_n, t_last_frame = n, time.time()
        n_frames += 1
        if frame.shape[:2] != (h, w):
            print(f'the stream changed size to {frame.shape[1]} x {frame.shape[0]} -- stopping (a calibration is only '
                  'valid for one resolution)', flush=True)
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_board(gray, pattern)
        key = (cv2.waitKey(1) & 0xFF) if show else 255
        if key == 255:
            key, pending_key = pending_key, 255
        if key in (ord('q'), 27):
            break
        if show and n_frames > 5:
            try:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:   # closed with the X button = finish
                    break
            except cv2.error:
                break
        if key == ord('n'):
            if views:
                hint = 'pictures are saved already -- the camera can only be changed before the first save'
            else:
                stream.close()
                new = None
                for k in range(1, len(a.source) + 1):
                    j = (src_i + k) % len(a.source)
                    new = open_source(a.source[j])
                    if new is not None:
                        src_i = j
                        break
                if new is None:
                    print('no other camera delivers pictures -- stopping', flush=True)
                    break
                stream = new
                frame, last_n, _ = stream.latest()
                h, w = frame.shape[:2]
                still_px = max(1.0, 0.002 * w)
                prev, still_since, n_frames, n_found = None, None, 0, 0
                print(f'switched to {stream.source}: {w} x {h} pixels', flush=True)
                if show:
                    dw = min(w, 1280)
                    cv2.resizeWindow(WIN, dw, int(h * dw / w))
                hint = f'now {stream.source} -- is this the iPhone? If not, press n again.'
                continue
        if key == ord('u') and saved_files:
            f = saved_files.pop()
            os.makedirs(os.path.join(a.out, 'undone'), exist_ok=True)
            os.replace(os.path.join(a.out, f), os.path.join(a.out, 'undone', f))
            views.pop()
            rows_out.pop()
            covered[:] = False
            for r in rows_out:
                for (cx, cy) in r['_corners']:
                    covered[min(GRID[1] - 1, int(cy / h * GRID[1])), min(GRID[0] - 1, int(cx / w * GRID[0]))] = True
            hint = f'undone: {f}'
        save = False
        if corners is None:
            hint = 'board not found: show the WHOLE board (all corners), no glare on it'
            prev, still_since = None, None
        else:
            n_found += 1
            v = view_of(corners, pattern, w, h)
            moved = None if prev is None else float(np.abs(corners - prev).reshape(-1, 2).mean())
            if t_frame - prev_t >= 0.08 or prev is None:
                prev, prev_t = corners, t_frame
            if moved is None or moved > still_px:
                still_since = None
            elif still_since is None:
                still_since = t_frame
            held = 0.0 if still_since is None else t_frame - still_since
            nearest = min((view_distance(v, s) for s in views), default=9.0)
            if key == ord('s'):
                save = True
            elif held < STILL_S:
                hint = 'hold the board still ...'
            elif nearest < NEW_VIEW:
                hint = 'got this view -- move to a RED area, change the distance or tilt the board'
            elif time.time() - last_save >= MIN_GAP_S:
                save = True
        if save:
            fname = f'img_{next_id:03d}.png'                 # numbers are never reused, also after an undo
            next_id += 1
            cv2.imwrite(os.path.join(a.out, fname), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            c2 = corners.reshape(-1, 2)
            for (cx, cy) in c2:
                covered[min(GRID[1] - 1, int(cy / h * GRID[1])), min(GRID[0] - 1, int(cx / w * GRID[0]))] = True
            views.append(v)
            saved_files.append(fname)
            rows_out.append(dict(time_s=round(t_frame - t_start, 2), file=fname, x=round(v[0], 3), y=round(v[1], 3),
                                 size=round(v[2], 3), tilt_lr=round(v[3], 3), tilt_tb=round(v[4], 3),
                                 sharpness=round(sharpness(gray, corners), 1), _corners=c2.tolist()))
            last_save, flash_until = time.time(), time.time() + 0.35
            hint = f'saved {fname} -- next: a RED area, another distance or tilt'
            print(f'  saved {fname}: centre ({v[0]:.2f}, {v[1]:.2f}), size {v[2]:.2f}, tilt {v[3]:+.2f} / {v[4]:+.2f}, '
                  f'covered {100.0 * covered.mean():.0f} %', flush=True)
        if show:
            cv2.imshow(WIN, draw(frame, corners, pattern, covered, len(views), views, hint,
                                 time.time() < flash_until, w, h, stream.source))
        if len(views) >= MAX_SAVED:
            print(f'{MAX_SAVED} pictures saved -- enough', flush=True)
            break
        if time.time() - t_start > MAX_MINUTES * 60:
            print(f'{MAX_MINUTES:.0f} minutes are up', flush=True)
            break

    # 3. summary
    stream.close()
    if show:
        cv2.destroyAllWindows()
    dur = time.time() - t_start
    with open(os.path.join(a.out, 'captures.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['time_s', 'file', 'x', 'y', 'size', 'tilt_lr', 'tilt_tb', 'sharpness'])
        wr.writeheader()
        for r in rows_out:
            wr.writerow({k: r[k] for k in wr.fieldnames})
    lines = [f'stream {stream.source}: {w} x {h} pixels, {n_frames} frames in {dur:.0f} s ({n_frames / max(dur, 1e-6):.1f} per s), '
             f'board found in {n_found}', f'saved {len(views)} pictures, image grid covered {100.0 * covered.mean():.0f} %']
    if views:
        v = np.array(views)
        lines.append('centre x %.2f-%.2f, y %.2f-%.2f; size %.2f-%.2f; tilt left/right up to %.2f, top/bottom up to %.2f' % (
            v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max(), v[:, 2].min(), v[:, 2].max(),
            np.abs(v[:, 3]).max(), np.abs(v[:, 4]).max()))
    with open(os.path.join(a.out, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
