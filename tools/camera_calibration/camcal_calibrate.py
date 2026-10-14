#!/usr/bin/env python3
"""camcal_calibrate.py -- camera calibration step 3: the INTRINSIC calibration from the checkerboard pictures.

WHAT IT FINDS
  K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]  fx, fy = focal length in pixels, (cx, cy) = where the optical axis hits
                                             the image (not exactly the centre)
  D = [k1, k2, p1, p2, k3]                    lens distortion (k = radial "barrel / pincushion", p = lens not parallel)
HOW
  1. corners found again in every FULL-resolution picture, carefully (findChessboardCornersSB = sub-pixel accurate),
  2. cv2.calibrateCamera: the K, D and board poses that best explain where all 54 corners of every picture landed,
  3. pictures whose corners fit much worse than the rest (> 2.5 x median and > 0.5 px) are dropped, calibrate again,
  4. honesty checks: OpenCV's own standard deviations of fx, fy, cx, cy and a HOLD-OUT test (calibrate on 80 %,
     measure the error on the 20 % never seen; 20 random splits) -- an over-fitted calibration is fine on its own
     pictures and bad on new ones.
OUTPUT (into the captures folder)
  calibration_<W>x<H>.yaml  -- ROS camera_info format (camera_name, K, D, R, P), full resolution + scaled copies for
                               smaller pictures with the SAME 16:9 framing (focal lengths and centre scale, D stays);
                               NOT valid for 640 x 480 (4:3 = a different crop of the sensor).
  calibration_report.txt, per_image.csv, undistort_check.jpg (original | undistorted: straight edges must look straight)
Usage: python camcal_calibrate.py <captures folder> [--cols 9 --rows 6 --square 0.025]"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

SCALED = [(1920, 1080), (1280, 720), (960, 540), (640, 360)]


def find(gray, pattern):
    ok, c = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE |
                                        cv2.CALIB_CB_ACCURACY)
    if ok:
        return c.astype(np.float32), 'SB'
    ok, c = cv2.findChessboardCorners(gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None, '-'
    win = max(3, int(round(5 * gray.shape[1] / 640)))
    c = cv2.cornerSubPix(gray, c, (win, win), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001))
    return c, 'classic'


FLAGS = 0


def calib(obj, img, size):
    rms, K, D, rv, tv, std_in, _, per = cv2.calibrateCameraExtended(obj, img, size, None, None, flags=FLAGS)
    return rms, K, D, rv, tv, std_in.ravel(), per.ravel()


def view_errors(obj, img, K, D):
    """RMS pixel error of each picture with K, D FIXED (its own board pose solved) -- used for the hold-out test."""
    out = []
    for o, i in zip(obj, img):
        ok, rv, tv = cv2.solvePnP(o, i, K, D)
        p, _ = cv2.projectPoints(o, rv, tv, K, D)
        out.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, 1)))))
    return np.array(out)


def yaml_text(name, w, h, K, D):
    P = np.hstack([K, np.zeros((3, 1))])
    f = lambda a: ', '.join(f'{x:.6f}' for x in np.asarray(a).ravel())       # noqa: E731
    return (f'image_width: {w}\nimage_height: {h}\ncamera_name: {name}\n'
            f'camera_matrix:\n  rows: 3\n  cols: 3\n  data: [{f(K)}]\n'
            f'distortion_model: plumb_bob\ndistortion_coefficients:\n  rows: 1\n  cols: 5\n  data: [{f(D)}]\n'
            f'rectification_matrix:\n  rows: 3\n  cols: 3\n  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n'
            f'projection_matrix:\n  rows: 3\n  cols: 4\n  data: [{f(P)}]\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--cols', type=int, default=9)
    ap.add_argument('--rows', type=int, default=6)
    ap.add_argument('--square', type=float, default=0.025, help='square size [m] -- MEASURE the print with a ruler')
    ap.add_argument('--square-y', type=float, default=None,
                    help='square size DOWN the board [m] (the --cols direction uses --square); measure 7 squares / 7')
    ap.add_argument('--fix-k3', action='store_true', help='k3 = 0 (normal lenses; job 104: k3 4.3 +- 4.9 = undetermined)')
    ap.add_argument('--tag', default='', help='suffix for the output files')
    a = ap.parse_args()
    global FLAGS
    FLAGS = cv2.CALIB_FIX_K3 if a.fix_k3 else 0
    tg = a.tag
    pattern = (a.cols, a.rows)
    objp = np.zeros((a.cols * a.rows, 3), np.float32)
    sq_y = a.square_y if a.square_y else a.square      # printers can stretch one direction (job 106: 176 vs 175 mm)
    objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * np.array([a.square, sq_y], np.float32)
    files = sorted(glob.glob(os.path.join(a.folder, 'img_*.png')))
    rep = []
    say = lambda s='': (print(s, flush=True), rep.append(s))                   # noqa: E731
    say(f'flags: {"k3 fixed to 0" if a.fix_k3 else "full 5-number model"}'); say(f'{len(files)} pictures in {a.folder}, board {a.cols} x {a.rows} inner corners, square {a.square * 1000:.2f} x {(a.square_y or a.square) * 1000:.2f} mm (across x down)')
    names, obj, img, size, how = [], [], [], None, {}
    for fn in files:
        im = cv2.imread(fn)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = (g.shape[1], g.shape[0])
        if (g.shape[1], g.shape[0]) != size:
            say(f'  {os.path.basename(fn)}: other size, skipped')
            continue
        c, m = find(g, pattern)
        how[m] = how.get(m, 0) + 1
        if c is None:
            say(f'  {os.path.basename(fn)}: board not found at full resolution, skipped')
            continue
        names.append(os.path.basename(fn))
        obj.append(objp)
        img.append(c)
    say(f'corners found in {len(names)} of {len(files)} ({how})')
    if len(names) < 8:
        say('TOO FEW pictures (need >= 8, better 20-40) -- capture again')
        open(os.path.join(a.folder, f'calibration_report{tg}.txt'), 'w').write('\n'.join(rep) + '\n')
        return
    w, h = size

    rms, K, D, rv, tv, std, per = calib(obj, img, size)
    say(f'\nfirst pass: RMS {rms:.3f} px over {len(names)} pictures')
    med = float(np.median(per))
    bad = [i for i, e in enumerate(per) if e > max(2.5 * med, 0.5)]
    for i in bad:
        say(f'  dropped {names[i]}: error {per[i]:.2f} px (median {med:.2f})')
    keep = [i for i in range(len(names)) if i not in bad]
    first_per = dict(zip(names, per))
    names, obj, img = [names[i] for i in keep], [obj[i] for i in keep], [img[i] for i in keep]
    if bad:
        rms, K, D, rv, tv, std, per = calib(obj, img, size)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    say(f'\nFINAL ({len(names)} pictures, {w} x {h}): RMS reprojection error {rms:.3f} px '
        f'(per picture median {np.median(per):.3f}, max {per.max():.3f})')
    say(f'  fx {fx:9.2f} +- {std[0]:.2f} px    fy {fy:9.2f} +- {std[1]:.2f} px   (fx/fy {fx / fy:.4f})')
    say(f'  cx {cx:9.2f} +- {std[2]:.2f} px    cy {cy:9.2f} +- {std[3]:.2f} px   (image centre {w / 2:.0f}, {h / 2:.0f})')
    say(f'  D  k1 {D[0, 0]:+.4f}  k2 {D[0, 1]:+.4f}  p1 {D[0, 2]:+.5f}  p2 {D[0, 3]:+.5f}  k3 {D[0, 4]:+.4f}')
    say(f'     (+- {std[4]:.4f} {std[5]:.4f} {std[6]:.5f} {std[7]:.5f} {std[8]:.4f})')
    say(f'  field of view: horizontal {np.degrees(2 * np.arctan(w / 2 / fx)):.1f} deg, '
        f'vertical {np.degrees(2 * np.arctan(h / 2 / fy)):.1f} deg')

    # hold-out test
    rng = np.random.default_rng(0)
    n = len(names)
    k_test = max(2, n // 5)
    ins, outs, fxs = [], [], []
    for _ in range(20):
        idx = rng.permutation(n)
        te, tr = idx[:k_test], idx[k_test:]
        r2, K2, D2, *_ = calib([obj[i] for i in tr], [img[i] for i in tr], size)
        ins.append(r2)
        outs.append(float(np.sqrt(np.mean(view_errors([obj[i] for i in te], [img[i] for i in te], K2, D2) ** 2))))
        fxs.append(K2[0, 0])
    say(f'\nHOLD-OUT (20 splits, {n - k_test} train / {k_test} test): train RMS {np.mean(ins):.3f} px, '
        f'TEST RMS {np.mean(outs):.3f} px (max {np.max(outs):.3f}); fx spread {np.std(fxs):.2f} px')
    say('  good: test error close to the train error and both < ~0.5 px (at this resolution < ~1 px is usable)')

    # coverage of the image by corners
    allc = np.vstack([c.reshape(-1, 2) for c in img])
    gx, gy = 8, 5
    cov = np.zeros((gy, gx), bool)
    for x, y in allc:
        cov[min(gy - 1, int(y / h * gy)), min(gx - 1, int(x / w * gx))] = True
    say(f'image covered by corners: {100 * cov.mean():.0f} % of an {gx} x {gy} grid (edges matter for distortion)')

    # outputs
    with open(os.path.join(a.folder, f'per_image{tg}.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['file', 'error_px', 'first_pass_px', 'distance_m', 'tilt_deg'])
        for i, nm in enumerate(names):
            R, _ = cv2.Rodrigues(rv[i])
            wr.writerow([nm, round(float(per[i]), 3), round(float(first_per[nm]), 3), round(float(np.linalg.norm(tv[i])), 3),
                         round(float(np.degrees(np.arccos(abs(R[2, 2])))), 1)])
    tilt = [np.degrees(np.arccos(abs(cv2.Rodrigues(r)[0][2, 2]))) for r in rv]
    dist = [float(np.linalg.norm(t)) for t in tv]
    say(f'board distance {min(dist):.2f}-{max(dist):.2f} m, tilt {min(tilt):.0f}-{max(tilt):.0f} deg')
    sizes = [(w, h)] + [s for s in SCALED if s[0] < w and abs(s[0] / s[1] - w / h) < 1e-3]
    for sw, sh in sizes:
        s = sw / w
        Ks = K.copy()
        Ks[0, :] *= s
        Ks[1, :] *= sh / h
        Ks[0, 2] = (K[0, 2] + 0.5) * s - 0.5            # pixel-centre convention when resizing
        Ks[1, 2] = (K[1, 2] + 0.5) * (sh / h) - 0.5
        open(os.path.join(a.folder, f'calibration_{sw}x{sh}{tg}.yaml'), 'w').write(yaml_text('iphone_iriun', sw, sh, Ks, D))
    say(f'written: calibration_{{{", ".join(f"{x}x{y}" for x, y in sizes)}}}.yaml')

    # visual check: the picture with the board nearest an edge, original | undistorted
    far = int(np.argmax([np.abs(c.reshape(-1, 2) - [w / 2, h / 2]).max() for c in img]))
    im = cv2.imread(os.path.join(a.folder, names[far]))
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, size, 0)
    und = cv2.undistort(im, K, D, None, newK)
    both = np.hstack([cv2.resize(im, (960, int(960 * h / w))), cv2.resize(und, (960, int(960 * h / w)))])
    cv2.putText(both, f'original {names[far]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(both, 'undistorted', (970, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(a.folder, f'undistort_check{tg}.jpg'), both)
    open(os.path.join(a.folder, f'calibration_report{tg}.txt'), 'w').write('\n'.join(rep) + '\n')


if __name__ == '__main__':
    main()
