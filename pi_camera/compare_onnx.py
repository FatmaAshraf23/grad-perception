#!/usr/bin/env python3
"""compare_onnx.py -- does the ONNX path (detectors.OnnxYolo, our own letterbox + decoding) give the SAME boxes as
ultralytics, and what does a smaller input (320 px, for the Pi's CPU) cost against the labels? (2026-11-08)

  python3 compare_onnx.py <images dir> <labels dir> --pt best.pt --onnx 320:best_320.onnx --onnx 640:best_640.onnx
Per size: ONNX vs ultralytics at the same imgsz (matched boxes: mean IoU, |confidence difference|, boxes only one has)
and every detector vs the YOLO labels (recall / precision at IoU >= 0.5); ONNX CPU time per picture (threads 4 / 2).
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detectors as dt   # noqa: E402


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(A, B, thr):
    """greedy best-IoU pairs between box lists A and B (each (u0, v0, u1, v1, conf)) -> [(i, j, iou)]"""
    pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(A) for j, b in enumerate(B)), reverse=True)
    ua, ub, out = set(), set(), []
    for o, i, j in pairs:
        if o < thr:
            break
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        out.append((i, j, o))
    return out


def labels_for(path, w, h):
    out = []
    if os.path.isfile(path):
        for line in open(path):
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, 1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images')
    ap.add_argument('labels')
    ap.add_argument('--pt', required=True)
    ap.add_argument('--onnx', action='append', required=True, help='imgsz:path')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45, help="our NMS overlap limit (ultralytics' default is 0.7)")
    ap.add_argument('--no-timing', action='store_true')
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(a.images, '*')) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    imgs = [(f, cv2.imread(f)) for f in files]
    imgs = [(f, im) for f, im in imgs if im is not None]
    print(f'{len(imgs)} pictures, sizes {sorted({im.shape[1::-1] for _, im in imgs})[:4]}')
    gt = {f: labels_for(os.path.join(a.labels, os.path.splitext(os.path.basename(f))[0] + '.txt'),
                        im.shape[1], im.shape[0]) for f, im in imgs}
    print(f'{sum(len(v) for v in gt.values())} labelled cones')

    def score(name, preds):
        tp = fp = fn = 0
        for f, _ in imgs:
            m = match([p[2:] + (p[1],) for p in preds[f]], gt[f], 0.5)
            tp += len(m)
            fp += len(preds[f]) - len(m)
            fn += len(gt[f]) - len(m)
        print(f'  {name:34s} vs labels: recall {100.0 * tp / max(1, tp + fn):5.1f} %, precision '
              f'{100.0 * tp / max(1, tp + fp):5.1f} % ({tp} found, {fn} missed, {fp} extra)')

    for spec in a.onnx:
        size, path = spec.split(':', 1)
        size = int(size)
        print(f'\n=== imgsz {size}: ultralytics (CPU) vs ONNX {os.path.basename(path)} (our NMS iou {a.iou})')
        ul = dt.UltralyticsYolo(a.pt, size, a.conf, device='cpu')
        ox = dt.OnnxYolo(path, size, a.conf, a.iou, threads=0)
        print(f"  ONNX input {ox.sess.get_inputs()[0].shape}, output {ox.sess.get_outputs()[0].shape}")
        pu = {f: ul(im) for f, im in imgs}
        po = {f: ox(im) for f, im in imgs}
        ious, dconf, only_u, only_o = [], [], 0, 0
        for f, _ in imgs:
            m = match([p[2:] for p in pu[f]], [p[2:] for p in po[f]], 0.3)
            ious += [o for _, _, o in m]
            dconf += [abs(pu[f][i][1] - po[f][j][1]) for i, j, _ in m]
            only_u += len(pu[f]) - len(m)
            only_o += len(po[f]) - len(m)
        print(f'  ONNX vs ultralytics: {len(ious)} matched boxes, IoU mean {np.mean(ious):.4f} min {np.min(ious):.4f}; '
              f'|conf diff| mean {np.mean(dconf):.4f} max {np.max(dconf):.4f}; only ultralytics {only_u}, only ONNX {only_o}')
        score(f'ultralytics {size}', pu)
        score(f'ONNX {size} (our decoding)', po)
        for th in (() if a.no_timing else (4, 2, 1)):
            o2 = dt.OnnxYolo(path, size, a.conf, a.iou, threads=th)
            for _, im in imgs[:5]:
                o2(im)                                            # warm-up
            t = []
            for _, im in imgs:
                t0 = time.perf_counter()
                o2(im)
                t.append((time.perf_counter() - t0) * 1000)
            print(f'  ONNX CPU time, {th} threads: mean {np.mean(t):.1f} ms, p95 {np.percentile(t, 95):.1f} ms per picture '
                  f'(laptop i5-12450H; the Pi is ~3-6x slower)')


if __name__ == '__main__':
    main()
