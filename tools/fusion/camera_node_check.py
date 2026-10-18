#!/usr/bin/env python3
"""Score camera_node against fake_mjpeg_server.py (no camera needed).

Subscribes to /camera/image_compressed, reads the picture NUMBER drawn into every fake picture and
compares the message's stamp with the time the server SENT that picture (the server's CSV log).
Reports:
  * messages per second and how many of them were DUPLICATES (the same picture published again),
  * pictures the node SKIPPED,
  * stamp - send time for the first copy of each picture (median, p5, p95, std): how late the stamp is,
  * how much LATER the duplicates were stamped (the old node re-stamped them),
  * receive - stamp (delivery time to a subscriber), stamps going backwards, outages > 0.5 s.

  python3 camera_node_check.py --send-log send.csv --seconds 20 --label new --out results/new
  --require-clean : exit 2 if there are duplicates, backwards stamps or unreadable pictures.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fake_mjpeg_server import read_index  # noqa: E402


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float('nan')


def analyse(rows, send, warmup):
    """rows: (rx, stamp, idx, hash) in arrival order; send: idx -> send time."""
    if not rows:
        return dict(n=0)
    t0 = rows[0][0]
    keep = [r for r in rows if r[0] - t0 >= warmup]
    if len(keep) < 2:
        return dict(n=len(keep))
    rx = np.array([r[0] for r in keep])
    st = np.array([r[1] for r in keep])
    idx = [r[2] for r in keep]
    n = len(keep)
    dur = float(rx[-1] - rx[0])
    first, dup_later, unknown = {}, [], 0
    for r in keep:
        if r[2] not in send:
            unknown += 1
            continue
        if r[2] in first:
            dup_later.append(r[1] - first[r[2]][1])
        else:
            first[r[2]] = r
    delays = np.array([r[1] - send[i] for i, r in first.items()])
    # skipped pictures: holes in the numbering (a jump > 1000 = server restart, not a hole)
    u = np.array(sorted(first))
    steps = np.diff(u) if len(u) > 1 else np.array([])
    skipped = int(sum(s - 1 for s in steps if 1 < s < 1000))
    restarts = int(sum(1 for s in steps if s >= 1000))
    gaps = np.diff(rx)
    outages = [(float(rx[i] - t0), float(gaps[i])) for i in np.where(gaps > 0.5)[0]]
    return dict(
        n=n, seconds=dur, rate=(n - 1) / dur if dur > 0 else float('nan'),
        unique=len(first), duplicates=len(dup_later), dup_pct=100.0 * len(dup_later) / n,
        dup_later_ms_median=1e3 * float(np.median(dup_later)) if dup_later else None,
        unknown=unknown, skipped=skipped, restarts=restarts,
        delay_ms=dict(median=1e3 * float(np.median(delays)), p5=1e3 * pct(delays, 5), p95=1e3 * pct(delays, 95),
                      std=1e3 * float(np.std(delays)), max=1e3 * float(np.max(delays)), min=1e3 * float(np.min(delays)))
        if len(delays) else None,
        delivery_ms=dict(median=1e3 * float(np.median(rx - st)), p95=1e3 * pct(rx - st, 95)),
        stamps_backwards=int((np.diff(st) < 0).sum()),
        outages=outages,
        unique_rate=len(first) / dur if dur > 0 else float('nan'),
        first_idx=int(u[0]) if len(u) else None, last_idx=int(u[-1]) if len(u) else None,
        idx_sequence_head=idx[:5])


def report(label, a):
    if a.get('n', 0) < 2:
        return f'[{label}] NO DATA ({a.get("n", 0)} messages after the warm-up)'
    L = [f'[{label}] {a["seconds"]:.1f} s: {a["n"]} messages = {a["rate"]:.1f} /s; '
         f'{a["unique"]} different pictures ({a["unique_rate"]:.1f} /s), {a["duplicates"]} duplicates ({a["dup_pct"]:.1f} %), '
         f'{a["skipped"]} pictures skipped, {a["unknown"]} unreadable']
    d = a['delay_ms']
    if d:
        L.append(f'  stamp - send time (first copy of each picture): median {d["median"]:.1f} ms, '
                 f'p5 {d["p5"]:.1f}, p95 {d["p95"]:.1f}, std {d["std"]:.1f}, min {d["min"]:.1f}, max {d["max"]:.1f}')
    if a['dup_later_ms_median'] is not None:
        L.append(f'  duplicates were stamped {a["dup_later_ms_median"]:.1f} ms (median) LATER than the first copy of the same picture')
    L.append(f'  delivery (receive - stamp): median {a["delivery_ms"]["median"]:.1f} ms, p95 {a["delivery_ms"]["p95"]:.1f} ms; '
             f'stamps going backwards: {a["stamps_backwards"]}; server restarts seen: {a["restarts"]}')
    if a['outages']:
        L.append('  outages > 0.5 s: ' + ', '.join(f'at {t:.1f} s for {g:.1f} s' for t, g in a['outages']))
    return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--send-log', required=True)
    ap.add_argument('--topic', default='/camera/image_compressed')
    ap.add_argument('--seconds', type=float, default=20.0, help='data to collect after the warm-up')
    ap.add_argument('--warmup', type=float, default=3.0, help='ignore the first seconds after the first message')
    ap.add_argument('--max-wait', type=float, default=30.0, help='give up if no message arrives for this long')
    ap.add_argument('--label', default='camera_node')
    ap.add_argument('--out')
    ap.add_argument('--require-clean', action='store_true')
    args = ap.parse_args(argv)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage

    rclpy.init()
    node = Node('camera_node_check')
    rows = []

    def cb(m):
        rx = time.time()
        raw = bytes(m.data)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        i = read_index(img) if img is not None else -1
        rows.append((rx, m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, i, hashlib.md5(raw).hexdigest()[:12]))

    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50)
    node.create_subscription(CompressedImage, args.topic, cb, qos)
    t_start = time.time()
    print(f'[{args.label}] listening on {args.topic} ...', flush=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.time()
            if rows and now - rows[0][0] > args.warmup + args.seconds:
                break
            last = rows[-1][0] if rows else t_start
            if now - last > args.max_wait:
                print(f'[{args.label}] no message for {args.max_wait:.0f} s -- stopping', flush=True)
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    send = {}
    with open(args.send_log) as f:
        for row in csv.reader(f):
            if row and row[0].isdigit():
                send.setdefault(int(row[0]), float(row[1]))
    a = analyse(rows, send, args.warmup)
    text = report(args.label, a)
    print(text, flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, 'summary.json'), 'w') as f:
            json.dump(dict(label=args.label, **a), f, indent=1)
        with open(os.path.join(args.out, 'messages.csv'), 'w') as f:
            f.write('receive_time,stamp,picture,hash,send_time\n')
            for r in rows:
                f.write(f'{r[0]:.6f},{r[1]:.6f},{r[2]},{r[3]},{send.get(r[2], float("nan")):.6f}\n')
        with open(os.path.join(args.out, 'report.txt'), 'w') as f:
            f.write(text + '\n')
    if args.require_clean and (a.get('n', 0) < 2 or a['duplicates'] or a['stamps_backwards'] or a['unknown']):
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
