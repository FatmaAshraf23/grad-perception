#!/usr/bin/env python3
"""cam_detect.py -- the car's camera + cone detector in ONE process on the Pi's HOST (Raspberry Pi OS), sending only
the boxes to ROS (decision 2026-11-08: the CSI camera needs picamera2, which is not in the ROS container).

  picture (stamped when TAKEN) -> detector (YOLO, CPU first) -> UDP 127.0.0.1:5600 -> camera_publisher detection_bridge
                                                                (in the container) -> /camera/detections + camera_info

Why one process for capture + YOLO, and only boxes over the link: a 640 x 480 picture is ~0.9 MB; sending pictures
through ROS on the Pi costs a core (perception CPU rule, 2026-10-03). The fusion only needs the boxes + the time the
picture was taken.
Newest picture wins: if the detector is slower than the camera, frames it cannot keep up with are SKIPPED (counted),
never queued -- an old picture's boxes are worth little at 5 m/s.

Usage (on the Pi host; the container's bridge listens on the same port):
  python3 cam_detect.py --source picamera2:820x616:30 --detector onnx:best_320.onnx:320:2 [--exposure-us 4000 --gain 4]
Laptop tests:
  python3 cam_detect.py --source folder:<pictures>:15 --detector ultralytics:best.pt --seconds 20
Stats every 5 s: pictures, skipped, detector ms (mean / p95), boxes, picture -> sent latency, CPU.
"""
import argparse
import os
import socket
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'camera_publisher'))     # the shared packet format (repo layout)
from camera_publisher import detection_packet as pkt   # noqa: E402
from detectors import make_detector, timed            # noqa: E402
from sources import make_source                       # noqa: E402


class Latest:
    """a reader thread keeps only the NEWEST frame (so a slow detector never works on an old picture)"""

    def __init__(self, src):
        self.src, self.frame, self.n, self.done = src, None, 0, False
        self.lock = threading.Condition()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        try:
            for f in self.src.frames():
                with self.lock:
                    self.frame, self.n = f, self.n + 1
                    self.lock.notify()
        finally:
            with self.lock:
                self.done = True
                self.lock.notify()

    def take(self, last_n, timeout=2.0):
        """-> (frame, n) newer than last_n, or (None, n) at the end / timeout"""
        with self.lock:
            end = time.monotonic() + timeout
            while self.n == last_n and not self.done:
                left = end - time.monotonic()
                if left <= 0:
                    break
                self.lock.wait(left)
            if self.n == last_n:
                return None, self.n
            return self.frame, self.n


def run(source, detector, host='127.0.0.1', port=5600, seconds=0.0, stats_every=5.0, log=print, max_pictures=0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    lt = Latest(source)
    seq = last_n = 0
    t_start = t_stats = time.monotonic()
    cpu0 = time.process_time()
    st = dict(pics=0, skipped=0, det=[], lat=[], boxes=0)
    tot = dict(pics=0, skipped=0, boxes=0)
    log(f'camera: {source.name}; detector: {detector.name}; sending to {host}:{port}')
    while True:
        f, n = lt.take(last_n)
        if f is None:
            break
        st['skipped'] += n - last_n - 1
        last_n = n
        boxes, ms = timed(detector, f.img)
        h, w = f.img.shape[:2]
        sock.sendto(pkt.encode(seq, f.stamp_ns, w, h, boxes, ms, f'{source.name.split()[0]}+{detector.name}'),
                    (host, port))
        seq += 1
        st['pics'] += 1
        st['det'].append(ms)
        st['boxes'] += len(boxes)
        st['lat'].append((time.clock_gettime_ns(time.CLOCK_REALTIME) - f.stamp_ns) / 1e6)
        now = time.monotonic()
        if now - t_stats >= stats_every:
            d, la = sorted(st['det']), sorted(st['lat'])
            cpu = (time.process_time() - cpu0) / (now - t_stats) * 100.0
            log(f"{st['pics']} pictures ({st['pics'] / (now - t_stats):.1f}/s), skipped {st['skipped']}; detector "
                f"{sum(d) / len(d):.1f} ms mean, p95 {d[int(0.95 * (len(d) - 1))]:.1f}; boxes {st['boxes']}; picture -> "
                f"sent {sum(la) / len(la):.0f} ms mean, p95 {la[int(0.95 * (len(la) - 1))]:.0f}; CPU {cpu:.0f} % of one core")
            for k in tot:
                tot[k] += st[k]
            st = dict(pics=0, skipped=0, det=[], lat=[], boxes=0)
            t_stats, cpu0 = now, time.process_time()
        if (seconds and now - t_start >= seconds) or (max_pictures and seq >= max_pictures):
            break
    for k in tot:
        tot[k] += st[k]
    source.close()
    log(f"done: {tot['pics']} pictures sent, {tot['skipped']} skipped, {tot['boxes']} boxes")
    return tot


def main():
    ap = argparse.ArgumentParser(description='camera + cone detector on the Pi host -> UDP -> ROS detection_bridge')
    ap.add_argument('--source', default='picamera2:820x616:30',
                    help='picamera2[:WxH[:fps]] | folder:<path>[:fps] | video:<path>')
    ap.add_argument('--detector', default='none', help='none | onnx:<file>[:imgsz[:threads, default 2]] | ultralytics:<file>[:imgsz]')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5600)
    ap.add_argument('--seconds', type=float, default=0.0, help='stop after (0 = never)')
    ap.add_argument('--exposure-us', type=int, default=4000, help='picamera2: fixed exposure (no auto exposure)')
    ap.add_argument('--gain', type=float, default=4.0, help='picamera2: fixed analogue gain')
    ap.add_argument('--stamp-offset-ms', type=float, default=0.0, help='picamera2: added to every stamp (latency_probe)')
    ap.add_argument('--loop', type=int, default=1, help='folder: play it this many times')
    a = ap.parse_args()
    if a.source.startswith('picamera2'):
        parts = a.source.split(':')
        from sources import PiCamera
        size = tuple(int(v) for v in parts[1].split('x')) if len(parts) > 1 else (820, 616)
        src = PiCamera(size=size, fps=float(parts[2]) if len(parts) > 2 else 30.0, exposure_us=a.exposure_us,
                       gain=a.gain, stamp_offset_ns=int(a.stamp_offset_ms * 1e6))
    else:
        src = make_source(a.source, loop=a.loop)
    try:
        run(src, make_detector(a.detector, a.conf), a.host, a.port, a.seconds)
    except KeyboardInterrupt:
        src.close()


if __name__ == '__main__':
    main()
