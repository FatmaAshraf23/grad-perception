#!/usr/bin/env python3
"""Tests for the Pi camera path without a Pi (2026-11-08): detection_packet, detectors (decoding, letterbox, NMS),
sources (clock choice, folder timing) and cam_detect end to end over a real local UDP socket. No ROS, no camera.
Run: python3 test_pi_camera.py
"""
import os
import socket
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'camera_publisher'))
from camera_publisher import detection_packet as pkt   # noqa: E402
import detectors as dt                                # noqa: E402
import sources as sr                                  # noqa: E402

RESULTS = []


def check(name, cond, info=''):
    RESULTS.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f'   [{info}]' if info else ''))


def raises(f, exc=ValueError):
    try:
        f()
    except exc:
        return True
    return False


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    # ---- packet
    boxes = [('cone', 0.91, 10.0, 20.0, 30.5, 60.25), ('cone', 0.33, 100, 110, 120, 150)]
    b = pkt.encode(7, 1_794_000_000_123_456_789, 820, 616, boxes, 12.345, 'picamera2+onnx')
    d = pkt.decode(b)
    check('packet round trip: seq, nanosecond stamp, size, boxes', d['seq'] == 7 and d['stamp_ns'] == 1_794_000_000_123_456_789
          and (d['w'], d['h']) == (820, 616) and d['boxes'][0] == ('cone', 0.91, 10.0, 20.0, 30.5, 60.25)
          and abs(d['det_ms'] - 12.35) < 1e-9, f'{len(b)} bytes')
    check('bad packets refused (garbage, wrong version, missing field, 5-value box, size 0)',
          raises(lambda: pkt.decode(b'xx')) and raises(lambda: pkt.decode(b'{"v":2}'))
          and raises(lambda: pkt.decode(b'{"v":1,"seq":1}'))
          and raises(lambda: pkt.decode(b'{"v":1,"seq":1,"stamp_ns":1,"w":1,"h":1,"boxes":[[1,2,3,4,5]]}'))
          and raises(lambda: pkt.decode(b'{"v":1,"seq":1,"stamp_ns":1,"w":0,"h":1,"boxes":[]}')))
    check('30 boxes fit easily in one datagram', len(pkt.encode(1, 1, 820, 616, boxes * 15)) < 3000)
    n0 = 10_000_000_000
    check('check_packet: fresh ok; gap of 2 counted; too old dropped; repeated seq dropped; future flagged',
          pkt.check_packet({'seq': 5, 'stamp_ns': n0 - 50_000_000}, n0, 4, 500_000_000) == (True, '', 0)
          and pkt.check_packet({'seq': 8, 'stamp_ns': n0}, n0, 5, 500_000_000) == (True, '', 2)
          and pkt.check_packet({'seq': 9, 'stamp_ns': n0 - 600_000_000}, n0, 8, 500_000_000) == (False, 'too_old', 0)
          and pkt.check_packet({'seq': 8, 'stamp_ns': n0}, n0, 8, 500_000_000)[1] == 'old_seq'
          and pkt.check_packet({'seq': 9, 'stamp_ns': n0 + 80_000_000}, n0, 8, 500_000_000) == (True, 'future', 0))
    # camera_info scaling = camcal_calibrate.py's rule (job 106: 2560x1440 -> 640x360)
    K = [2993.888, 0, 1342.190, 0, 3002.382, 706.822, 0, 0, 1]
    Ks = pkt.scale_camera_info(K, 2560, 1440, 640, 360)
    check('camera_info scaling 2560x1440 -> 640x360: fx/4, (cx + 0.5)/4 - 0.5 (as camcal_calibrate.py)',
          abs(Ks[0] - 748.472) < 1e-3 and abs(Ks[2] - ((1342.190 + 0.5) / 4 - 0.5)) < 1e-6
          and abs(Ks[4] - 750.5955) < 1e-3, f'fx {Ks[0]:.3f} cx {Ks[2]:.3f} fy {Ks[4]:.3f} cy {Ks[5]:.3f}')
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        f.write('image_width: 1640\nimage_height: 1232\ncamera_name: pi_v2\ncamera_matrix:\n  rows: 3\n  cols: 3\n'
                '  data: [1270.0, 0, 819.5, 0, 1270.0, 615.5, 0, 0, 1]\ndistortion_model: plumb_bob\n'
                'distortion_coefficients:\n  rows: 1\n  cols: 5\n  data: [0.1, -0.2, 0, 0, 0]\n')
    cal = pkt.load_camera_yaml(f.name)
    Kh = pkt.scale_camera_info(cal['K'], cal['width'], cal['height'], 820, 616)
    check('Pi V2 calibration 1640x1232 -> stream 820x616: exactly half, centre stays the centre',
          abs(Kh[0] - 635.0) < 1e-9 and abs(Kh[2] - 409.5) < 1e-9 and abs(Kh[5] - 307.5) < 1e-9)
    os.unlink(f.name)

    # ---- detectors: letterbox + decoding
    img = np.zeros((616, 820, 3), np.uint8)
    lb, s, px, py = dt.letterbox(img, 320)
    check('letterbox 820x616 -> 320: scale 0.390, padded top/bottom only', lb.shape == (320, 320, 3) and abs(s - 320 / 820) < 1e-9
          and px == 0 and py == int(round((320 - round(616 * s)) / 2 - 0.1)), f'pad {px}, {py}')
    truth = (200.0, 150.0, 260.0, 260.0)                          # a box in the ORIGINAL picture
    tx = [v * s + o for v, o in zip(truth, (px, py, px, py))]     # where it is in the letterboxed picture
    cx, cy, w, h = (tx[0] + tx[2]) / 2, (tx[1] + tx[3]) / 2, tx[2] - tx[0], tx[3] - tx[1]
    raw = np.zeros((1, 5, 2100), np.float32)                      # YOLOv8/11 raw: [1, 4 + 1 class, N]
    raw[0, :, 10] = [cx, cy, w, h, 0.9]
    raw[0, :, 11] = [cx + 1, cy, w, h, 0.8]                       # a duplicate -> NMS removes it
    raw[0, :, 12] = [50, 50, 20, 30, 0.1]                         # below conf
    raw[0, :, 13] = [280, 100, 20, 30, 0.6]                       # another cone
    out = dt.decode_yolo(raw, s, px, py, 820, 616)
    check('raw output: duplicate removed by NMS, low score dropped, 2 cones', len(out) == 2 and out[0][0] == 'cone'
          and abs(out[0][1] - 0.9) < 1e-6, f'{[(round(o[1], 2)) for o in out]}')
    check('raw output: the box maps back to the ORIGINAL pixels (IoU > 0.999)', iou(out[0][2:], truth) > 0.999,
          f'{[round(v, 2) for v in out[0][2:]]}')
    check('raw output given as [N, 4 + classes] (transposed export): same result',
          dt.decode_yolo(raw[0].T[None], s, px, py, 820, 616) == out)
    e2e = np.zeros((1, 300, 6), np.float32)                       # NMS-free end2end: [1, N, x1 y1 x2 y2 score class]
    e2e[0, 0] = [tx[0], tx[1], tx[2], tx[3], 0.85, 0]
    e2e[0, 1] = [10, 10, 20, 20, 0.05, 0]
    out2 = dt.decode_yolo(e2e, s, px, py, 820, 616)
    check('end2end output [1, 300, 6]: 1 cone, mapped back (IoU > 0.999)', len(out2) == 1 and iou(out2[0][2:], truth) > 0.999)
    edge = np.zeros((1, 5, 10), np.float32)
    edge[0, :, 0] = [5, 160, 30, 40, 0.9]                         # partly outside the picture -> clipped
    o3 = dt.decode_yolo(edge, s, px, py, 820, 616)
    check('a box past the picture edge is clipped to it', len(o3) == 1 and o3[0][2] == 0.0)
    check('a bad shape is refused', raises(lambda: dt.decode_yolo(np.zeros((2, 3, 4, 5)), 1, 0, 0, 10, 10)))
    kept = dt.nms(np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], float), np.array([0.5, 0.9, 0.7]))
    check('NMS: best first, the overlapping weaker box removed', kept == [1, 2])
    check('make_detector: none -> no boxes; unknown spec refused', dt.make_detector('none')(img) == []
          and raises(lambda: dt.make_detector('tpu:x')))

    # ---- sources: which clock is SensorTimestamp on?
    check('sensor clock: the clock the frame is just before (0..2 s) wins',
          sr.pick_sensor_clock(1000, {'monotonic': 1000 + 30_000_000, 'boottime': 1000 + 9_000_000_000}) == 'monotonic'
          and sr.pick_sensor_clock(5_000, {'monotonic': 4_000, 'boottime': 5_000 + 40_000_000}) == 'boottime'
          and sr.pick_sensor_clock(10, {'monotonic': 10 + 9_000_000_000}) is None)
    off = sr.clock_offsets()
    real = time.clock_gettime_ns(time.CLOCK_REALTIME)
    check('clock offsets: monotonic + offset = realtime (within 1 ms)',
          abs(time.clock_gettime_ns(time.CLOCK_MONOTONIC) + off['monotonic'] - real) < 1_000_000)

    # ---- folder source + cam_detect over a real UDP socket
    import cv2
    tmp = tempfile.mkdtemp()
    for k in range(12):
        cv2.imwrite(os.path.join(tmp, f'p_{k:02d}.png'), np.full((120, 160, 3), k * 10, np.uint8))
    src = sr.make_source(f'folder:{tmp}:30')
    fr = list(src.frames())
    dts = np.diff([f.stamp_ns for f in fr]) / 1e6
    check('folder source: 12 pictures, 1/30 s apart, paced like a camera', len(fr) == 12 and np.allclose(dts, 1000 / 30, atol=0.01))
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(('127.0.0.1', 0))
    rx.settimeout(2.0)
    port = rx.getsockname()[1]

    class FakeDet:
        name = 'fake'

        def __call__(self, im):
            time.sleep(0.005)
            return [('cone', 0.7, 1, 2, 30, 40)] if im[0, 0, 0] % 20 == 0 else []
    import cam_detect as cd
    t0 = time.clock_gettime_ns(time.CLOCK_REALTIME)
    tot = cd.run(sr.make_source(f'folder:{tmp}:30'), FakeDet(), '127.0.0.1', port, log=lambda *_: None)
    got = []
    try:
        while True:
            got.append(pkt.decode(rx.recvfrom(65535)[0]))
            if len(got) >= tot['pics']:
                break
    except socket.timeout:
        pass
    check('cam_detect -> UDP: every picture arrives, in order, boxes as the detector gave them',
          len(got) == tot['pics'] == 12 and [g['seq'] for g in got] == list(range(12))
          and sum(len(g['boxes']) for g in got) == 6, f"{len(got)} packets, {tot['skipped']} skipped")
    check('stamps = the picture times (not the send times): increasing, after the start',
          all(b2['stamp_ns'] > b1['stamp_ns'] for b1, b2 in zip(got, got[1:])) and got[0]['stamp_ns'] >= t0)

    class SlowDet:
        name = 'slow'

        def __call__(self, im):
            time.sleep(0.1)                                        # 10 Hz detector on a 30 Hz camera
            return []
    for k in range(12, 40):
        cv2.imwrite(os.path.join(tmp, f'p_{k:02d}.png'), np.full((120, 160, 3), 0, np.uint8))
    tot2 = cd.run(sr.make_source(f'folder:{tmp}:30'), SlowDet(), '127.0.0.1', port, log=lambda *_: None)
    check('detector slower than the camera: old frames are SKIPPED (newest wins), never queued',
          tot2['skipped'] >= 20 and tot2['pics'] + tot2['skipped'] >= 38, f"sent {tot2['pics']}, skipped {tot2['skipped']}")

    print(f'\n{sum(RESULTS)} of {len(RESULTS)} passed')
    return 0 if all(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
