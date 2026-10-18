#!/usr/bin/env python3
"""A fake camera stream for testing camera_node WITHOUT a camera.

It serves http://127.0.0.1:<port>/video exactly like the Windows Flask camera_server.py
(multipart/x-mixed-replace; boundary=frame; each part = "--frame", Content-Type: image/jpeg, JPEG, CRLF).
  * Every picture carries its NUMBER as 16 black/white blocks (big enough to survive JPEG), so a
    receiver can tell which picture it got (read_index()).
  * The moment each picture is written to the socket (time.time(), the same clock ROS uses for its
    stamps on this machine) is logged to a CSV: index,send_time.
Then  stamp - send_time  = how late a node stamps a picture, and a repeated number = a duplicate.

Options to imitate problems:
  --fps 20 --jitter-ms 5      picture rate and timing jitter (the iPhone chain delivered ~20 new pictures/s)
  --content-length            add a Content-Length header to every part (Flask's server does NOT)
  --stall-after N             stop sending after N pictures but keep the connection open (a frozen stream)
  --start-index K             first picture number (use a different K after a restart)
Stop with Ctrl+C.
"""
import argparse
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

BITS = 16
BLOCK = 20          # pixels per block
GAP = 4
X0, Y0 = 12, 12


def make_picture(idx, w=640, h=480):
    """grey picture + a moving bright bar + the picture number as 16 black/white blocks (2 rows of 8)."""
    img = np.full((h, w, 3), 90, np.uint8)
    x = (idx * 7) % w
    img[:, max(0, x - 10):x + 10] = 200
    img[Y0 - 6:Y0 + 2 * (BLOCK + GAP) + 2, X0 - 6:X0 + 8 * (BLOCK + GAP) + 2] = 128   # quiet background
    for b in range(BITS):
        r, c = divmod(b, 8)
        y = Y0 + r * (BLOCK + GAP)
        xx = X0 + c * (BLOCK + GAP)
        img[y:y + BLOCK, xx:xx + BLOCK] = 255 if (idx >> b) & 1 else 0
    return img


def read_index(img):
    """the picture number from a (decoded, possibly grey) picture made by make_picture()."""
    idx = 0
    for b in range(BITS):
        r, c = divmod(b, 8)
        y = Y0 + r * (BLOCK + GAP) + BLOCK // 2
        x = X0 + c * (BLOCK + GAP) + BLOCK // 2
        if float(img[y - 4:y + 5, x - 4:x + 5].mean()) > 128:
            idx |= 1 << b
    return idx


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_handler(opts, log_file, state):
    period = 1.0 / opts.fps
    jitter = opts.jitter_ms / 1000.0
    rng = np.random.default_rng(opts.seed)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):          # keep the console quiet
            pass

        def do_GET(self):
            if self.path != '/video':
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'fake camera server is running')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            print(f'client connected: {self.client_address}', flush=True)
            next_t = time.monotonic()
            try:
                while not state['stop']:
                    with state['lock']:
                        idx = state['next']
                        state['next'] += 1
                        sent = state['sent']
                    if opts.stall_after and sent >= opts.stall_after:
                        if not state['stalled']:
                            state['stalled'] = True
                            print(f'STALLED after {sent} pictures (connection kept open)', flush=True)
                        time.sleep(0.1)
                        continue
                    ok, buf = cv2.imencode('.jpg', make_picture(idx), [cv2.IMWRITE_JPEG_QUALITY, 70])
                    data = buf.tobytes()
                    head = b'--frame\r\nContent-Type: image/jpeg\r\n'
                    if opts.content_length:
                        head += b'Content-Length: %d\r\n' % len(data)
                    head += b'\r\n'
                    next_t += period
                    j = float(np.clip(rng.normal(0.0, jitter), -0.4 * period, 0.4 * period)) if jitter > 0 else 0.0
                    wait = next_t + j - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                    t_send = time.time()
                    self.wfile.write(head + data + b'\r\n')
                    self.wfile.flush()
                    with state['lock']:
                        state['sent'] += 1
                        log_file.write(f'{idx},{t_send:.6f}\n')
                        log_file.flush()
            except (BrokenPipeError, ConnectionResetError):
                print('client went away', flush=True)

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=5055)
    ap.add_argument('--fps', type=float, default=20.0)
    ap.add_argument('--jitter-ms', type=float, default=5.0)
    ap.add_argument('--content-length', action='store_true')
    ap.add_argument('--stall-after', type=int, default=0)
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--log', required=True, help='CSV index,send_time (appended)')
    ap.add_argument('--seed', type=int, default=1)
    opts = ap.parse_args(argv)
    new = not os.path.exists(opts.log)
    log_file = open(opts.log, 'a')
    if new:
        log_file.write('index,send_time\n')
    state = dict(next=opts.start_index, sent=0, stop=False, stalled=False, lock=threading.Lock())
    srv = Server(('127.0.0.1', opts.port), make_handler(opts, log_file, state))

    def stop(*_):
        state['stop'] = True
        threading.Thread(target=srv.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f'serving http://127.0.0.1:{opts.port}/video  fps {opts.fps} jitter {opts.jitter_ms} ms  '
          f'content_length {opts.content_length}  stall_after {opts.stall_after}  start {opts.start_index}', flush=True)
    srv.serve_forever(poll_interval=0.1)
    srv.server_close()
    log_file.close()
    print(f'stopped after {state["sent"]} pictures', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
