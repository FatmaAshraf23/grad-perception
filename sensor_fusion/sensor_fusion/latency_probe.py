"""latency_probe -- how LATE are the camera's time stamps? (fusion prep step 5)

WHY: the fusion node looks up where the car was at each picture's STAMP. If the stamp is 100 ms later than the
moment the picture shows, every camera box is placed 100 ms x speed wrong (50 cm at 5 m/s). This node measures that
stamp offset, and how long the boxes (/camera/detections) take after the stamp.

HOW: it starts flash_window (a window that switches BLACK <-> WHITE at random moments and reports every switch); the
camera films it; for every switch the first picture that shows it gives delay = its stamp - the switch time; stamp
offset = median(delay) - half a frame interval (sensor_fusion latency_analysis). It includes the screen's own delay
(~10-30 ms). The node waits until the camera has seen 3 switches in a row (aim the camera meanwhile), then measures
`flips` switches, stops the window and writes the result.

ON THE CAR (the camera node running, a screen on the car's computer; point the camera at the window):
  ros2 run sensor_fusion latency_probe
DEVELOPMENT (iPhone -> Iriun -> Windows camera_server -> WSL): the window must be on the WINDOWS screen:
  ros2 run sensor_fusion latency_probe --ros-args -p flash_cmd:="<Windows python.exe> -u C:/<path>/flash_window.py"
Parameters: image_topic (/camera/image_compressed), image_compressed (true; false = sensor_msgs/Image),
detections_topic (/camera/detections), flash_cmd ('' = the flash_window.py next to this file, with this Python;
--stop-on-eof is added),
flash_args (e.g. "--pos 40,40 --size 900x650"), flips (60), search_s (180), max_delay_s (0.8), min_jump (12 grey
levels), out_dir (~/latency_tests). Publishes every switch on /latency_probe/flips (std_msgs/Header: stamp = the
switch time, frame_id white / black).
Writes <out_dir>/latency_<time>/: report.txt, latency.png, frames.npz, flips.csv, detections.csv, meta.json,
view_start.jpg / view_end.jpg (what the camera saw when the measurement started / ended: was it aimed, did it move?).
Exit code 0 = measured (>= 10 switches seen), 2 = not.  Analyse a folder again:
  ros2 run sensor_fusion latency_probe --analyze <folder>
"""
import csv
import json
import os
import shlex
import subprocess
import sys
import threading
import time

import numpy as np

from sensor_fusion import latency_analysis as la

FLASH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flash_window.py')


def _stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def pipe_summary(flips):
    """The switch reports' path (flasher -> here): clock drift and how late the reports arrived."""
    if len(flips) < 2:
        return None, (0.0, 0.0, 0.0)
    src = np.array([f[2] for f in flips])
    arr = np.array([f[3] for f in flips])
    off, drift, t0 = la.clock_map(src, arr)
    late = arr - (src + off + drift * (src - t0))
    text = (f'clock drift flasher vs ROS {drift * 1e6:+.1f} ppm; the reports arrived {np.median(late) * 1000:.1f} ms '
            f'(median) / {late.max() * 1000:.1f} ms (max) after the earliest possible -> switch times good to ~1 ms')
    return text, (off, drift, t0)


def write_results(folder, stamps, recv, cells, dup, flips, dets, use_range, meta, max_delay, min_jump, title):
    """Analyse the switches flips[use_range[0]:use_range[1]] + write report.txt and latency.png (also used by
    --analyze). -> (result, report lines)."""
    pipe, (off, drift, t0) = pipe_summary(flips)
    use = flips[use_range[0]:use_range[1]]
    sw = [(f[2] + off + drift * (f[2] - t0), f[1]) for f in use]
    res = la.analyze(stamps, recv, cells, dup, sw, dets if len(dets) else None, max_delay, min_jump)
    lines = [title, f"result: {meta.get('reason', '')}"] + la.report_lines(res, pipe)
    with open(os.path.join(folder, 'report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    if res['seen'] >= 3:
        la.plot(res, stamps, cells, os.path.join(folder, 'latency.png'), title)
    return res, lines


def analyze_folder(folder):
    z = np.load(os.path.join(folder, 'frames.npz'))
    with open(os.path.join(folder, 'flips.csv')) as f:
        flips = [(int(r['n']), int(r['level']), float(r['t_flasher']), float(r['t_arrival'])) for r in csv.DictReader(f)]
    with open(os.path.join(folder, 'detections.csv')) as f:
        dets = np.array([[float(r['stamp']), float(r['received']), float(r['latency_ms'])] for r in csv.DictReader(f)])
    with open(os.path.join(folder, 'meta.json')) as f:
        meta = json.load(f)
    res, lines = write_results(folder, z['stamps'], z['recv'], z['cells'].astype(np.float32), z['dup'], flips,
                               dets.reshape(-1, 3), (meta.get('measure_from') or 0, meta.get('measure_to')), meta,
                               meta.get('max_delay_s', 0.8),
                               meta.get('min_jump', 12.0), f'latency test {folder} (analysed again)')
    print('\n'.join(lines))
    return 0 if res['seen'] >= 10 else 2


def main(args=None):
    if '--analyze' in sys.argv:
        return analyze_folder(sys.argv[sys.argv.index('--analyze') + 1])

    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import CompressedImage, Image
    from std_msgs.msg import Header

    class LatencyProbe(Node):

        def __init__(self):
            super().__init__('latency_probe')

            from rcl_interfaces.msg import ParameterDescriptor

            def p(name, value):         # dynamic typing: numbers may be given as 60 or 60.0
                return self.declare_parameter(name, value, ParameterDescriptor(dynamic_typing=True)).value

            import cv2
            self.cv2 = cv2
            self.image_topic = p('image_topic', '/camera/image_compressed')
            self.compressed = bool(p('image_compressed', True))
            det_topic = p('detections_topic', '/camera/detections')
            flash_cmd, flash_args = p('flash_cmd', ''), p('flash_args', '')
            self.n_flips = int(p('flips', 60))
            self.search_s = float(p('search_s', 180.0))
            self.max_delay = float(p('max_delay_s', 0.8))
            self.min_jump = float(p('min_jump', 12.0))
            self.folder = os.path.join(os.path.expanduser(p('out_dir', '~/latency_tests')),
                                       time.strftime('latency_%Y%m%d_%H%M%S'))
            os.makedirs(self.folder, exist_ok=True)
            self.lock = threading.Lock()
            self.stamps, self.recv, self.cells, self.dup, self.last_key = [], [], [], [], None
            self.dets, self.flips = [], []          # flips: (n, level, flasher time, arrival time here)
            self.warned, self.last_msg = False, None
            q = qos_profile_sensor_data
            self.create_subscription(CompressedImage if self.compressed else Image, self.image_topic, self.on_image, q)
            try:
                from apex_msgs.msg import CameraDetectionArray
                self.create_subscription(CameraDetectionArray, det_topic, self.on_dets, q)
            except ImportError:
                self.get_logger().warn('apex_msgs not found: no timing of /camera/detections')
            self.flip_pub = self.create_publisher(Header, '/latency_probe/flips', 10)
            self.Header, self.Time = Header, Time
            self.cmd = ((shlex.split(flash_cmd) if flash_cmd else [sys.executable, '-u', FLASH]) + shlex.split(flash_args)
                        + ['--stop-on-eof'])
            self.get_logger().info(f'starting the blinking window: {" ".join(self.cmd)}')
            self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
            threading.Thread(target=self.read_flasher, daemon=True).start()
            self.t0 = time.time()
            self.phase, self.measure_from, self.checked, self.in_row = 'search', None, 0, 0
            self.last_status, self.result, self.lines = 0.0, None, []
            self.create_timer(0.5, self.tick)
            self.get_logger().info(f'latency test: {self.image_topic} + {det_topic}; waiting until the camera sees the '
                                   f'blinking window (max {self.search_s:.0f} s), then {self.n_flips} switches; '
                                   f'results -> {self.folder}')

        def read_flasher(self):
            for line in self.proc.stdout:
                t_arr = time.time()
                parts = line.split()
                if len(parts) == 4 and parts[0] == 'FLIP':
                    try:
                        n, level, tf = int(parts[1]), int(parts[2]), float(parts[3])
                    except ValueError:
                        continue
                    with self.lock:
                        self.flips.append((n, level, tf, t_arr))
                        off = min(a - f for _, _, f, a in self.flips)
                    h = self.Header()
                    h.stamp = self.Time(nanoseconds=int((tf + off) * 1e9)).to_msg()
                    h.frame_id = 'white' if level else 'black'
                    self.flip_pub.publish(h)
                elif line.strip():
                    self.get_logger().info(f'flash_window: {line.strip()}')

        def on_image(self, msg):
            t = time.time()
            self.last_msg = msg
            cv2 = self.cv2
            if self.compressed:
                buf = np.frombuffer(msg.data, np.uint8)
                key = hash(buf.tobytes())
                gray = cv2.imdecode(buf, cv2.IMREAD_REDUCED_GRAYSCALE_4)
            else:
                ch = {'mono8': 1, 'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4}.get(msg.encoding)
                if ch is None:
                    if not self.warned:
                        self.get_logger().error(f'encoding {msg.encoding} not supported (mono8 / bgr8 / rgb8 / ...)')
                        self.warned = True
                    return
                a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
                key = hash(a.tobytes())
                img = a[:, :msg.width * ch].reshape(msg.height, msg.width, ch)[::4, ::4]
                gray = img[:, :, 0] if ch == 1 else img[:, :, :3].mean(axis=2).astype(np.uint8)
            if gray is None:
                return
            c = la.cell_means(gray)
            with self.lock:
                self.dup.append(key == self.last_key)
                self.last_key = key
                self.stamps.append(_stamp(msg.header))
                self.recv.append(t)
                self.cells.append(c)

        def save_view(self, name):
            """The camera's latest picture -> <folder>/<name>: shows how the camera was aimed."""
            m = self.last_msg
            if m is None:
                return
            path = os.path.join(self.folder, name)
            try:
                if self.compressed:
                    with open(path, 'wb') as f:
                        f.write(bytes(m.data))
                    return
                ch = {'mono8': 1, 'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4}.get(m.encoding)
                if ch is None:
                    return
                img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * ch]
                img = img.reshape(m.height, m.width, ch)
                img = img[:, :, 0] if ch == 1 else (img[:, :, 2::-1] if m.encoding.startswith('rgb') else img[:, :, :3])
                self.cv2.imwrite(path, img)
            except (OSError, ValueError) as e:
                self.get_logger().warn(f'could not save {name}: {e}')

        def on_dets(self, msg):
            t = time.time()
            with self.lock:
                self.dets.append((_stamp(msg.header), t, float(msg.latency_ms)))

        def tick(self):
            if self.result is not None:
                return
            now = time.time()
            with self.lock:
                flips = list(self.flips)
                n_img = len(self.stamps)
                rate = sum(1 for r in self.recv[-200:] if r > now - 5.0) / 5.0
            if flips:
                off = min(a - f for _, _, f, a in flips)
                new = []
                while self.checked < len(flips) and flips[self.checked][2] + off + self.max_delay + 0.2 < now:
                    new.append(self.checked)
                    self.checked += 1
                if new:
                    with self.lock:
                        st, ce = np.array(self.stamps), np.array(self.cells)
                    for k in new:
                        seen = len(st) > 0 and la.seen_alone(st, ce, flips[k][2] + off, flips[k][1] == 1,
                                                             self.max_delay, self.min_jump) is not None
                        if self.phase == 'search':
                            self.in_row = self.in_row + 1 if seen else 0
                            if self.in_row >= 3:
                                self.phase, self.measure_from = 'measure', k - 2
                                self.save_view('view_start.jpg')
                                self.get_logger().info(f'the camera sees the blinking window (3 switches in a row) '
                                                       f'-> measuring {self.n_flips} switches (~'
                                                       f'{self.n_flips * 1.2:.0f} s); keep the camera still')
            if self.phase == 'measure' and self.checked - self.measure_from >= self.n_flips:
                return self.finish('done')
            if self.phase == 'search' and now - self.t0 > self.search_s:
                return self.finish(f'the camera did not see the blinking window within {self.search_s:.0f} s')
            if self.proc.poll() is not None and self.checked >= len(flips) and now - self.t0 > 3.0:
                return self.finish('the blinking window stopped before the end')
            if now - self.last_status >= 5.0:
                self.last_status = now
                s = f'{n_img} pictures ({rate:.1f} per s), {len(flips)} switches'
                if self.phase == 'search':
                    s += f' -- SEARCHING: point the camera at the blinking window ({self.in_row} in a row seen)'
                else:
                    s += f' -- measured {self.checked - self.measure_from} of {self.n_flips}'
                self.get_logger().info(s)

        def stop_flasher(self):
            if self.proc.poll() is None:
                try:
                    self.proc.stdin.close()
                except OSError:
                    pass
                try:
                    self.proc.wait(timeout=4.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()

        def finish(self, reason):
            if self.result is not None:
                return
            self.stop_flasher()
            self.save_view('view_end.jpg')
            with self.lock:
                stamps, recv = np.array(self.stamps, float), np.array(self.recv, float)
                cells = np.array(self.cells, np.float32).reshape(len(stamps), -1)
                dup, flips = np.array(self.dup, bool), list(self.flips)
                dets = np.array(self.dets, float).reshape(-1, 3)
            np.savez_compressed(os.path.join(self.folder, 'frames.npz'), stamps=stamps, recv=recv,
                                cells=cells.astype(np.float16), dup=dup)
            _, (off, drift, t0) = pipe_summary(flips)
            with open(os.path.join(self.folder, 'flips.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['n', 'level', 't_flasher', 't_arrival', 't_switch', 'measured'])
                for k, (n, lev, tf, ta) in enumerate(flips):
                    w.writerow([n, lev, f'{tf:.6f}', f'{ta:.6f}', f'{tf + off + drift * (tf - t0):.6f}',
                                int(self.measure_from is not None and self.measure_from <= k < self.checked)])
            with open(os.path.join(self.folder, 'detections.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['stamp', 'received', 'latency_ms'])
                w.writerows([[f'{a:.6f}', f'{b:.6f}', f'{c:.2f}'] for a, b, c in dets])
            meta = dict(reason=reason, measure_from=self.measure_from, measure_to=self.checked,
                        image_topic=self.image_topic,
                        flash_cmd=self.cmd, flips=self.n_flips, max_delay_s=self.max_delay, min_jump=self.min_jump)
            with open(os.path.join(self.folder, 'meta.json'), 'w') as f:
                json.dump(meta, f, indent=1)
            self.result, self.lines = write_results(self.folder, stamps, recv, cells, dup, flips, dets,
                                                    (self.measure_from or 0, self.checked), meta, self.max_delay,
                                                    self.min_jump, f'latency test {self.folder}')
            for line in self.lines:
                self.get_logger().info(line)

    rclpy.init(args=args)
    node = LatencyProbe()
    try:
        while rclpy.ok() and node.result is None:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        if node.result is None:
            node.finish('stopped by hand (Ctrl+C)')
    finally:
        node.stop_flasher()
        ok = node.result is not None and node.result['seen'] >= 10
        node.destroy_node()
        rclpy.try_shutdown()
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
