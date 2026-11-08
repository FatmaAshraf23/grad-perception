"""sources.py -- where the pictures come from, each with the time it was TAKEN (CLOCK_REALTIME ns = ROS time).

  PiCamera   the car: Raspberry Pi Camera V2 (IMX219) through picamera2, on Raspberry Pi OS (not in the container).
             Sensor mode 1640 x 1232 (2x2 binned, the WHOLE sensor = the calibrated view), the ISP shrinks it to the
             stream size (default 820 x 616 = exactly half, same view: calibration_820x616 from camcal_calibrate.py).
             Fixed exposure / gain (no auto exposure: motion blur and changing brightness hurt YOLO at speed).
             STAMP = SensorTimestamp + ExposureTime / 2 (the middle of the exposure), converted to CLOCK_REALTIME
             with the offset between the two clocks measured at start-up (SensorTimestamp is on the kernel's
             monotonic clock; which one -- MONOTONIC or BOOTTIME -- is checked on the first frame). The rolling shutter
             reads the rows over ~20 ms: what is left of the offset, latency_probe measures on the car (stamp_offset).
  Folder     a folder of pictures (+ an rpicam pts.txt, else --fps): laptop tests with the cone test pictures.
  Video      a video file (its own times): laptop tests.
Each yields Frame(img BGR, stamp_ns, info dict).
"""
import glob
import os
import time
from collections import namedtuple

Frame = namedtuple('Frame', 'img stamp_ns info')
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def clock_offsets():
    """CLOCK_REALTIME - CLOCK_MONOTONIC and - CLOCK_BOOTTIME [ns], each from the tightest of a few readings"""
    out = {}
    for name, clk in (('monotonic', time.CLOCK_MONOTONIC), ('boottime', getattr(time, 'CLOCK_BOOTTIME', None))):
        if clk is None:
            continue
        best = None
        for _ in range(5):
            a = time.clock_gettime_ns(time.CLOCK_REALTIME)
            m = time.clock_gettime_ns(clk)
            b = time.clock_gettime_ns(time.CLOCK_REALTIME)
            if best is None or b - a < best[0]:
                best = (b - a, (a + b) // 2 - m)
        out[name] = best[1]
    return out


def pick_sensor_clock(sensor_ts_ns, now_by_clock, max_age_ns=2_000_000_000):
    """which kernel clock a fresh frame's SensorTimestamp is on: the one it is just BEFORE (0 .. max_age).
    now_by_clock = {name: that clock's 'now' [ns]} -> name or None"""
    best = None
    for name, now in now_by_clock.items():
        age = now - sensor_ts_ns
        if 0 <= age <= max_age_ns and (best is None or age < best[1]):
            best = (name, age)
    return None if best is None else best[0]


class PiCamera:
    def __init__(self, size=(820, 616), sensor=(1640, 1232), fps=30.0, exposure_us=4000, gain=4.0,
                 stamp_offset_ns=0):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        cfg = self.cam.create_video_configuration(
            main={'size': tuple(size), 'format': 'RGB888'},          # RGB888 in picamera2 = B, G, R bytes (OpenCV order)
            raw={'size': tuple(sensor)}, buffer_count=4,
            controls={'FrameRate': float(fps), 'AeEnable': False, 'ExposureTime': int(exposure_us),
                      'AnalogueGain': float(gain)})
        self.cam.configure(cfg)
        self.cam.start()
        self.stamp_offset_ns = int(stamp_offset_ns)
        self.clock, self.offset_ns = None, None
        self.size = tuple(size)
        self.name = f'picamera2 {size[0]}x{size[1]} (sensor {sensor[0]}x{sensor[1]}) {fps:.0f} fps, exposure ' \
                    f'{exposure_us} us, gain {gain}'

    def frames(self):
        while True:
            req = self.cam.capture_request()
            try:
                img = req.make_array('main')
                md = req.get_metadata()
            finally:
                req.release()
            ts = int(md.get('SensorTimestamp', 0))
            if self.clock is None:                                   # first frame: which clock is it on?
                now = {'monotonic': time.clock_gettime_ns(time.CLOCK_MONOTONIC)}
                if hasattr(time, 'CLOCK_BOOTTIME'):
                    now['boottime'] = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
                self.clock = pick_sensor_clock(ts, now) or 'monotonic'
                self.offset_ns = clock_offsets()[self.clock]
            exp_ns = int(md.get('ExposureTime', 0)) * 1000
            stamp = ts + self.offset_ns + exp_ns // 2 + self.stamp_offset_ns
            yield Frame(img, stamp, {'exposure_us': md.get('ExposureTime'), 'gain': md.get('AnalogueGain'),
                                     'frame_ms': (md.get('FrameDuration') or 0) / 1000.0, 'clock': self.clock})

    def close(self):
        self.cam.stop()


class Folder:
    """pictures in name order; times from pts.txt (rpicam --save-pts) if present, else 1 / fps apart, starting NOW
    (realtime) so the stamps look like a live camera's. pace=True: deliver at that rate (like a camera)."""

    def __init__(self, path, fps=15.0, loop=1, pace=True):
        import cv2
        self.cv2 = cv2
        self.files = sorted(f for f in glob.glob(os.path.join(path, '*')) if f.lower().endswith(IMG_EXT))
        if not self.files:
            raise SystemExit(f'no pictures in {path}')
        self.fps, self.loop, self.pace = float(fps), int(loop), pace
        self.name = f'folder {path} ({len(self.files)} pictures x {loop}, {fps:.0f} fps)'

    def frames(self):
        t0 = time.clock_gettime_ns(time.CLOCK_REALTIME)
        k = 0
        for _ in range(self.loop):
            for f in self.files:
                img = self.cv2.imread(f)
                if img is None:
                    continue
                stamp = t0 + int(k * 1e9 / self.fps)
                if self.pace:
                    wait = (stamp - time.clock_gettime_ns(time.CLOCK_REALTIME)) / 1e9
                    if wait > 0:
                        time.sleep(wait)
                else:
                    stamp = time.clock_gettime_ns(time.CLOCK_REALTIME)
                yield Frame(img, stamp, {'file': os.path.basename(f)})
                k += 1

    def close(self):
        pass


class Video:
    def __init__(self, path, pace=True):
        import cv2
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f'cannot open {path}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.pace = pace
        self.name = f'video {path} ({self.fps:.0f} fps)'

    def frames(self):
        t0 = time.clock_gettime_ns(time.CLOCK_REALTIME)
        k = 0
        while True:
            ok, img = self.cap.read()
            if not ok:
                return
            stamp = t0 + int(k * 1e9 / self.fps)
            if self.pace:
                wait = (stamp - time.clock_gettime_ns(time.CLOCK_REALTIME)) / 1e9
                if wait > 0:
                    time.sleep(wait)
            yield Frame(img, stamp, {'frame': k})
            k += 1

    def close(self):
        self.cap.release()


def make_source(spec, **kw):
    """'picamera2[:WxH[:fps]]' | 'folder:<path>[:fps]' | 'video:<path>'"""
    parts = spec.split(':')
    if parts[0] == 'picamera2':
        size = tuple(int(v) for v in parts[1].split('x')) if len(parts) > 1 else (820, 616)
        cam_kw = {k: v for k, v in kw.items() if k in ('exposure_us', 'gain', 'stamp_offset_ns', 'sensor')}
        return PiCamera(size=size, fps=float(parts[2]) if len(parts) > 2 else 30.0, **cam_kw)
    if parts[0] == 'folder':
        return Folder(parts[1], float(parts[2]) if len(parts) > 2 else 15.0, loop=kw.get('loop', 1),
                      pace=kw.get('pace', True))
    if parts[0] == 'video':
        return Video(parts[1], pace=kw.get('pace', True))
    raise ValueError(f'unknown source {spec!r} (picamera2[:WxH[:fps]] | folder:<path>[:fps] | video:<path>)')
