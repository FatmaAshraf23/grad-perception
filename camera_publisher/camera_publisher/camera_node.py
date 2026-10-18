"""camera_node -- Windows Flask MJPEG stream -> /camera/image_compressed (JPEG) + /camera/camera_info.

2026-10-18 (housekeeping before the car camera; the latency test of 2026-10-16 found three faults):
  * STAMP ON ARRIVAL.  The old node stamped a picture when its 30 Hz TIMER published it, which can be
    up to ~50 ms after the picture arrived.  Now the stamp is taken in the capture thread the moment
    cap.read() returns.  (A stamp can never be earlier than the picture's arrival here; what the stream
    itself delayed before that - Iriun, the Windows server - is measured by `latency_probe` and later
    corrected in the fusion node by `camera_delay_s`.)
  * EACH PICTURE ONCE.  The old timer re-sent the newest picture 30 times a second although the stream
    delivers ~20 new ones: 35 % duplicates, each with a later stamp - the fusion node would have trusted
    the same picture at 3 different times.  Now a picture is encoded and published exactly once, from the
    capture thread, when it arrives.  The message rate is the stream's real picture rate.
  * CLEAN SHUTDOWN.  The old destroy_node() called cap.release() while the capture thread was still inside
    cap.read() on the same VideoCapture -> segmentation fault on Ctrl+C.  Now the capture thread owns the
    VideoCapture: it opens, reads and releases it itself; destroy_node() only asks it to stop and waits.
  * RECONNECT.  If the stream cannot be opened or stops, the thread retries every `reconnect_s` seconds
    (the old node gave up for good when the first open failed).

2026-10-14 (camera calibration step 4): /camera/camera_info (from the calibration YAML, same stamp +
frame_id as every image), frame_id camera_optical_frame, parameters url / camera_info_file / frame_id, and
a size check (the calibration is only valid for the picture size it was made for).
Earlier version: camera_node.py.bak_job112 / git history.
"""
import os
import threading
import time

import cv2
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage

try:                                    # Humble: Ctrl+C may end spin() with this instead of KeyboardInterrupt
    from rclpy.executors import ExternalShutdownException
except ImportError:                     # older rclpy
    ExternalShutdownException = KeyboardInterrupt


def load_camera_info(path):
    """Read a ROS camera_info YAML (the format camera_calibration / camcal_calibrate.py write) into a CameraInfo."""
    with open(path) as f:
        c = yaml.safe_load(f)
    info = CameraInfo()
    info.width = int(c['image_width'])
    info.height = int(c['image_height'])
    info.distortion_model = c['distortion_model']
    info.k = [float(x) for x in c['camera_matrix']['data']]
    info.d = [float(x) for x in c['distortion_coefficients']['data']]
    info.r = [float(x) for x in c['rectification_matrix']['data']]
    info.p = [float(x) for x in c['projection_matrix']['data']]
    return info


class CameraPublisher(Node):

    def __init__(self):
        super().__init__('camera_publisher')

        default_info = os.path.join(get_package_share_directory('camera_publisher'), 'config',
                                    'iphone_iriun_640x480.yaml')
        self.url = self.declare_parameter('url', 'http://172.29.48.1:5000/video').value
        info_file = self.declare_parameter('camera_info_file', default_info).value
        self.frame_id = self.declare_parameter('frame_id', 'camera_optical_frame').value
        self.reconnect_s = float(self.declare_parameter('reconnect_s', 1.0).value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        self.publisher = self.create_publisher(CompressedImage, '/camera/image_compressed', qos)
        self.info_publisher = self.create_publisher(CameraInfo, '/camera/camera_info', qos)

        self.camera_info = None
        try:
            self.camera_info = load_camera_info(info_file)
            k = self.camera_info.k
            self.get_logger().info(
                f'calibration {info_file}: {self.camera_info.width} x {self.camera_info.height}, '
                f'fx {k[0]:.1f} fy {k[4]:.1f} cx {k[2]:.1f} cy {k[5]:.1f}')
        except (OSError, KeyError, ValueError, yaml.YAMLError) as e:
            self.get_logger().error(f'could not read the calibration {info_file}: {e} -- NO /camera/camera_info')
        self.size_warned = False

        self.running = True
        self.n_pictures = 0           # pictures published so far
        self.n_connects = 0           # successful (re)connections
        self.last_size = None
        self._stat_t = time.monotonic()
        self._stat_n = 0
        self.camera_thread = threading.Thread(target=self.capture_loop, name='capture', daemon=True)
        self.camera_thread.start()
        self.stats_timer = self.create_timer(5.0, self.log_stats)
        self.get_logger().info(f'Compressed camera publisher started ({self.url}, frame_id {self.frame_id}); '
                               f'stamp = arrival of each picture, each picture published once')

    # ------------------------------------------------------------------ capture thread (owns the stream)
    def capture_loop(self):
        cap = None
        failed_reads = 0
        last_error_log = 0.0
        try:
            while self.running and rclpy.ok():
                if cap is None:
                    cap = cv2.VideoCapture(self.url)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        if time.monotonic() - last_error_log > 10.0:
                            last_error_log = time.monotonic()
                            self.get_logger().error(f'Could not open camera stream {self.url} -- retrying every '
                                                    f'{self.reconnect_s:.0f} s')
                        self._sleep(self.reconnect_s)
                        continue
                    failed_reads = 0
                    self.n_connects += 1
                    if self.n_connects > 1:
                        self.get_logger().warn(f'camera stream reconnected (connection {self.n_connects})')
                ret, frame = cap.read()
                if not ret:
                    failed_reads += 1
                    time.sleep(0.01)
                    if failed_reads >= 100:           # ~1 s without a picture: the stream is gone
                        self.get_logger().warn('no pictures for 1 s -- closing and reconnecting')
                        cap.release()
                        cap = None
                    continue
                failed_reads = 0
                stamp = self.get_clock().now().to_msg()      # the picture ARRIVED now
                try:
                    self.publish_picture(frame, stamp)
                except Exception as e:                       # context shut down while we were publishing
                    if self.running and rclpy.ok():
                        self.get_logger().error(f'could not publish a picture: {e}')
        finally:
            if cap is not None:
                cap.release()                                # released by the thread that reads it

    def _sleep(self, seconds):
        """sleep in small steps so that a shutdown request is noticed quickly"""
        end = time.monotonic() + seconds
        while self.running and time.monotonic() < end:
            time.sleep(0.05)

    def publish_picture(self, frame, stamp):
        success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not success:
            return
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()
        self.publisher.publish(msg)
        self.n_pictures += 1
        self._stat_n += 1
        h, w = frame.shape[:2]
        self.last_size = (w, h)

        if self.camera_info is not None:
            if (w, h) != (self.camera_info.width, self.camera_info.height):
                if not self.size_warned:      # wrong calibration is worse than none: don't publish it
                    self.get_logger().error(
                        f'picture is {w} x {h} but the calibration is for {self.camera_info.width} x '
                        f'{self.camera_info.height} -- /camera/camera_info NOT published (recalibrate or convert)')
                    self.size_warned = True
                return
            self.camera_info.header = msg.header      # same stamp + frame_id = this info belongs to this picture
            self.info_publisher.publish(self.camera_info)

    # ------------------------------------------------------------------ executor thread
    def log_stats(self):
        now = time.monotonic()
        rate = self._stat_n / max(now - self._stat_t, 1e-9)
        self._stat_t, self._stat_n = now, 0
        size = f'{self.last_size[0]} x {self.last_size[1]}' if self.last_size else 'no picture yet'
        self.get_logger().info(f'{rate:.1f} new pictures/s published ({size}), {self.n_pictures} in total')

    def destroy_node(self):
        self.running = False
        t = getattr(self, 'camera_thread', None)
        if t is not None and t.is_alive():
            t.join(timeout=3.0)
            if t.is_alive():
                # the thread is stuck inside a blocking read/open.  NEVER release the stream from here
                # (that was the old segfault); the daemon thread ends with the process.
                self.get_logger().warn('capture thread still blocked in the stream -- leaving it to end with the process')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():                  # Ctrl+C has usually shut the context down already
            rclpy.shutdown()


if __name__ == '__main__':
    main()
