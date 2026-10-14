"""camera_node -- Windows Flask MJPEG stream -> /camera/image_compressed (JPEG) + /camera/camera_info.

2026-10-14 (camera calibration step 4): added /camera/camera_info (from the calibration YAML, same stamp + frame_id
as every image), frame_id camera_optical_frame, parameters url / camera_info_file / frame_id, and a size check (the
calibration is only valid for the picture size it was made for). Original: camera_node.py.bak_job112.
"""
import os
import threading

import cv2
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage


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

        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            self.get_logger().error(f'Could not open camera stream {self.url}')
            return

        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.camera_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.camera_thread.start()
        self.timer = self.create_timer(1.0 / 30.0, self.publish_frame)
        self.get_logger().info(f'Compressed camera publisher started ({self.url}, frame_id {self.frame_id})')

    def capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.latest_frame = frame

    def publish_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()

        success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not success:
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()
        self.publisher.publish(msg)

        if self.camera_info is not None:
            h, w = frame.shape[:2]
            if (w, h) != (self.camera_info.width, self.camera_info.height):
                if not self.size_warned:      # wrong calibration is worse than none: don't publish it
                    self.get_logger().error(
                        f'picture is {w} x {h} but the calibration is for {self.camera_info.width} x '
                        f'{self.camera_info.height} -- /camera/camera_info NOT published (recalibrate or convert)')
                    self.size_warned = True
                return
            self.camera_info.header = msg.header      # same stamp + frame_id = this info belongs to this picture
            self.info_publisher.publish(self.camera_info)

    def destroy_node(self):
        self.running = False
        if hasattr(self, 'camera_thread'):
            self.camera_thread.join(timeout=1.0)
        if hasattr(self, 'cap'):
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
