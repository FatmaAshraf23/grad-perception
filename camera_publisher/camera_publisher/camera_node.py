import cv2
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy
)

from sensor_msgs.msg import CompressedImage


class CameraPublisher(Node):

    def __init__(self):
        super().__init__('camera_publisher')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        self.publisher = self.create_publisher(
            CompressedImage,
            '/camera/image_compressed',
            qos
        )

        self.url = "http://172.29.48.1:5000/video"

        self.cap = cv2.VideoCapture(self.url)

        if not self.cap.isOpened():
            self.get_logger().error(
                'Could not open camera stream'
            )
            return

        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

        self.camera_thread = threading.Thread(
            target=self.capture_frames,
            daemon=True
        )

        self.camera_thread.start()

        self.timer = self.create_timer(
            1.0 / 30.0,
            self.publish_frame
        )

        self.get_logger().info(
            'Compressed camera publisher started'
        )

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

        # Encode the frame as JPEG
        success, encoded = cv2.imencode(
            '.jpg',
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                70
            ]
        )

        if not success:
            return

        msg = CompressedImage()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.format = 'jpeg'

        msg.data = encoded.tobytes()

        self.publisher.publish(msg)

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
