import time

import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy

from sensor_msgs.msg import CompressedImage

from ultralytics import YOLO

import cv2
import numpy as np


class YoloNode(Node):

    def __init__(self):
        super().__init__('yolo_node')

        # Load pretrained YOLO11n
        self.model = YOLO('/home/fatma/ros2_ws/best.pt')

        # Match the camera publisher QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_compressed',
            self.image_callback,
            qos
        )

        self.get_logger().info('YOLO node started')

    def image_callback(self, msg):

        # Convert JPEG bytes to NumPy array
        np_arr = np.frombuffer(
            msg.data,
            np.uint8
        )

        # Decode JPEG
        frame = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            self.get_logger().error(
                'Failed to decode image'
            )
            return

        # Start timing YOLO inference
        start_time = time.perf_counter()

        # Run YOLO on RTX 3050
        results = self.model(
            frame,
            device=0,
            verbose=False
        )

        # Calculate inference time
        inference_time = time.perf_counter() - start_time

        if inference_time > 0:
            fps = 1.0 / inference_time

            self.get_logger().info(
                f'YOLO inference: '
                f'{inference_time * 1000:.1f} ms | '
                f'FPS: {fps:.1f}'
            )

        # Draw detections
        annotated_frame = results[0].plot()

        # Display result
        cv2.imshow(
            'YOLO - ROS 2 Camera',
            annotated_frame
        )

        cv2.waitKey(1)


def main(args=None):

    rclpy.init(args=args)

    node = YoloNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
