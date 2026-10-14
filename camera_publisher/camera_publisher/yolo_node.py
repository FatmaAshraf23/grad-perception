"""yolo_node -- YOLO cone detector on /camera/image_compressed -> apex_msgs/CameraDetectionArray on /camera/detections.

2026-10-14 (fusion prep step 4): the boxes are now PUBLISHED (before: only drawn in a window), stamped with the
PICTURE's time and its frame_id, so the fusion node can combine them with the LiDAR. The window is optional
(parameter show, default true = as before; false on the car / without a display). Inference time is logged every 2 s
instead of every picture. Parameters: model, image_topic, detections_topic, device (0 = first GPU, 'cpu'),
conf (minimum confidence), show.
"""
import time

import cv2
import numpy as np
import rclpy
from apex_msgs.msg import CameraDetection, CameraDetectionArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO


class YoloNode(Node):

    def __init__(self):
        super().__init__('yolo_node')
        p = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        model = p('model', '/home/fatma/ros2_ws/best.pt')
        image_topic = p('image_topic', '/camera/image_compressed')
        det_topic = p('detections_topic', '/camera/detections')
        dev = str(p('device', '0'))
        self.device = int(dev) if dev.isdigit() else dev
        self.conf = float(p('conf', 0.25))
        self.show = bool(p('show', True))
        self.model = YOLO(model)
        self.names = self.model.names
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.subscription = self.create_subscription(CompressedImage, image_topic, self.image_callback, qos)
        self.pub = self.create_publisher(CameraDetectionArray, det_topic, 10)
        self.seq = 0
        self.times, self.last_log = [], time.time()
        self.get_logger().info(f'YOLO node started: model {model} (classes {dict(self.names)}), device {self.device}, '
                               f'conf >= {self.conf}, {image_topic} -> {det_topic}, window {self.show}')

    def image_callback(self, msg):
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().error('Failed to decode image')
            return
        t0 = time.perf_counter()
        results = self.model(frame, device=self.device, conf=self.conf, verbose=False)
        self.times.append(time.perf_counter() - t0)

        out = CameraDetectionArray()
        out.header = msg.header                       # the PICTURE's time + the camera's frame
        out.schema_version = 'apex-camera-detections/0.1'
        out.sequence_id = self.seq
        self.seq += 1
        out.source = 'yolo'
        out.image_height, out.image_width = frame.shape[:2]
        boxes = results[0].boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for (u0, v0, u1, v1), c, k in zip(xyxy, conf, cls):
                d = CameraDetection()
                d.class_name = str(self.names.get(int(k), str(k))) if isinstance(self.names, dict) else str(k)
                d.class_id = int(k)
                d.confidence = float(c)
                d.u_min, d.v_min, d.u_max, d.v_max = float(u0), float(v0), float(u1), float(v1)
                out.detections.append(d)
        now = self.get_clock().now().nanoseconds * 1e-9
        out.latency_ms = float((now - (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)) * 1000.0)
        self.pub.publish(out)

        if time.time() - self.last_log >= 2.0:
            t = np.array(self.times) * 1000.0
            self.get_logger().info(f'YOLO: {len(t)} pictures in the last 2 s, inference mean {t.mean():.1f} ms '
                                   f'(max {t.max():.1f}), {len(out.detections)} boxes in the last one, '
                                   f'picture -> boxes {out.latency_ms:.0f} ms')
            self.times, self.last_log = [], time.time()
        if self.show:
            cv2.imshow('YOLO - ROS 2 Camera', results[0].plot())
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.show:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
