#!/usr/bin/env python3
"""detection_bridge -- the car camera's boxes from the Pi host (pi_camera/cam_detect.py, UDP) into ROS (2026-11-08).

  UDP 127.0.0.1:5600 (detection_packet) -> /camera/detections (apex_msgs/CameraDetectionArray, source "pi_camera")
                                         -> /camera/camera_info (the calibration, scaled to the picture size)
The same messages yolo_node (laptop) and fake_camera (simulator) publish, so sensor_fusion fusion_node works unchanged:
header.stamp = the time the PICTURE was taken (from the packet), frame_id = the camera's optical frame.

Checks: packets older than max_age_s (a stalled host process / a clock problem) are dropped and counted; a sequence
gap counts lost pictures; a stamp in the FUTURE (> 50 ms) is logged (the host and container clocks must be one).
camera_info: from camera_info_file (camcal_calibrate.py YAML, e.g. calibration_1640x1232_pi.yaml), scaled to the
packet's w x h when they differ (a pure shrink of the same view -- the camera must stream the calibrated framing).

Parameters: port 5600, camera_info_file '', frame_id camera_optical_frame, max_age_s 0.5, stats_every_s 5.0
"""
import socket
import time

import rclpy
from apex_msgs.msg import CameraDetection, CameraDetectionArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo

from camera_publisher import detection_packet as pkt

SCHEMA = 'apex-camera-detections/0.1'


class DetectionBridge(Node):

    def __init__(self):
        super().__init__('detection_bridge')

        def p(n, v):
            return self.declare_parameter(n, v).value
        self.port = int(p('port', 5600))
        self.frame_id = p('frame_id', 'camera_optical_frame')
        self.max_age_ns = int(float(p('max_age_s', 0.5)) * 1e9)
        self.stats_every = float(p('stats_every_s', 5.0))
        info_file = p('camera_info_file', '')
        self.cal = pkt.load_camera_yaml(info_file) if info_file else None
        self.pub = self.create_publisher(CameraDetectionArray, p('detections_topic', '/camera/detections'), 10)
        self.pub_info = self.create_publisher(CameraInfo, p('camera_info_topic', '/camera/camera_info'),
                                              qos_profile_sensor_data)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', self.port))
        self.sock.setblocking(False)
        self.last_seq, self.seq_out = None, 0
        self.st = dict(rx=0, pub=0, bad=0, too_old=0, old_seq=0, future=0, lost=0, boxes=0, lat=[], det=[])
        self.last_rx = time.monotonic()
        self.last_stats = time.monotonic()
        self.info_cache = {}
        self.create_timer(0.005, self.poll)                  # 200 Hz poll: < 5 ms added delay, cheap when idle
        self.get_logger().info(
            f'listening on UDP {self.port} -> /camera/detections (frame {self.frame_id}); camera_info '
            + (f"{self.cal['width']}x{self.cal['height']} from {info_file}" if self.cal else 'NOT published (no file)'))

    def info_for(self, w, h, stamp):
        if self.cal is None:
            return None
        key = (w, h)
        if key not in self.info_cache:
            K = self.cal['K'] if (w, h) == (self.cal['width'], self.cal['height']) else \
                pkt.scale_camera_info(self.cal['K'], self.cal['width'], self.cal['height'], w, h)
            m = CameraInfo()
            m.width, m.height = w, h
            m.distortion_model = 'plumb_bob'
            m.k = [float(v) for v in K]
            m.d = [float(v) for v in self.cal['D']]
            m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            m.p = [m.k[0], m.k[1], m.k[2], 0.0, m.k[3], m.k[4], m.k[5], 0.0, 0.0, 0.0, 1.0, 0.0]
            self.info_cache[key] = m
            if (w, h) != (self.cal['width'], self.cal['height']):
                self.get_logger().info(f"camera_info scaled {self.cal['width']}x{self.cal['height']} -> {w}x{h}")
        m = self.info_cache[key]
        m.header.stamp, m.header.frame_id = stamp, self.frame_id
        return m

    def poll(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            self.on_packet(data)
        self.report()

    def on_packet(self, data):          # NOT "handle": rclpy Node uses self.handle for its C handle
        st = self.st
        st['rx'] += 1
        self.last_rx = time.monotonic()
        try:
            d = pkt.decode(data)
        except ValueError as e:
            st['bad'] += 1
            if st['bad'] == 1:
                self.get_logger().warn(f'bad packet: {e}')
            return
        now_ns = self.get_clock().now().nanoseconds
        ok, why, lost = pkt.check_packet(d, now_ns, self.last_seq, self.max_age_ns)
        st['lost'] += lost
        if why:
            st[why] += 1
        if why != 'old_seq':
            self.last_seq = d['seq']               # also for a too-old one: its seq is not "lost" again later
        if not ok:
            return
        msg = CameraDetectionArray()
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(int(d['stamp_ns']), 1_000_000_000)
        msg.header.frame_id = self.frame_id
        msg.schema_version = SCHEMA
        msg.sequence_id = self.seq_out
        self.seq_out += 1
        msg.source = 'pi_camera'
        msg.image_width, msg.image_height = int(d['w']), int(d['h'])
        for c, conf, u0, v0, u1, v1 in d['boxes']:
            b = CameraDetection()
            b.class_name, b.class_id, b.confidence = c, 0 if c == 'cone' else -1, float(conf)
            b.u_min, b.v_min, b.u_max, b.v_max = float(u0), float(v0), float(u1), float(v1)
            msg.detections.append(b)
        msg.latency_ms = float((now_ns - int(d['stamp_ns'])) / 1e6)
        self.pub.publish(msg)
        info = self.info_for(msg.image_width, msg.image_height, msg.header.stamp)
        if info is not None:
            self.pub_info.publish(info)
        st['pub'] += 1
        st['boxes'] += len(msg.detections)
        st['lat'].append(msg.latency_ms)
        st['det'].append(float(d['det_ms']))

    def report(self):
        now = time.monotonic()
        if now - self.last_stats < self.stats_every:
            return
        st = self.st
        if st['rx'] == 0:
            self.get_logger().warn(f'no camera packets for {now - self.last_rx:.0f} s -- is cam_detect.py running on the '
                                   f'host (port {self.port})?')
        else:
            la = sorted(st['lat']) or [0.0]
            de = st['det'] or [0.0]
            self.get_logger().info(
                f"{st['pub']} pictures published ({st['pub'] / (now - self.last_stats):.1f}/s), {st['boxes']} boxes; "
                f"picture -> published {sum(la) / len(la):.0f} ms mean, p95 {la[int(0.95 * (len(la) - 1))]:.0f}; "
                f"detector {sum(de) / len(de):.1f} ms; dropped: too old {st['too_old']}, out of order {st['old_seq']}, "
                f"bad {st['bad']}; lost (seq gaps) {st['lost']}; stamps in the future {st['future']}")
        self.st = dict(rx=0, pub=0, bad=0, too_old=0, old_seq=0, future=0, lost=0, boxes=0, lat=[], det=[])
        self.last_stats = now


def main(args=None):
    rclpy.init(args=args)
    node = DetectionBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
