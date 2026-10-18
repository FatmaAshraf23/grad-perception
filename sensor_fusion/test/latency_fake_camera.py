"""latency_fake_camera.py -- a FAKE camera to test latency_probe without hardware (a test helper, not installed).

It 'films' latency_probe's blinking window through /latency_probe/flips (one message per switch, stamped with the
switch time): a picture taken at time c shows the window as the screen showed it at c (the screen being display_s
late); the picture arrives latency_s later; it is then either published by a timer_hz timer stamped when it is
PUBLISHED -- what camera_publisher camera_node does today -- or (stamp_on_arrival) published at once, stamped with its
arrival time. Expected latency_probe stamp offset:
  timer:      display_s + latency_s + 1 / (2 * timer_hz)   (default 0.020 + 0.110 + 0.0167 = 0.147 s)
  on arrival: display_s + latency_s                         (0.130 s)
  python3 latency_fake_camera.py [--ros-args -p latency_s:=0.11 -p display_s:=0.02 -p stamp_on_arrival:=false]
"""
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header


class FakeCamera(Node):

    def __init__(self):
        super().__init__('latency_fake_camera')

        from rcl_interfaces.msg import ParameterDescriptor

        def p(name, value):             # dynamic typing: numbers may be given as 30 or 30.0
            return self.declare_parameter(name, value, ParameterDescriptor(dynamic_typing=True)).value

        import cv2
        self.cv2 = cv2
        self.latency, self.display = float(p('latency_s', 0.110)), float(p('display_s', 0.020))
        self.fps, timer_hz = float(p('fps', 29.97)), float(p('timer_hz', 30.0))
        self.on_arrival = bool(p('stamp_on_arrival', False))
        self.flips = []                       # (switch time, level)
        self.create_subscription(Header, '/latency_probe/flips', self.on_flip, 10)
        self.pub = self.create_publisher(CompressedImage, p('topic', '/camera/image_compressed'),
                                         qos_profile_sensor_data)
        self.c0 = time.time() + 0.3           # the first picture's exposure middle
        self.j = 0                            # next picture to arrive
        self.latest, self.last_sent = None, None
        self.rng = np.random.default_rng(0)
        self.n_pub = self.n_dup = 0
        self.create_timer(0.002, self.arrivals)
        if not self.on_arrival:
            self.create_timer(1.0 / timer_hz, self.on_timer)
        self.get_logger().info(f'fake camera: {self.fps:.2f} pictures/s, latency {self.latency * 1000:.0f} ms, screen '
                               f'{self.display * 1000:.0f} ms, stamped '
                               + ('on arrival' if self.on_arrival else f'by a {timer_hz:.0f} Hz timer (camera_node)'))

    def on_flip(self, msg):                   # std_msgs/Header: stamp = the switch time, frame_id white / black
        self.flips.append((msg.stamp.sec + msg.stamp.nanosec * 1e-9, 1 if msg.frame_id == 'white' else 0))

    def white_at(self, t):
        lev = 0
        for tf, level in self.flips:          # the screen shows a switch display_s after it was drawn
            if tf + self.display <= t:
                lev = level
        return lev == 1

    def render(self, j, white):
        cv2 = self.cv2
        img = np.full((480, 640), 100, np.uint8)
        img[120:360, 160:480] = 230 if white else 20
        img = cv2.add(img, self.rng.integers(0, 4, (480, 640), dtype=np.uint8))
        cv2.putText(img, str(j), (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 128, 1)     # every picture different
        return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    def arrivals(self):
        now = time.time()
        while self.c0 + self.j / self.fps + self.latency <= now:
            c = self.c0 + self.j / self.fps
            self.latest = (self.j, self.render(self.j, self.white_at(c)), c + self.latency)
            self.j += 1
            if self.on_arrival:
                self.send(self.latest[1], self.latest[2], False)

    def on_timer(self):
        if self.latest is not None:
            self.send(self.latest[1], time.time(), self.latest[0] == self.last_sent)
            self.last_sent = self.latest[0]

    def send(self, data, stamp, dup):
        m = CompressedImage()
        m.header.stamp = Time(nanoseconds=int(stamp * 1e9)).to_msg()
        m.header.frame_id = 'camera_optical_frame'
        m.format = 'jpeg'
        m.data = data
        self.pub.publish(m)
        self.n_pub += 1
        self.n_dup += int(dup)


def main():
    rclpy.init()
    node = FakeCamera()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.get_logger().info(f'fake camera: {node.n_pub} pictures sent, {node.n_dup} of them duplicates, '
                               f'{len(node.flips)} switches heard')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
