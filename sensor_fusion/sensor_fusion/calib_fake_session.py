"""calib_fake_session -- a FAKE calibration session in ROS, to test calib_capture + calib_solve without the hardware.

Publishes what the car would during a calibration: /camera/image_compressed (rendered pictures of our checkerboard
through the lens's distortion, with blur, noise and JPEG), /camera/camera_info, and a 2-D LiDAR's /scan (rays cast
against the board and a 6 x 5 m room, + range noise), while a virtual helper holds the board STILL in 20 positions
(still_s each) and MOVES it in between (move_s). It stops by itself after the last position.
The truth = the mount YAML given (default car): calib_solve --truth car prints the errors.
  <install>/lib/sensor_fusion/calib_fake_session --ros-args -p still_s:=2.5
"""
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, LaserScan

from sensor_fusion import lidar_camera_calib as lc
from sensor_fusion import projection as pj
from sensor_fusion import synthetic as sy
from sensor_fusion.render import Renderer

IPHONE_640 = ([[748.472839, 0, 335.172741], [0, 1000.794742, 235.273704], [0, 0, 1]],
              [0.138469, 0.185298, -0.007312, 0.008586, 0.0], 640, 480)


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class FakeSession(Node):

    def __init__(self):
        super().__init__('calib_fake_session')

        def p(name, value):
            return self.declare_parameter(name, value).value

        from apex_description.mounts import load_mounts
        import cv2
        self.cv2 = cv2
        mounts = p('mounts', 'car')
        camera_yaml = p('camera_info_file', '')
        self.still_s, self.move_s = float(p('still_s', 2.5)), float(p('move_s', 1.5))
        start = float(p('start_delay_s', 3.0))
        fps, scan_hz = float(p('fps', 15.0)), float(p('scan_hz', 7.0))
        self.sigma_r, self.jpeg = float(p('sigma_r', 0.01)), int(p('jpeg_quality', 90))
        seed, tilted = int(p('seed', 0)), bool(p('tilted', True))
        self.cam_frame, self.laser_frame = p('camera_frame', 'camera_optical_frame'), p('laser_frame', 'laser')
        m = load_mounts(mounts)
        Ts = pj.mount_transforms(m)
        self.T_bl, self.T_ob = Ts['laser'], pj.invert(Ts['camera_optical_frame'])
        self.cam = pj.CameraModel.from_yaml(camera_yaml) if camera_yaml else pj.CameraModel(*IPHONE_640)
        self.objp = lc.board_object_points()
        self.specs = sy.default_specs(self.T_bl[2, 3], tilted)
        self.renderer = Renderer(self.cam, seed=seed)
        self.rng = np.random.default_rng(seed + 1)
        self.scan_hz = scan_hz
        info = CameraInfo()
        info.header.frame_id = self.cam_frame
        info.width, info.height = self.cam.width, self.cam.height
        info.distortion_model = 'plumb_bob'
        info.k = [float(v) for v in self.cam.K.ravel()]
        info.d = [float(v) for v in self.cam.D]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [float(v) for v in np.hstack([self.cam.K, np.zeros((3, 1))]).ravel()]
        self.info = info
        q = qos_profile_sensor_data
        self.pub_img = self.create_publisher(CompressedImage, p('image_topic', '/camera/image_compressed'), q)
        self.pub_info = self.create_publisher(CameraInfo, p('camera_info_topic', '/camera/camera_info'), q)
        self.pub_scan = self.create_publisher(LaserScan, p('scan_topic', '/scan'), q)
        self.t0 = time.time() + start
        self.done = False
        self.create_timer(1.0 / fps, self.on_frame)
        self.create_timer(1.0 / scan_hz, self.on_scan)
        self.get_logger().info(f'fake calibration session: {len(self.specs)} board positions, {self.still_s:.1f} s '
                               f'still + {self.move_s:.1f} s moving each, starts in {start:.0f} s; truth {m["_path"]}; '
                               f'camera {self.cam.width} x {self.cam.height} at {fps:.0f} Hz, LiDAR {scan_hz:.0f} Hz')

    def spec_at(self, t):
        """The board's position at time t: None before the start, 'done' after the last position."""
        tau = t - self.t0
        if tau < 0:
            return None
        per = self.still_s + self.move_s
        k = int(tau // per)
        if k >= len(self.specs):
            return 'done'
        u = tau - k * per
        if u < self.still_s or k + 1 >= len(self.specs):
            return self.specs[k]
        w = smoothstep((u - self.still_s) / self.move_s)
        return tuple(a + w * (b - a) for a, b in zip(self.specs[k], self.specs[k + 1]))

    def board(self, spec):
        dist, lat, zc, turn, tilt = spec
        return sy.board_pose((self.T_bl[0, 3] + dist, lat, zc), turn, tilt, self.objp)

    def on_frame(self):
        spec = self.spec_at(time.time())
        if spec == 'done':
            if not self.done:
                self.done = True
                self.get_logger().info(f'fake session done: {len(self.specs)} positions shown')
            return
        img = self.renderer.render(None if spec is None else self.T_ob @ self.board(spec))
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cam_frame
        msg.format = 'jpeg'
        msg.data = self.cv2.imencode('.jpg', img, [self.cv2.IMWRITE_JPEG_QUALITY, self.jpeg])[1].tobytes()
        self.pub_img.publish(msg)
        self.info.header.stamp = msg.header.stamp
        self.pub_info.publish(self.info)

    def on_scan(self):
        spec = self.spec_at(time.time())
        if spec == 'done':
            return
        boards = [] if spec is None else [self.board(spec)]
        r, a0, inc = sy.raycast_scan(self.T_bl, boards, self.objp, sigma=self.sigma_r, rng=self.rng)
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.laser_frame
        msg.angle_min, msg.angle_increment = float(a0), float(inc)
        msg.angle_max = float(a0 + inc * (len(r) - 1))
        msg.scan_time = 1.0 / self.scan_hz
        msg.range_min, msg.range_max = 0.15, 12.0
        msg.ranges = r.astype(np.float32).tolist()
        self.pub_scan.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeSession()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
