"""calib_capture -- record a camera-LiDAR calibration session: every time the checkerboard stands STILL in a NEW
position, save its 54 corner pixels, the camera picture and every LiDAR scan taken while it stood still.

WHY "STILL": the camera (~30 pictures/s) and the LiDAR (~7 scans/s; one turn takes ~0.14 s) do not measure at the
same moment, and both arrive with some delay. While the board moves, the picture and the scan show it in different
places -> wrong pairs. While it stands still, timing does not matter.

ON THE CAR (it stands still; the LiDAR driver, the camera node and apex_description sensor_frames are running):
  ros2 launch sensor_fusion calib_capture.launch.py                 (topics + out_dir are launch arguments)
  Hold the board (on stiff cardboard) so the LiDAR's scan plane crosses it. 20 positions: 1-2 m away, on the
  left / centre / right of the picture, turned up to +-35 deg AND tilted forward / back 15-35 deg, both ways (16
  gentler poses left the height / pitch WEAK in the end-to-end test). Hold each one still ~2 s until "pose N saved";
  at max_poses (20) it says "enough". Ctrl+C ends (the session is saved after every pose). Then:
  ros2 run sensor_fusion calib_solve ~/calib_sessions/<session> --mounts car
Feedback: the log (state every 2 s + what is still missing), and /calib_capture/debug (the picture with the corners,
the newest LiDAR scan drawn in red with the FIRST GUESS, the state) for Foxglove / RViz.
"""
import math
import os
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, LaserScan

from sensor_fusion import lidar_camera_calib as lc
from sensor_fusion import projection as pj
from sensor_fusion.calib_session import Session


def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def decode_image(msg, compressed):
    import cv2
    if compressed:
        return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    enc = msg.encoding.lower()
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    if enc in ('bgr8', 'rgb8'):
        img = a[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy() if enc == 'rgb8' else img.copy()
    if enc in ('mono8', '8uc1'):
        return cv2.cvtColor(a[:, :msg.width].copy(), cv2.COLOR_GRAY2BGR)
    raise ValueError(f'image encoding {msg.encoding!r} not supported (bgr8, rgb8, mono8)')


def board_angles(R_cb):
    """Board normal (away from the camera) -> (turned, tilted) in degrees: turned + = its right side away,
    tilted + = its top away from the camera."""
    n = R_cb[:, 2] if R_cb[2, 2] >= 0 else -R_cb[:, 2]
    return math.degrees(math.atan2(n[0], n[2])), math.degrees(math.atan2(n[1], n[2]))


class CalibCapture(Node):

    def __init__(self):
        super().__init__('calib_capture')

        def p(name, value):
            return self.declare_parameter(name, value).value

        self.compressed = bool(p('image_compressed', True))
        image_topic = p('image_topic', '/camera/image_compressed')
        info_topic = p('camera_info_topic', '/camera/camera_info')
        scan_topic = p('scan_topic', '/scan')
        out_dir = os.path.expanduser(p('out_dir', '~/calib_sessions'))
        self.pattern = (int(p('pattern_cols', 9)), int(p('pattern_rows', 6)))
        self.square = (float(p('square_x', 0.025)), float(p('square_y', 0.176 / 7)))
        self.still_s = float(p('still_s', 1.0))
        self.still_px = float(p('still_px', 0.8))
        self.min_scans = int(p('min_scans', 5))
        self.max_poses = int(p('max_poses', 20))
        self.new_view_m = float(p('new_view_m', 0.10))
        self.new_view_deg = float(p('new_view_deg', 8.0))
        hz = float(p('detect_hz', 5.0))
        self.publish_debug = bool(p('publish_debug', True))
        self.objp = lc.board_object_points(self.pattern, self.square)
        self.folder = os.path.join(out_dir, time.strftime('session_%Y%m%d_%H%M%S'))
        self.img_msg = self.img_done = None
        self.cam = self.session = None
        self.scans = deque(maxlen=200)                  # (stamp, ranges, meta) of the last ~25 s
        self.laser_frame = ''
        self.prev = None
        self.still_since = None
        self.still_corners = []
        self.saved = []                                 # (R_cb, t_cb, mean u, tilt)
        self.T_cam_laser = None
        self.last_status = 0.0
        self.warned = set()
        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        msg_type = CompressedImage if self.compressed else Image
        self.create_subscription(msg_type, image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, info_topic, self.on_info, qos_profile_sensor_data)
        self.create_subscription(LaserScan, scan_topic, self.on_scan, qos_profile_sensor_data)
        self.debug_pub = self.create_publisher(CompressedImage, '/calib_capture/debug', 1) if self.publish_debug else None
        self.create_timer(1.0 / hz, self.tick)
        self.get_logger().info(f'image {image_topic} ({"compressed" if self.compressed else "raw"}), '
                               f'{info_topic}, scan {scan_topic}; board {self.pattern[0]} x {self.pattern[1]} corners, '
                               f'squares {self.square[0] * 1000:.2f} x {self.square[1] * 1000:.2f} mm; save after '
                               f'{self.still_s:.1f} s still; session -> {self.folder}')

    # ---- inputs
    def on_image(self, msg):
        self.img_msg = msg

    def on_info(self, msg):
        if self.cam is None:
            self.cam = pj.CameraModel.from_camera_info(msg)
            self.session = Session(self.cam.K, self.cam.D, msg.width, msg.height, self.pattern, self.square,
                                   camera_frame=msg.header.frame_id)
            self.get_logger().info(f'camera_info: {msg.width} x {msg.height}, fx {self.cam.fx:.1f} fy {self.cam.fy:.1f}, '
                                   f'frame {msg.header.frame_id!r}')

    def on_scan(self, msg):
        self.laser_frame = msg.header.frame_id
        self.scans.append((stamp_s(msg.header), np.asarray(msg.ranges, np.float32),
                           (msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max)))

    def warn_once(self, key, text):
        if key not in self.warned:
            self.warned.add(key)
            self.get_logger().warn(text)

    def lookup_tf(self):
        """The first guess T_cam_from_laser from TF (sensor_frames) -- only for the live "LiDAR sees the board"
        check and stored with the session; the solver takes its first guess from the mount YAML."""
        if self.T_cam_laser is not None or not self.laser_frame or self.session is None:
            return
        cam_frame = self.session.camera_frame or 'camera_optical_frame'
        try:
            tf = self.tf_buffer.lookup_transform(cam_frame, self.laser_frame, Time())
        except Exception:
            self.warn_once('tf', f'no TF {self.laser_frame} -> {cam_frame} yet (is sensor_frames running?): '
                                 'saving without the live LiDAR check')
            return
        self.T_cam_laser = pj.T_from_tf(tf)
        self.session.laser_frame, self.session.T_tf = self.laser_frame, self.T_cam_laser
        self.get_logger().info(f'first guess from TF {self.laser_frame} -> {cam_frame} found')

    # ---- the loop
    def tick(self):
        import cv2
        now = time.time()
        if self.session is None:
            self.status(now, 'waiting for camera_info ...')
            return
        msg = self.img_msg
        if msg is None or msg is self.img_done:
            return
        self.img_done = msg
        t_img = stamp_s(msg.header)
        img = decode_image(msg, self.compressed)
        if img is None or img.shape[1] != self.cam.width or img.shape[0] != self.cam.height:
            self.warn_once('size', f'picture {None if img is None else img.shape[1::-1]} != camera_info '
                                   f'{self.cam.width} x {self.cam.height}: wrong calibration for this stream')
            return
        self.lookup_tf()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, self.pattern,
                                                     flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY)
        if not found:
            self.prev, self.still_since, self.still_corners = None, None, []
            self.status(now, 'no board in the picture (the WHOLE board must be visible)')
            self.debug(img, None, None, 'no board')
            return
        c = corners.reshape(-1, 2).astype(float)
        # STILL = every detection of the window within still_px of the window's MEAN (not only of the previous
        # frame: a board still drifting slowly at the end of a move must not count -- 2026-10-14)
        ref = np.mean(self.still_corners, 0) if self.still_corners else None
        moved = math.inf if ref is None else float(np.sqrt(np.mean(np.sum((c - ref) ** 2, 1))))
        self.prev = c
        if moved > self.still_px or self.still_since is None:
            self.still_since, self.still_corners = t_img, [c]
        else:
            self.still_corners.append(c)
        held = t_img - self.still_since
        pose = lc.board_pose_pnp(c, self.cam, self.objp)
        if pose is None:
            self.status(now, 'board found, but its pose failed (PnP)')
            return
        R_cb, t_cb, _ = pose
        turn, tilt = board_angles(R_cb)
        on_board = self.lidar_on_board(R_cb, t_cb)
        novel = all(np.linalg.norm(t_cb - t) > self.new_view_m or
                    math.degrees(math.acos(min(1.0, abs(float(R_cb[:, 2] @ R[:, 2]))))) > self.new_view_deg
                    for R, t, _, _ in self.saved)
        state = (f'board {np.linalg.norm(t_cb):.2f} m, turned {turn:+.0f}, tilted {tilt:+.0f} deg, '
                 f'{"still " + format(held, ".1f") + " s" if held > 0 else "moving"}, '
                 f'LiDAR points on it: {"?" if on_board is None else on_board}')
        if len(self.saved) >= self.max_poses:
            self.status(now, f'{len(self.saved)} poses saved = enough: Ctrl+C, then calib_solve')
        elif not novel:
            self.status(now, state + ' -- already saved: move to a NEW position / angle')
        elif on_board is not None and on_board == 0:
            self.status(now, state + ' -- the LiDAR does not see the board: raise / lower it so the scan crosses it')
        elif held >= self.still_s:
            scans = [s for s in self.scans if self.still_since - 0.05 <= s[0] <= t_img + 0.05]
            if len(scans) < self.min_scans:
                self.status(now, state + f' -- only {len(scans)} LiDAR scans while still (want {self.min_scans}): '
                                         'hold still a little longer')
            else:
                k = self.session.add_pose(np.mean(self.still_corners, 0), t_img, scans)
                cv2.imwrite(os.path.join(self.folder, f'pose_{k:02d}.png'), img)
                self.session.save(self.folder)
                self.saved.append((R_cb, t_cb, float(c[:, 0].mean()), tilt))
                self.get_logger().info(f'pose {k} saved: {state}, {len(scans)} scans -- {self.coverage()}')
                self.still_since, self.still_corners = t_img, [c]
        else:
            self.status(now, state)
        self.debug(img, c, (R_cb, t_cb), state)

    def lidar_on_board(self, R_cb, t_cb):
        """Roughly how many points of the newest scan lie on the board, judged with the FIRST GUESS (TF) -- only to
        catch a scan that misses the board completely, so the gates are generous (a guess 2 deg off already moves
        the points ~7 cm at 2 m; job 131 lost 2 poses with a 6 cm gate). None without TF / scans."""
        if self.T_cam_laser is None or not self.scans:
            return None
        _, r, meta = self.scans[-1]
        pts, _ = pj.scan_to_points(r, *meta)
        Xb = (pj.apply(self.T_cam_laser, pts) - t_cb) @ R_cb
        m = lc.BOARD_MARGIN + 0.15
        lo, hi = self.objp[:, :2].min(0) - m, self.objp[:, :2].max(0) + m
        return int(np.count_nonzero((np.abs(Xb[:, 2]) < 0.20) & np.all(Xb[:, :2] >= lo, 1) & np.all(Xb[:, :2] <= hi, 1)))

    def coverage(self):
        """What the saved poses cover -- and what is still missing."""
        w = self.cam.width
        third = [sum(1 for _, _, u, _ in self.saved if lo <= u < hi) for lo, hi in ((0, w / 3), (w / 3, 2 * w / 3), (2 * w / 3, w))]
        tilts = [t for _, _, _, t in self.saved]
        fwd, back = sum(1 for t in tilts if t < -10), sum(1 for t in tilts if t > 10)
        miss = [nm for nm, n in zip(('LEFT', 'CENTRE', 'RIGHT'), third) if n == 0]
        if fwd == 0:
            miss.append('tilted top TOWARDS the camera')
        if back == 0:
            miss.append('tilted top AWAY')
        return (f'{len(self.saved)} saved (left {third[0]}, centre {third[1]}, right {third[2]}; tilted towards {fwd}, '
                f'away {back})' + (f'; still missing: {", ".join(miss)}' if miss else ''))

    def status(self, now, text):
        if now - self.last_status >= 2.0:
            self.last_status = now
            self.get_logger().info(text)

    def debug(self, img, corners, pose, text):
        if self.debug_pub is None or self.debug_pub.get_subscription_count() == 0:
            return
        import cv2
        out = img.copy()
        if corners is not None:
            cv2.drawChessboardCorners(out, self.pattern, corners.reshape(-1, 1, 2).astype(np.float32), True)
        if self.T_cam_laser is not None and self.scans:
            _, r, meta = self.scans[-1]
            pts, _ = pj.scan_to_points(r, *meta)
            uv, ok = self.cam.project(pj.apply(self.T_cam_laser, pts))
            for u, v in uv[ok & self.cam.in_image(uv)]:
                cv2.circle(out, (int(u), int(v)), 2, (0, 0, 255), -1)
        cv2.putText(out, text[:90], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f'{len(self.saved)} poses saved', (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        m = CompressedImage()
        m.header.stamp = self.get_clock().now().to_msg()
        m.format = 'jpeg'
        m.data = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        self.debug_pub.publish(m)

    def finish(self):
        if self.session is not None and self.session.corners:
            self.session.save(self.folder)
            self.get_logger().info(f'session saved: {self.folder} ({len(self.session.corners)} poses, '
                                   f'{len(self.session.scan_ranges)} scans). Next: ros2 run sensor_fusion calib_solve '
                                   f'{self.folder} --mounts car')
        else:
            self.get_logger().info('no pose saved')


def main(args=None):
    rclpy.init(args=args)
    node = CalibCapture()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.finish()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
