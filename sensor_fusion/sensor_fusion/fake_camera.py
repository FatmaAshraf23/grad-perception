"""fake_camera -- the camera + YOLO for the SIMULATOR (it has no camera). Publishes apex_msgs/CameraDetectionArray on
/camera/detections and /camera/camera_info, as camera_publisher yolo_node + camera_node do on the car, made from the
TRUE cone positions and the TRUE pose of the car:
  1. where each cone is seen from: the true pose (/ego_racecar/odom) + the camera mount (apex_description
     sensor_mounts_sim.yaml) -> its box in the picture (projection.cone_box with the camera's lens);
  2. seen only if in front and mostly inside the picture (>= min_visible), within max_range_m, not hidden behind a
     wall or another cone (line of sight on the simulator's WORLD map) and taller than min_box_px;
  3. like a real detector: box edges jitter (box_noise_px + box_noise_frac of its size), small cones are missed more
     (p_miss for big ones, rising to 100 % at min_box_px), now and then a false box (false_per_picture), the
     confidence grows with the box size;
  4. like a real camera + YOLO: a picture every 1/fps s whose boxes are published latency_s later, stamped with the
     PICTURE's time (the pose comes from a short history of /ego_racecar/odom).
frame_id: 'camera_optical_frame' as on the car. In the simulator use est/camera_optical_frame: the ESTIMATED car's
camera -- what the fusion would look up in TF on the car (map -> camera at the picture's time).
"""
import csv
import math
import os
import time
from collections import deque

import numpy as np
import rclpy
from apex_msgs.msg import CameraDetection, CameraDetectionArray
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo

from sensor_fusion import projection as pj


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def load_cones(path):
    with open(path) as f:
        return [dict(label=r['label'], x=float(r['x']), y=float(r['y']), size=float(r.get('size_m') or 0.2))
                for r in csv.DictReader(f)]


class OccupancyMap:
    """A map_server map (yaml + image): is (x, y) inside a wall / cone? (nav2 'trinary': p = (255 - pixel) / 255)."""

    def __init__(self, yaml_path):
        import cv2
        import yaml
        with open(yaml_path) as f:
            c = yaml.safe_load(f)
        img_path = c['image'] if os.path.isabs(c['image']) else os.path.join(os.path.dirname(yaml_path), c['image'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f'cannot read the map image {img_path}')
        p = img / 255.0 if int(c.get('negate', 0)) else (255.0 - img) / 255.0
        self.occ = np.flipud(p > float(c.get('occupied_thresh', 0.65)))          # row 0 = the lowest y
        self.res = float(c['resolution'])
        self.ox, self.oy = float(c['origin'][0]), float(c['origin'][1])

    def occupied(self, x, y):
        i = np.floor((np.asarray(x) - self.ox) / self.res).astype(int)
        j = np.floor((np.asarray(y) - self.oy) / self.res).astype(int)
        inside = (i >= 0) & (j >= 0) & (i < self.occ.shape[1]) & (j < self.occ.shape[0])
        out = np.zeros(np.shape(i), bool)
        out[inside] = self.occ[j[inside], i[inside]]
        return out

    def line_of_sight(self, x0, y0, x1, y1, stop_before):
        """True if nothing occupied lies between (x0, y0) and (x1, y1) -- the last `stop_before` m (the target
        itself) not checked."""
        d = math.hypot(x1 - x0, y1 - y0) - stop_before
        if d <= 0:
            return True
        s = np.arange(0.0, d, self.res / 2) / (d + stop_before)
        return not self.occupied(x0 + s * (x1 - x0), y0 + s * (y1 - y0)).any()


def fake_detections(cam, T_opt_map, cam_xy, cones, cone_height, world, rng, cfg, stats=None):
    """The detector's boxes for one picture -> [(class_name, conf, u0, v0, u1, v1, label or None)]."""
    out = []
    for c in cones:
        dist = math.hypot(c['x'] - cam_xy[0], c['y'] - cam_xy[1])
        if dist > cfg['max_range_m']:
            continue
        b = pj.cone_box(cam, T_opt_map, (c['x'], c['y']), c['size'] / 2, cone_height)
        if b is None or b['visible_fraction'] < cfg['min_visible']:
            continue
        if world is not None and not world.line_of_sight(cam_xy[0], cam_xy[1], c['x'], c['y'], c['size'] / 2 + 0.10):
            if stats is not None:
                stats['hidden'] += 1
            continue
        u0, v0, u1, v1 = b['box']
        u0, u1 = max(u0, 0.0), min(u1, cam.width - 1.0)
        v0, v1 = max(v0, 0.0), min(v1, cam.height - 1.0)
        h = v1 - v0
        if h < cfg['min_box_px']:
            continue
        if stats is not None:
            stats['visible'] += 1
        p_det = (1.0 - cfg['p_miss']) * min(1.0, (h - cfg['min_box_px']) / max(1e-6, cfg['full_box_px'] - cfg['min_box_px']))
        if rng.random() > p_det:
            if stats is not None:
                stats['missed'] += 1
            continue
        sig = cfg['box_noise_px'] + cfg['box_noise_frac'] * max(u1 - u0, h)
        u0, v0, u1, v1 = (v + rng.normal(0.0, sig) for v in (u0, v0, u1, v1))
        u0, u1 = sorted((min(max(u0, 0.0), cam.width - 1.0), min(max(u1, 0.0), cam.width - 1.0)))
        v0, v1 = sorted((min(max(v0, 0.0), cam.height - 1.0), min(max(v1, 0.0), cam.height - 1.0)))
        if u1 - u0 < 2 or v1 - v0 < 2:
            continue
        conf = float(np.clip(0.45 + 0.5 * min(1.0, h / 80.0) + rng.normal(0.0, 0.05), 0.25, 0.99))
        out.append(('cone', conf, u0, v0, u1, v1, c['label']))
    for _ in range(rng.poisson(cfg['false_per_picture'])):           # a false box now and then
        w = rng.uniform(8.0, 40.0)
        u0, v0 = rng.uniform(0, cam.width - w), rng.uniform(cam.height * 0.25, cam.height - w)
        out.append(('cone', float(rng.uniform(0.25, 0.5)), u0, v0, u0 + w, v0 + w * rng.uniform(1.0, 2.0), None))
        if stats is not None:
            stats['false'] += 1
    return out


class FakeCamera(Node):

    def __init__(self):
        super().__init__('fake_camera')

        def p(name, value):
            return self.declare_parameter(name, value).value

        from ament_index_python.packages import get_package_share_directory
        from apex_description.mounts import load_mounts
        default_cam = os.path.join(get_package_share_directory('camera_publisher'), 'config', 'iphone_iriun_640x480.yaml')
        self.cam = pj.CameraModel.from_yaml(p('camera_info_file', default_cam) or default_cam)
        m = load_mounts(p('mounts', 'sim'))
        self.T_base_opt = pj.mount_transforms(m)['camera_optical_frame']
        self.cones = load_cones(p('cones_file', ''))
        wm = p('world_map', '')
        self.world = OccupancyMap(wm) if wm else None
        self.cone_height = float(p('cone_height', 0.25))
        self.cfg = {k: float(p(k, v)) for k, v in (('max_range_m', 8.0), ('min_visible', 0.6), ('min_box_px', 10.0),
                                                     ('full_box_px', 25.0), ('p_miss', 0.05), ('box_noise_px', 1.5),
                                                     ('box_noise_frac', 0.03), ('false_per_picture', 0.02))}
        self.fps, self.latency = float(p('fps', 15.0)), float(p('latency_s', 0.08))
        self.frame_id = p('frame_id', 'camera_optical_frame')
        self.rng = np.random.default_rng(int(p('seed', 0)))
        self.poses = deque(maxlen=2000)                               # (t, x, y, yaw) of the TRUE car
        self.create_subscription(Odometry, p('odom_topic', '/ego_racecar/odom'), self.on_odom, 50)
        self.pub = self.create_publisher(CameraDetectionArray, p('detections_topic', '/camera/detections'), 10)
        self.pub_info = self.create_publisher(CameraInfo, p('camera_info_topic', '/camera/camera_info'),
                                              qos_profile_sensor_data)
        info = CameraInfo()
        info.header.frame_id = self.frame_id
        info.width, info.height = self.cam.width, self.cam.height
        info.distortion_model = 'plumb_bob'
        info.k = [float(v) for v in self.cam.K.ravel()]
        info.d = [float(v) for v in self.cam.D]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [float(v) for v in np.hstack([self.cam.K, np.zeros((3, 1))]).ravel()]
        self.info = info
        self.seq = 0
        self.stats = dict(pictures=0, boxes=0, visible=0, hidden=0, missed=0, false=0, work=0.0)
        self.last_log = time.time()
        self.create_timer(1.0 / self.fps, self.tick)
        self.get_logger().info(f'fake camera: {len(self.cones)} cones (h {self.cone_height} m), world map {wm or "none"}, '
                               f'camera {self.cam.width} x {self.cam.height} at {self.fps:.0f} Hz, latency '
                               f'{self.latency * 1000:.0f} ms, frame {self.frame_id}, mount {m["_path"]}')

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        self.poses.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, msg.pose.pose.position.x,
                           msg.pose.pose.position.y, yaw_of(q)))

    def pose_at(self, t):
        if len(self.poses) < 2 or not (self.poses[0][0] <= t <= self.poses[-1][0]):
            return None
        a = np.array(self.poses)
        yaw = np.unwrap(a[:, 3])
        return float(np.interp(t, a[:, 0], a[:, 1])), float(np.interp(t, a[:, 0], a[:, 2])), \
            float(np.interp(t, a[:, 0], yaw))

    def tick(self):
        t0 = time.perf_counter()
        now = self.get_clock().now().nanoseconds * 1e-9
        t_pic = now - self.latency                                      # the picture whose boxes are ready now
        pose = self.pose_at(t_pic)
        if pose is None:
            return
        x, y, yaw = pose
        T_map_base = pj.T_from_xyz_rpy((x, y, 0.0), (0.0, 0.0, yaw))
        T_map_opt = T_map_base @ self.T_base_opt
        dets = fake_detections(self.cam, pj.invert(T_map_opt), T_map_opt[:2, 3], self.cones, self.cone_height,
                               self.world, self.rng, self.cfg, self.stats)
        msg = CameraDetectionArray()
        msg.header.stamp.sec = int(t_pic)
        msg.header.stamp.nanosec = int(round((t_pic - int(t_pic)) * 1e9)) % 1000000000
        msg.header.frame_id = self.frame_id
        msg.schema_version = 'apex-camera-detections/0.1'
        msg.sequence_id = self.seq
        self.seq += 1
        msg.source = 'fake_camera'
        msg.image_width, msg.image_height = self.cam.width, self.cam.height
        for cls, conf, u0, v0, u1, v1, _ in dets:
            d = CameraDetection()
            d.class_name, d.class_id, d.confidence = cls, 0, conf
            d.u_min, d.v_min, d.u_max, d.v_max = float(u0), float(v0), float(u1), float(v1)
            msg.detections.append(d)
        msg.latency_ms = float((self.get_clock().now().nanoseconds * 1e-9 - t_pic) * 1000.0)
        self.pub.publish(msg)
        self.info.header.stamp = msg.header.stamp
        self.pub_info.publish(self.info)
        s = self.stats
        s['pictures'] += 1
        s['boxes'] += len(dets)
        s['work'] += time.perf_counter() - t0
        if time.time() - self.last_log >= 5.0:
            self.get_logger().info(f"{s['pictures']} pictures, {s['boxes']} boxes ({s['false']} false); cones in view "
                                   f"{s['visible']}, missed {s['missed']}, hidden by walls {s['hidden']}; work "
                                   f"{s['work'] / max(1, s['pictures']) * 1000:.1f} ms per picture")
            self.last_log = time.time()


def main(args=None):
    rclpy.init(args=args)
    node = FakeCamera()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
