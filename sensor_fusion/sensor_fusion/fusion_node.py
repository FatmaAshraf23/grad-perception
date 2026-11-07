#!/usr/bin/env python3
"""fusion_node -- camera confirmation of the LiDAR obstacle list (camera-LiDAR fusion v1, 2026-11-07).

  /obstacles (apex_msgs/ObstacleArray, LiDAR, one per scan) ────────┐
  /camera/detections (CameraDetectionArray, picture-time stamps) ──┼──> fusion_core.CameraFusion ──> /obstacles/fused
  /state_estimate (pose + std at every 10 ms) ─────────────────────┤                                 /obstacles/fused/markers
  /camera/camera_info (or camera_info_file) + the camera mount ────┘

/obstacles/fused = EVERY obstacle of the newest /obstacles message, unchanged, + camera fields (camera_state UNKNOWN /
CONFIRMED / NO_CONE, class_name, camera_confidence, camera_looks, camera_hits, camera_last_hit_us) + the list's
camera_status (OFF / OK / STALE) and camera_age_ms; schema "apex-obstacles/0.2". Published right after each
/obstacles message (same stamp = the scan's time), so planning gets the same rate and latency + < 1 ms. The camera
never removes or adds an obstacle (decisions 2026-11-07: one list, never drop, camera-only warnings later).

A picture is processed once StateEstimate has a state at or after its time (stamp + camera_delay_s); after
wait_for_pose_s it is dropped. All matching rules: fusion_core.py.

Parameters
  mounts           apex_description mount file: 'sim' (simulator), 'car', or a path -- the camera's position on the car
                   (base_link = the StateEstimate point; the SAME file fake_camera / sensor_frames use)
  camera_info_file a camera_info YAML; '' = use the first /camera/camera_info message
  camera_delay_s   added to every picture stamp [s]: the stamp offset latency_probe measured (sim 0: exact stamps)
  camera_stale_s   no picture used for this long -> camera_status STALE (0.5)
  wait_for_pose_s  a picture waits at most this long for a StateEstimate at/after its time (0.2)
  + every fusion_core.CameraFusion.DEFAULTS entry as a parameter (gate_k, size_ratio, miss_limit, ...)
REAL CAR: mounts 'car' (sensor_mounts_car.yaml filled from calib_solve), camera_info_file = the car camera's
calibration, camera_delay_s from latency_probe, sigma_mount_deg from calib_solve's spread.
"""
import time
from collections import deque

import rclpy
from apex_msgs.msg import CameraDetectionArray, Obstacle, ObstacleArray, StateEstimate
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray

from sensor_fusion import fusion_core as fc
from sensor_fusion import projection as pj

SCHEMA = 'apex-obstacles/0.2'


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class FusionNode(Node):

    def __init__(self):
        super().__init__('fusion_node')

        def p(name, value):
            return self.declare_parameter(name, value).value

        from apex_description.mounts import load_mounts
        m = load_mounts(p('mounts', 'sim'))
        self.T_base_opt = pj.mount_transforms(m)['camera_optical_frame']
        self.mount_path = m.get('_path', '?')
        info_file = p('camera_info_file', '')
        self.cfg = {}
        for k, v in fc.CameraFusion.DEFAULTS.items():
            val = p(k, float(v) if isinstance(v, float) else int(v))
            self.cfg[k] = val
        self.stale_s = float(p('camera_stale_s', 0.5))
        self.wait_s = float(p('wait_for_pose_s', 0.2))
        self.stats_every = float(p('stats_every_s', 5.0))
        self.fusion = None
        if info_file:
            self.make_fusion(pj.CameraModel.from_yaml(info_file), info_file)
        self.poses = fc.PoseHistory(keep_s=2.0)
        self.obstacles = []                    # the newest LiDAR list as fusion_core dicts
        self.last_obs_msg = None
        self.pending = deque(maxlen=30)        # (detections message, wall time it arrived)
        self.last_used_stamp = None            # stamp [s] of the newest USED picture (ROS time)
        self.counts = dict(wait_dropped=0, old=0, not_ok=0, used=0, fused_msgs=0, work_us=0.0, frame_warned=False)
        self.last_stats = time.monotonic()
        self.last_cpu = time.process_time()
        self.last_states = {}

        self.create_subscription(StateEstimate, '/state_estimate', self.se_cb, 50)
        self.create_subscription(ObstacleArray, p('obstacles_topic', '/obstacles'), self.obs_cb, 10)
        self.create_subscription(CameraDetectionArray, p('detections_topic', '/camera/detections'), self.det_cb, 10)
        if not info_file:
            self.create_subscription(CameraInfo, p('camera_info_topic', '/camera/camera_info'), self.info_cb,
                                     qos_profile_sensor_data)
        self.pub = self.create_publisher(ObstacleArray, p('fused_topic', '/obstacles/fused'), 10)
        self.marker_pub = (self.create_publisher(MarkerArray, '/obstacles/fused/markers', 10)
                           if p('publish_markers', True) else None)
        self.get_logger().info(
            f"camera fusion v1: /obstacles + /camera/detections -> /obstacles/fused; camera mount {self.mount_path}; "
            f"camera_delay_s {self.cfg['camera_delay_s']:.3f}; gate {self.cfg['gate_k']:.1f} sigma, size x"
            f"{self.cfg['size_ratio']:.2f}; CONFIRMED after {self.cfg['confirm_hits']} matches, NO_CONE after "
            f"{self.cfg['miss_limit']} looks without one; " + ('camera model from ' + info_file if info_file
                                                                 else 'waiting for /camera/camera_info'))

    def make_fusion(self, cam, src):
        self.fusion = fc.CameraFusion(cam, self.T_base_opt, **self.cfg)
        self.get_logger().info(f'camera model {cam.width} x {cam.height}, fx {cam.fx:.1f} fy {cam.fy:.1f} ({src})')

    # ------------------------------------------------------------------ inputs
    def info_cb(self, msg):
        if self.fusion is None:
            self.make_fusion(pj.CameraModel.from_camera_info(msg), 'camera_info topic')

    def se_cb(self, m):
        self.poses.add(stamp_s(m.header.stamp), m.x_m, m.y_m, m.psi_rad, m.std_xy_m, m.std_psi_rad, m.r_radps,
                       1 if m.mode == StateEstimate.MODE_OK else 0)
        self.try_pictures()

    def det_cb(self, msg):
        if not self.counts['frame_warned'] and not msg.header.frame_id.endswith('camera_optical_frame'):
            self.get_logger().warn(f"detections in frame '{msg.header.frame_id}': expected a camera_optical_frame -- "
                                   'the fusion uses the camera mount from the mounts file, not this frame')
            self.counts['frame_warned'] = True
        self.pending.append((msg, time.monotonic()))
        self.try_pictures()

    def obs_cb(self, msg):
        self.last_obs_msg = msg
        self.obstacles = [dict(id=o.id, x=o.x_m, y=o.y_m, radius=o.radius_m, std_xy=o.std_xy_m,
                               last_seen_s=o.last_seen_us * 1e-6) for o in msg.obstacles]
        if self.fusion is not None:
            self.fusion.forget_except([o.id for o in msg.obstacles])
        self.publish_fused(msg)

    # ------------------------------------------------------------------ work
    def try_pictures(self):
        if self.fusion is None:
            return
        while self.pending:
            msg, arrived = self.pending[0]
            t0 = time.perf_counter()
            boxes = [(d.class_name, d.confidence, d.u_min, d.v_min, d.u_max, d.v_max) for d in msg.detections]
            r = self.fusion.picture(stamp_s(msg.header.stamp), msg.image_width, msg.image_height, boxes, self.obstacles,
                                    self.poses)
            if r == 'wait':
                if time.monotonic() - arrived < self.wait_s:
                    return                                   # keep it (and every newer one) for the next state
                self.counts['wait_dropped'] += 1
            elif r == 'old':
                self.counts['old'] += 1
            elif r == 'not_ok':
                self.counts['not_ok'] += 1
            else:
                self.counts['used'] += 1
                self.last_used_stamp = stamp_s(msg.header.stamp)
            self.counts['work_us'] += (time.perf_counter() - t0) * 1e6
            self.pending.popleft()

    def publish_fused(self, src):
        out = ObstacleArray()
        out.header = src.header
        out.schema_version = SCHEMA
        out.sequence_id = src.sequence_id
        out.stamp_us = src.stamp_us
        out.track_id, out.track_hash = src.track_id, src.track_hash
        out.status, out.list_epoch = src.status, src.list_epoch
        out.max_range_m, out.blind_strip_m = src.max_range_m, src.blind_strip_m
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.fusion is None:
            out.camera_status, out.camera_age_ms = ObstacleArray.CAMERA_OFF, -1.0
        elif self.last_used_stamp is None:
            out.camera_status, out.camera_age_ms = ObstacleArray.CAMERA_STALE, -1.0
        else:
            age = now - self.last_used_stamp
            out.camera_age_ms = float(age * 1000.0)
            out.camera_status = ObstacleArray.CAMERA_OK if age <= self.stale_s else ObstacleArray.CAMERA_STALE
        states = {}
        for o in src.obstacles:
            f = Obstacle()
            for k in ('id', 'x_m', 'y_m', 'radius_m', 'std_xy_m', 's_track_m', 'e_y_m', 'detections', 'first_seen_us',
                      'last_seen_us', 'seen_now'):
                setattr(f, k, getattr(o, k))
            if self.fusion is not None:
                st = self.fusion.state_of(o.id)
                f.camera_state = st['state']
                f.class_name = st['class_name']
                f.camera_confidence = float(st['confidence'])
                f.camera_looks, f.camera_hits = int(st['looks']), int(st['hits'])
                f.camera_last_hit_us = int(st['last_hit_us'])
                states[o.id] = st['state']
            out.obstacles.append(f)
        out.publish_us = int(round(self.get_clock().now().nanoseconds * 1e-3))
        self.pub.publish(out)
        self.counts['fused_msgs'] += 1
        if self.marker_pub is not None:
            self.marker_pub.publish(self.markers(out))
        for oid, st in states.items():
            if self.last_states.get(oid) != st:
                self.get_logger().info(f'obstacle id {oid}: camera {fc.STATE_NAMES[st]}')
        self.last_states = states
        self.maybe_report(out)

    def markers(self, msg):
        arr = MarkerArray()
        clear = Marker()
        clear.header, clear.action = msg.header, Marker.DELETEALL
        arr.markers.append(clear)
        colours = {Obstacle.CAMERA_UNKNOWN: (0.6, 0.6, 0.6), Obstacle.CAMERA_CONFIRMED: (0.1, 0.8, 0.3),
                   Obstacle.CAMERA_NO_CONE: (0.9, 0.2, 0.2)}
        for k, o in enumerate(msg.obstacles):
            for j, kind in enumerate((Marker.CYLINDER, Marker.TEXT_VIEW_FACING)):
                mk = Marker()
                mk.header = msg.header
                mk.ns, mk.id, mk.type = 'fused', 2 * k + j, kind
                mk.pose.position.x, mk.pose.position.y = o.x_m, o.y_m
                mk.pose.orientation.w = 1.0
                if kind == Marker.CYLINDER:
                    mk.scale.x = mk.scale.y = 2 * o.radius_m
                    mk.scale.z, mk.pose.position.z = 0.3, 0.15
                    mk.color.r, mk.color.g, mk.color.b = colours.get(o.camera_state, (1.0, 1.0, 1.0))
                    mk.color.a = 0.9
                else:
                    mk.text = f'id {o.id} {fc.STATE_NAMES.get(o.camera_state, "?")} {o.camera_hits}/{o.camera_looks}'
                    mk.pose.position.z = 0.5
                    mk.scale.z = 0.12
                    mk.color.r = mk.color.g = mk.color.b = mk.color.a = 1.0
                arr.markers.append(mk)
        return arr

    def maybe_report(self, out):
        now = time.monotonic()
        if now - self.last_stats < self.stats_every:
            return
        c = self.counts
        cpu = (time.process_time() - self.last_cpu) / (now - self.last_stats) * 100.0
        n = {s: sum(1 for o in out.obstacles if o.camera_state == s) for s in fc.STATE_NAMES}
        core = self.fusion.report() if self.fusion is not None else 'no camera model yet'
        npics = c['used'] + c['not_ok'] + c['old'] + c['wait_dropped']
        self.get_logger().info(
            f"{c['fused_msgs']} fused lists; list now {len(out.obstacles)} (CONFIRMED {n[fc.CONFIRMED]}, NO_CONE "
            f"{n[fc.NO_CONE]}, UNKNOWN {n[fc.UNKNOWN]}), camera age {out.camera_age_ms:.0f} ms; pictures: used "
            f"{c['used']}, StateEstimate not OK {c['not_ok']}, too old {c['old']}, no pose in time {c['wait_dropped']}; "
            f"work {c['work_us'] / max(1, npics):.0f} us per picture; {core}; node CPU {cpu:.1f} % of one core")
        for k in ('wait_dropped', 'old', 'not_ok', 'used', 'fused_msgs'):
            c[k] = 0
        c['work_us'] = 0.0
        self.last_stats = now
        self.last_cpu = time.process_time()


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
