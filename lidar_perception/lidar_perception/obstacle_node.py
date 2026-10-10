#!/usr/bin/env python3
"""obstacle_node.py -- LiDAR obstacle list for planning: map difference + tracking (obstacle detection step 3).

  LaserScan (scan_topic) ───────────────┐
  /state_estimate (pose, mode, flags) ──┼──> ObstacleList (obstacle_list.py, ROS-free, replay-tested) ──> /obstacles
  /map (the car's map, latched) ────────┘                                                              /obstacles/markers
For every scan: the pose at the scan's time (StateEstimate, interpolated), the safety rules (jump -> reset;
StateEstimate not OK -> frozen; too many unexplained points -> frozen), detection (points > 10 cm from every wall
of the car's map, grouped into objects) and tracking (match, average, confirm after 2, remember, forget only when
the LiDAR looks through the place). Step 3.1: the tracker is told how sure the pose is (std_xy, std_psi) --
detections made with an unsure pose count less and make the reported std_xy_m larger. Step 3.2 (align_scans): before
detection each scan is put on the walls of the car's map (scan_align.py), which removes most of StateEstimate's pose
error at that moment (sim: across the car p95 3.3 -> 1.3 cm, heading 0.57 -> 0.32 deg) -- obstacle positions get more
accurate and the alignment's own, calibrated uncertainty replaces StateEstimate's.
Publishes apex_msgs/ObstacleArray (draft schema "apex-obstacles/0.1") with
the CONFIRMED obstacles in the map frame and in track coordinates (apex_track), once per scan, stamped with the
scan's time; plus a MarkerArray for Foxglove.

A scan waits until a StateEstimate message at or after its stamp has arrived (normally a few ms), so the pose is
interpolated, not guessed; after WAIT_FOR_POSE_S it is processed with up to 50 ms of extrapolation, or dropped.

Parameters
  scan_topic      /scan_a1 (sim, 7 Hz RPLIDAR model) -- real car: the rplidar driver's topic
  track / track_yaml   apex_track track for s / e_y (as state_estimate)
  laser_x, laser_y, laser_yaw   the LiDAR's position relative to the StateEstimate point (the CG) -- sim
                  0.275 / 0 / 0 (ref_to_cg_m = 0). REAL CAR: measure (base_link -> laser minus base_link -> CG).
  publish_markers  true: /obstacles/markers for Foxglove
  align_scans     true: align every scan to the map's walls before detection (step 3.2); false = step 3.1
  stats_every_s   log line every ... s (scans, statuses, list size, work per scan, latency, alignment, CPU)
"""
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from apex_msgs.msg import Obstacle, ObstacleArray, StateEstimate
from apex_track import load_track
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from lidar_perception import obstacle_list as ol
from lidar_perception.map_difference import CONE_RADIUS, MAX_RANGE, TOL
from lidar_perception.scan_align import ScanAligner
from lidar_perception.wall_distance import WallDistance

SCHEMA = 'apex-obstacles/0.1'
WAIT_FOR_POSE_S = 0.1


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def to_us(t_s):
    return int(round(t_s * 1e6))


class ObstacleNode(Node):

    def __init__(self):
        super().__init__('obstacle_node')
        d = self.declare_parameter
        d('scan_topic', '/scan_a1')
        d('track', 'levine')
        d('track_yaml', '')
        d('laser_x', 0.275)
        d('laser_y', 0.0)
        d('laser_yaw', 0.0)
        d('publish_markers', True)
        d('align_scans', True)
        d('stats_every_s', 5.0)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.mount = (float(p('laser_x')), float(p('laser_y')), float(p('laser_yaw')))
        path = p('track_yaml') or (get_package_share_directory('apex_track') + f"/tracks/{p('track')}/track.yaml")
        self.track = load_track(path, with_boundaries=False)
        self.builder = None                  # ObstacleList, made when /map arrives
        self.pending = None                  # (scan message, wall time it arrived)
        self.seq = 0
        self.align = bool(p('align_scans'))
        self.stats = self.new_stats()
        self.last_stats = time.monotonic()
        self.last_cpu = time.process_time()
        self.known_ids = set()

        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, latched)
        self.create_subscription(StateEstimate, '/state_estimate', self.se_cb, 50)
        self.create_subscription(LaserScan, p('scan_topic'), self.scan_cb, qos_profile_sensor_data)
        self.pub = self.create_publisher(ObstacleArray, '/obstacles', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/obstacles/markers', 10) if p('publish_markers') else None
        self.get_logger().info(
            f'track {self.track.track_id} ({self.track.track_hash}); scans {p("scan_topic")}; LiDAR at '
            f'{self.mount} from the StateEstimate point; tolerance {TOL * 100:.0f} cm, range {MAX_RANGE:.0f} m; '
            f'scan alignment {"ON" if self.align else "OFF"}; waiting for /map')

    @staticmethod
    def new_stats():
        return dict(scans=0, nopose=0, dropped=0, status={}, work=[], latency=[], share=[], pose_n=0, pose_s=0.0,
                    al_ok=0, al_no=0, al_shift=[], al_turn=[], al_us=[])

    # ------------------------------------------------------------------ inputs
    def map_cb(self, msg):
        info = msg.info
        grid = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        t0 = time.perf_counter()
        wd = WallDistance(grid, info.resolution, info.origin.position.x, info.origin.position.y)
        aligner = ScanAligner(wd, grid) if self.align else None
        self.builder = ol.ObstacleList(wd, self.mount, aligner)   # a new map = a new list
        self.get_logger().info(f'/map {info.width} x {info.height} @ {info.resolution:.3f} m: distance-to-wall grid'
                               + (' + signed distance field for the alignment' if aligner else '')
                               + f' built in {(time.perf_counter() - t0) * 1000:.0f} ms ({int(wd.wall.sum())} wall cells)')

    def se_cb(self, m):
        if self.builder is None:
            return
        t0 = time.perf_counter()
        self.builder.add_pose(stamp_s(m.header.stamp), m.x_m, m.y_m, m.psi_rad, m.vx_mps, m.r_radps,
                              ok=m.mode == StateEstimate.MODE_OK,
                              reset_flag=bool(m.flags & StateEstimate.FLAG_FILTER_RESET),
                              std_xy=m.std_xy_m, std_psi=m.std_psi_rad)
        self.stats['pose_n'] += 1
        self.stats['pose_s'] += time.perf_counter() - t0
        self.try_process()

    def scan_cb(self, msg):
        if self.pending is not None:                           # an older scan never got a pose: process or drop it
            self.try_process(force=True)
            if self.pending is not None:
                self.stats['dropped'] += 1
        self.pending = (msg, time.monotonic())
        self.try_process()

    # ------------------------------------------------------------------ work
    def try_process(self, force=False):
        if self.pending is None or self.builder is None:
            return
        msg, arrived = self.pending
        t = stamp_s(msg.header.stamp)
        newest = self.builder.newest_pose_t()
        if not force and (newest is None or newest < t) and time.monotonic() - arrived < WAIT_FOR_POSE_S:
            return                                             # wait for a pose at or after the scan's time
        self.pending = None
        t_work = time.perf_counter()
        res = self.builder.process_scan(t, np.asarray(msg.ranges, dtype=float), msg.angle_min, msg.angle_increment,
                                        msg.range_min, msg.range_max)
        self.stats['scans'] += 1
        if res is None:
            self.stats['nopose'] += 1
            return
        out = self.make_message(msg.header.stamp, t, res)
        work_us = (time.perf_counter() - t_work) * 1e6
        out.publish_us = to_us(self.get_clock().now().nanoseconds * 1e-9)
        self.pub.publish(out)
        if self.marker_pub is not None:
            self.marker_pub.publish(self.make_markers(msg.header.stamp, res))
        self.bookkeeping(res, work_us, (out.publish_us - out.stamp_us) / 1000.0)

    def make_message(self, stamp, t, res):
        m = ObstacleArray()
        m.header.stamp = stamp
        m.header.frame_id = 'map'
        m.schema_version = SCHEMA
        m.sequence_id = self.seq
        self.seq += 1
        m.stamp_us = to_us(t)
        m.track_id, m.track_hash = self.track.track_id, self.track.track_hash
        m.status = res['status']
        m.list_epoch = res['epoch']
        m.max_range_m, m.blind_strip_m = float(MAX_RANGE), float(TOL)
        for tr in res['obstacles']:
            o = Obstacle()
            o.id = tr.id
            o.x_m, o.y_m = float(tr.x), float(tr.y)
            o.radius_m = float(max(tr.size / 2.0, CONE_RADIUS))
            o.std_xy_m = float(tr.sigma)
            self.track.reset()                                 # every obstacle on its own (global search)
            _, s, e_y, _, _ = self.track.project(tr.x, tr.y, 0.0)
            o.s_track_m, o.e_y_m = float(s), float(e_y)
            o.detections = int(tr.hits)
            o.first_seen_us, o.last_seen_us = to_us(tr.first_t), to_us(tr.last_t)
            o.seen_now = res['seen_now'][tr.id]
            m.obstacles.append(o)
        return m

    def make_markers(self, stamp, res):
        arr = MarkerArray()
        clear = Marker()
        clear.header.stamp, clear.header.frame_id, clear.action = stamp, 'map', Marker.DELETEALL
        arr.markers.append(clear)
        for k, tr in enumerate(res['obstacles']):
            seen = res['seen_now'][tr.id]
            for j, kind in enumerate((Marker.CYLINDER, Marker.TEXT_VIEW_FACING)):
                mk = Marker()
                mk.header.stamp, mk.header.frame_id = stamp, 'map'
                mk.ns, mk.id, mk.type = 'obstacles', 2 * k + j, kind
                mk.pose.position.x, mk.pose.position.y = float(tr.x), float(tr.y)
                mk.pose.orientation.w = 1.0
                if kind == Marker.CYLINDER:
                    r = max(tr.size / 2.0, CONE_RADIUS)
                    mk.scale.x = mk.scale.y = 2 * r
                    mk.scale.z, mk.pose.position.z = 0.3, 0.15
                    mk.color.r, mk.color.g, mk.color.b = (0.92, 0.41, 0.20) if seen else (0.55, 0.55, 0.52)
                    mk.color.a = 0.9
                else:
                    mk.text = f'id {tr.id}'
                    mk.pose.position.z = 0.45
                    mk.scale.z = 0.15
                    mk.color.r = mk.color.g = mk.color.b = mk.color.a = 1.0
                arr.markers.append(mk)
        return arr

    def bookkeeping(self, res, work_us, latency_ms):
        st = self.stats
        name = ol.STATUS_NAMES[res['status']]
        st['status'][name] = st['status'].get(name, 0) + 1
        st['work'].append(work_us)
        st['latency'].append(latency_ms)
        st['share'].append(res['new_share'])
        if res['aligned']:
            st['al_ok'] += 1
            st['al_shift'].append(res['correction'][0])
            st['al_turn'].append(res['correction'][1])
            st['al_us'].append(res['align_us'])
        elif self.align and res['status'] != ol.STATUS_FROZEN_LOCALIZATION:
            st['al_no'] += 1
        ids = {tr.id for tr in res['obstacles']}
        for tid in sorted(ids - self.known_ids):
            tr = next(t for t in res['obstacles'] if t.id == tid)
            self.get_logger().info(f'obstacle id {tid} CONFIRMED at ({tr.x:.2f}, {tr.y:.2f})')
        for tid in sorted(self.known_ids - ids):
            self.get_logger().info(f'obstacle id {tid} left the list'
                                   + (' (list reset after a pose jump)' if res['status'] == ol.STATUS_RESET else ''))
        self.known_ids = ids
        if res['status'] == ol.STATUS_RESET:
            self.get_logger().warn(f'pose jump -> obstacle list emptied (epoch {res["epoch"]})')
        now = time.monotonic()
        if now - self.last_stats >= self.get_parameter('stats_every_s').value and st['work']:
            w, lat = np.array(st['work']), np.array(st['latency'])
            cpu = (time.process_time() - self.last_cpu) / (now - self.last_stats) * 100.0
            al = (f'aligned {st["al_ok"]} (refused {st["al_no"]}), correction median {np.median(st["al_shift"]) * 100:.1f} cm '
                  f'/ {np.degrees(np.median(st["al_turn"])):.2f} deg, max {max(st["al_shift"]) * 100:.1f} cm, '
                  f'{np.median(st["al_us"]):.0f} us each; ' if st['al_ok'] else
                  (f'aligned 0 (refused {st["al_no"]}); ' if self.align else ''))
            self.get_logger().info(
                f'{st["scans"]} scans ({st["nopose"]} without pose, {st["dropped"]} dropped): '
                + ', '.join(f'{k} {v}' for k, v in sorted(st['status'].items()))
                + f'; list {len(ids)}; work per scan mean {w.mean():.0f} us, max {w.max():.0f} us; latency '
                f'mean {lat.mean():.1f} ms, max {lat.max():.1f} ms; new points max {max(st["share"]) * 100:.1f} %; '
                f'{st["pose_n"]} poses at {st["pose_s"] / max(st["pose_n"], 1) * 1e6:.0f} us each; '
                + al + f'node CPU {cpu:.1f} % of one core')
            self.stats = self.new_stats()
            self.last_stats = now
            self.last_cpu = time.process_time()


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
