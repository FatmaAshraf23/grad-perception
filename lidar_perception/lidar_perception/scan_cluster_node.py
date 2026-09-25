#!/usr/bin/env python3
"""2D LiDAR clustering + cone detection node.

LaserScan -> valid points -> ROI -> x,y -> DBSCAN -> cluster features
          -> cone test (small + isolated foreground object) -> markers + /lidar/cones
"""
import math

import numpy as np
from sklearn.cluster import DBSCAN


# ----------------------------------------------------------------------------
# Pure processing (no ROS here, so it can be tested on its own)
# ----------------------------------------------------------------------------

def is_isolated_run(beam_idx, ranges, valid, jump):
    """True if the cluster is ONE unbroken run of beams and the beam just
    outside it on each side sees something clearly further away (or nothing).

    A cone standing in the open passes: the beams beside it hit the far wall.
    A piece of a wall fails: the beams beside it hit the same wall, about the
    same distance away.
    """
    n = len(ranges)
    beams = set(int(i) for i in beam_idx)
    starts = [i for i in beams if (i - 1) % n not in beams]
    ends = [i for i in beams if (i + 1) % n not in beams]
    if len(starts) != 1 or len(ends) != 1:
        return False
    nearest = float(np.min(ranges[beam_idx]))
    left = (starts[0] - 1) % n
    right = (ends[0] + 1) % n

    def clear(i):
        return (not valid[i]) or ranges[i] > nearest + jump

    return clear(left) and clear(right)


def process_scan(ranges, angle_min, angle_increment, range_min, range_max, p):
    """Return a list of cluster dicts for one scan. p is a dict of parameters."""
    ranges = np.asarray(ranges, dtype=np.float32)
    angles = angle_min + np.arange(len(ranges)) * angle_increment

    # 1. Valid readings only
    valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)

    # 2. Region of interest
    half_fov = math.radians(p['roi_fov_deg']) / 2.0
    roi = valid & (ranges <= p['roi_max_range']) & (np.abs(angles) <= half_fov)
    idx = np.nonzero(roi)[0]
    if len(idx) == 0:
        return []

    # 3. Polar -> Cartesian (LiDAR frame: x forward, y left)
    r = ranges[idx]
    a = angles[idx]
    xy = np.column_stack((r * np.cos(a), r * np.sin(a)))

    # 4. DBSCAN
    labels = DBSCAN(eps=p['dbscan_eps'],
                    min_samples=p['dbscan_min_samples']).fit_predict(xy)

    clusters = []
    for cid in sorted(set(labels) - {-1}):
        mask = labels == cid
        pts = xy[mask]
        beam_idx = idx[mask]
        cx, cy = pts.mean(axis=0)
        dist = float(math.hypot(cx, cy))
        size = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

        # 5. Cone test: small enough, enough points, isolated foreground object
        cone = (size <= p['cone_max_size']
                and len(pts) >= p['cone_min_points']
                and is_isolated_run(beam_idx, ranges, valid, p['cone_min_jump']))

        # The LiDAR only sees the front face of a cone, so the centroid is
        # about one radius too close. Push it back along the ray.
        if cone and dist > 0:
            scale = (dist + p['cone_radius']) / dist
            cx, cy, dist = cx * scale, cy * scale, dist + p['cone_radius']

        clusters.append({
            'id': int(cid),
            'n': int(len(pts)),
            'x': float(cx),
            'y': float(cy),
            'dist': dist,
            'bearing': math.degrees(math.atan2(cy, cx)),
            'size': size,
            'cone': bool(cone),
            'points': pts,
        })
    return clusters


# ----------------------------------------------------------------------------
# ROS 2 node
# ----------------------------------------------------------------------------

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import Point, Pose, PoseArray
    from sensor_msgs.msg import LaserScan
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:  # allows testing process_scan() without ROS installed
    Node = object


PARAMS = {
    'roi_max_range': 8.0,        # m
    'roi_fov_deg': 360.0,        # deg
    'dbscan_eps': 0.15,          # m, max neighbour gap inside a cluster
    'dbscan_min_samples': 2,     # points to form a cluster
    'cone_max_size': 0.25,       # m, bigger clusters are not cones
    'cone_min_points': 2,        # fewer points -> not trusted as a cone
    'cone_min_jump': 0.25,       # m, background must be this much further
    'cone_radius': 0.075,        # m, real cone radius (15 cm cone)
}


class ScanClusterNode(Node):

    def __init__(self):
        super().__init__('scan_cluster_node')
        self.declare_parameter('scan_topic', '/scan_a1')
        for name, default in PARAMS.items():
            self.declare_parameter(name, default)

        topic = self.get_parameter('scan_topic').value
        self.sub = self.create_subscription(
            LaserScan, topic, self.scan_callback, qos_profile_sensor_data)
        self.marker_pub = self.create_publisher(MarkerArray, '/lidar/clusters', 10)
        self.cone_pub = self.create_publisher(PoseArray, '/lidar/cones', 10)
        self.get_logger().info(
            f'Listening to {topic}, publishing /lidar/clusters and /lidar/cones')

    def params(self):
        return {name: self.get_parameter(name).value for name in PARAMS}

    def scan_callback(self, msg):
        clusters = process_scan(msg.ranges, msg.angle_min, msg.angle_increment,
                                msg.range_min, msg.range_max, self.params())
        cones = sorted([c for c in clusters if c['cone']], key=lambda c: c['dist'])

        # Cone positions for other nodes (planning, fusion) in the LiDAR frame
        pa = PoseArray()
        pa.header = msg.header
        for c in cones:
            pose = Pose()
            pose.position.x = c['x']
            pose.position.y = c['y']
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        self.cone_pub.publish(pa)

        self.publish_markers(msg.header, clusters)

        text = ' | '.join(
            f"{c['dist']:.2f} m @ {c['bearing']:+.0f} deg ({c['n']} pts, {c['size']:.2f} m)"
            for c in cones)
        self.get_logger().info(
            f'{len(clusters)} clusters, {len(cones)} cones: {text}',
            throttle_duration_sec=2.0)

    def publish_markers(self, header, clusters):
        arr = MarkerArray()
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        for i, c in enumerate(clusters):
            m = Marker()
            m.header = header
            m.ns = 'cones' if c['cone'] else 'clusters'
            m.id = i
            m.pose.orientation.w = 1.0
            if c['cone']:
                # Cone: a tall green cylinder at the estimated cone centre
                m.type = Marker.CYLINDER
                m.pose.position.x = c['x']
                m.pose.position.y = c['y']
                m.scale.x = m.scale.y = 0.15
                m.scale.z = 0.3
                m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 0.9
            else:
                # Anything else (walls): draw the cluster's actual points.
                # A centroid + circle is meaningless for a long or L-shaped wall.
                m.type = Marker.POINTS
                m.scale.x = m.scale.y = 0.05
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 0.8
                for px, py in c['points']:
                    pt = Point()
                    pt.x, pt.y = float(px), float(py)
                    m.points.append(pt)
            arr.markers.append(m)

            if c['cone']:
                t = Marker()
                t.header = header
                t.ns = 'cone_labels'
                t.id = i
                t.type = Marker.TEXT_VIEW_FACING
                t.action = Marker.ADD
                t.pose.position.x = c['x']
                t.pose.position.y = c['y']
                t.pose.position.z = 0.45
                t.pose.orientation.w = 1.0
                t.scale.z = 0.15
                t.color.r = t.color.g = t.color.b = t.color.a = 1.0
                t.text = f"{c['dist']:.2f}m"
                arr.markers.append(t)

        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ScanClusterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
