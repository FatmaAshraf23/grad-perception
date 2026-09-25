#!/usr/bin/env python3
"""2D LiDAR clustering node.

LaserScan -> valid points -> ROI -> x,y -> DBSCAN -> cluster features -> markers
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray


class ScanClusterNode(Node):

    def __init__(self):
        super().__init__('scan_cluster_node')

        # Parameters (can be changed at run time with --ros-args -p name:=value)
        self.declare_parameter('scan_topic', '/scan_a1')
        self.declare_parameter('roi_max_range', 6.0)      # m, ignore anything further
        self.declare_parameter('roi_fov_deg', 360.0)      # keep this much of the field of view
        self.declare_parameter('dbscan_eps', 0.15)        # m, max gap between neighbours in a cluster
        self.declare_parameter('dbscan_min_samples', 3)   # points needed to form a cluster

        topic = self.get_parameter('scan_topic').value
        self.sub = self.create_subscription(
            LaserScan, topic, self.scan_callback, qos_profile_sensor_data)
        self.marker_pub = self.create_publisher(MarkerArray, '/lidar/clusters', 10)
        self.get_logger().info(f'Listening to {topic}, publishing /lidar/clusters')

    def scan_callback(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # 1. Keep only valid readings (no inf/nan, inside the sensor's range)
        valid = (np.isfinite(ranges)
                 & (ranges >= msg.range_min)
                 & (ranges <= msg.range_max))

        # 2. Region of interest: close enough and inside the chosen field of view
        max_r = self.get_parameter('roi_max_range').value
        half_fov = math.radians(self.get_parameter('roi_fov_deg').value) / 2.0
        roi = valid & (ranges <= max_r) & (np.abs(angles) <= half_fov)

        r = ranges[roi]
        a = angles[roi]

        # 3. Polar -> Cartesian in the LiDAR frame (x forward, y left)
        xy = np.column_stack((r * np.cos(a), r * np.sin(a)))

        clusters = []
        if len(xy) > 0:
            # 4. DBSCAN clustering on x,y (same idea as your KITTI script, now 2D)
            labels = DBSCAN(
                eps=self.get_parameter('dbscan_eps').value,
                min_samples=self.get_parameter('dbscan_min_samples').value,
            ).fit_predict(xy)

            # 5. Features for each cluster
            for cid in sorted(set(labels) - {-1}):
                pts = xy[labels == cid]
                cx, cy = pts.mean(axis=0)
                size = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
                clusters.append({
                    'id': int(cid),
                    'n': len(pts),
                    'x': float(cx),
                    'y': float(cy),
                    'dist': float(math.hypot(cx, cy)),
                    'bearing': math.degrees(math.atan2(cy, cx)),
                    'size': size,
                })

        self.publish_markers(msg.header, clusters)

        # Print a short summary at most every 2 s
        clusters.sort(key=lambda c: c['dist'])
        summary = ' | '.join(
            f"#{c['id']}: {c['dist']:.2f} m @ {c['bearing']:+.0f} deg, "
            f"size {c['size']:.2f} m, {c['n']} pts"
            for c in clusters[:5])
        self.get_logger().info(
            f'{len(xy)} pts in ROI, {len(clusters)} clusters. Nearest: {summary}',
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
            m.ns = 'clusters'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = c['x']
            m.pose.position.y = c['y']
            m.pose.orientation.w = 1.0
            d = max(c['size'], 0.05)
            m.scale.x = d
            m.scale.y = d
            m.scale.z = 0.2
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 0.6
            arr.markers.append(m)

            t = Marker()
            t.header = header
            t.ns = 'labels'
            t.id = i
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = c['x']
            t.pose.position.y = c['y']
            t.pose.position.z = 0.3
            t.pose.orientation.w = 1.0
            t.scale.z = 0.15
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = f"{c['dist']:.1f}m"
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
