#!/usr/bin/env python3
"""Cone map node.

Takes cone detections from /lidar/cones (in the LiDAR frame, which moves with
the car), transforms them into the fixed 'map' frame with TF, and merges
repeated sightings of the same cone into one landmark.

  /lidar/cones (laser frame) --TF--> map frame --associate--> cone map
  publishes: /lidar/cones_map   (PoseArray, confirmed cones, map frame)
             /lidar/cone_map_markers (MarkerArray for Foxglove)
"""
import math


# ----------------------------------------------------------------------------
# Pure logic (no ROS here, so it can be tested on its own)
# ----------------------------------------------------------------------------

def transform_2d(x, y, tx, ty, yaw):
    """Rotate (x, y) by yaw and then shift by (tx, ty)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return tx + c * x - s * y, ty + s * x + c * y


def yaw_from_quaternion(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class ConeMap:
    """Keeps one entry per physical cone; each new detection either updates
    the nearest existing cone (if close enough) or starts a new one."""

    def __init__(self, gate):
        self.gate = gate      # m, max distance to count as the same cone
        self.cones = []       # list of dicts: x, y, n (times seen)

    def add(self, x, y):
        best, best_d = None, self.gate
        for c in self.cones:
            d = math.hypot(c['x'] - x, c['y'] - y)
            if d < best_d:
                best, best_d = c, d
        if best is None:
            self.cones.append({'x': x, 'y': y, 'n': 1})
        else:
            # Running average of all sightings of this cone
            best['n'] += 1
            best['x'] += (x - best['x']) / best['n']
            best['y'] += (y - best['y']) / best['n']

    def confirmed(self, min_hits):
        return [c for c in self.cones if c['n'] >= min_hits]


# ----------------------------------------------------------------------------
# ROS 2 node
# ----------------------------------------------------------------------------

try:
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.time import Time
    from geometry_msgs.msg import Pose, PoseArray
    from tf2_ros import Buffer, TransformListener
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:  # allows testing ConeMap without ROS installed
    Node = object


class ConeMapNode(Node):

    def __init__(self):
        super().__init__('cone_map_node')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('association_gate', 0.4)   # m
        self.declare_parameter('min_hits', 3)             # sightings before a cone is trusted
        # Known cone positions (x1, y1, x2, y2, ...) used ONLY to report accuracy.
        # Default = the cones in make_grad_track.py. Set to [] on the real track.
        self.declare_parameter('truth_cones', [7.0, 0.5, 9.5, 1.1, 13.2, 4.5,
                                               8.0, 8.2, 5.0, 7.8, 0.8, 4.5])

        self.map_frame = self.get_parameter('map_frame').value
        self.cone_map = ConeMap(self.get_parameter('association_gate').value)
        self.min_hits = self.get_parameter('min_hits').value
        t = list(self.get_parameter('truth_cones').value)
        self.truth = list(zip(t[0::2], t[1::2]))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(PoseArray, '/lidar/cones', self.cones_callback, 10)
        self.pose_pub = self.create_publisher(PoseArray, '/lidar/cones_map', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/lidar/cone_map_markers', 10)
        self.create_timer(3.0, self.report)
        self.get_logger().info(f'Mapping /lidar/cones into frame "{self.map_frame}"')

    def cones_callback(self, msg):
        if not msg.poses:
            return
        # Where was the LiDAR (in the map) at the moment this scan was taken?
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, msg.header.frame_id,
                Time.from_msg(msg.header.stamp), timeout=Duration(seconds=0.1))
        except Exception as e:  # transform not available yet
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=2.0)
            return

        tr = tf.transform.translation
        q = tf.transform.rotation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        for p in msg.poses:
            mx, my = transform_2d(p.position.x, p.position.y, tr.x, tr.y, yaw)
            self.cone_map.add(mx, my)

        self.publish(msg.header.stamp)

    def publish(self, stamp):
        cones = self.cone_map.confirmed(self.min_hits)

        pa = PoseArray()
        pa.header.frame_id = self.map_frame
        pa.header.stamp = stamp
        for c in cones:
            pose = Pose()
            pose.position.x = c['x']
            pose.position.y = c['y']
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        self.pose_pub.publish(pa)

        arr = MarkerArray()
        for i, c in enumerate(cones):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = stamp
            m.ns = 'cone_map'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = c['x']
            m.pose.position.y = c['y']
            m.pose.position.z = 0.2
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.2
            m.scale.z = 0.4
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.4, 1.0, 0.8
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def report(self):
        cones = self.cone_map.confirmed(self.min_hits)
        lines = [f'{len(cones)} confirmed cones '
                 f'({len(self.cone_map.cones)} candidates incl. unconfirmed)']
        for tx, ty in self.truth:
            if not cones:
                break
            best = min(cones, key=lambda c: math.hypot(c['x'] - tx, c['y'] - ty))
            err = math.hypot(best['x'] - tx, best['y'] - ty)
            status = f"error {err * 100:.1f} cm, seen {best['n']}x" if err < 0.5 else 'not found yet'
            lines.append(f'  truth ({tx:.1f}, {ty:.1f}) -> {status}')
        self.get_logger().info('\n'.join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = ConeMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
