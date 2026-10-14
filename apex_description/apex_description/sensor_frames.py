"""sensor_frames -- static TFs: where the LiDAR and the camera sit on the car, from ONE YAML (config/).

  <prefix>base_link   -> <prefix>laser                 (publish_laser; in the simulator the simulator and
                                                         sim_scan_relay publish the LiDAR frames themselves)
  <prefix>base_link   -> <prefix>camera_link           the camera's position + tilt (car-like axes)
  <prefix>camera_link -> <prefix>camera_optical_frame  the fixed turn to the camera convention
                                                         (z = where it looks, x = right, y = down in the picture)
Parameters:
  mounts_file    'car', 'sim' or a path -- REQUIRED (the simulator's numbers on the real car would put every
                 fused detection in the wrong place, so there is no silent default)
  prefixes       comma separated: '' on the car; 'ego_racecar/,est/' in the simulator (true + estimated car)
  publish_laser  default true
A YAML with status: placeholder starts with a WARNING: those numbers are not measured.
"""
import sys

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster

from apex_description.mounts import frame_list, load_mounts


class SensorFrames(Node):

    def __init__(self):
        super().__init__('sensor_frames')
        name = str(self.declare_parameter('mounts_file', '').value).strip()
        prefixes = [p.strip() for p in str(self.declare_parameter('prefixes', '').value).split(',')]
        laser = bool(self.declare_parameter('publish_laser', True).value)
        if not name:
            raise ValueError("parameter mounts_file not set: 'car' on the real car, 'sim' in the simulator, or a path")
        m = load_mounts(name)
        self.get_logger().info(f"mounts {m['_path']} (status {m['status']}), prefixes {prefixes}, "
                               f"publish_laser {laser}")
        if m['status'] == 'placeholder':
            self.get_logger().warn('PLACEHOLDER sensor positions -- NOT measured on this car. Fused detections will '
                                   'be only roughly placed: measure them (comments in the YAML).')
        stamp = self.get_clock().now().to_msg()
        msgs = []
        for pre in prefixes:
            for parent, child, t, rpy, q in frame_list(m, pre, laser):
                ts = TransformStamped()
                ts.header.stamp = stamp
                ts.header.frame_id = parent
                ts.child_frame_id = child
                ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = t
                (ts.transform.rotation.x, ts.transform.rotation.y,
                 ts.transform.rotation.z, ts.transform.rotation.w) = q
                msgs.append(ts)
                self.get_logger().info(f'  {parent} -> {child}: x {t[0]:.3f} y {t[1]:.3f} z {t[2]:.3f} m, '
                                       f'roll {rpy[0]:.1f} pitch {rpy[1]:.1f} yaw {rpy[2]:.1f} deg')
        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(msgs)
        self.get_logger().info(f'{len(msgs)} static transforms published (they stay while this node runs)')


def main(args=None):
    rclpy.init(args=args)
    try:
        node = SensorFrames()
    except (OSError, ValueError) as e:
        rclpy.logging.get_logger('sensor_frames').fatal(str(e))
        rclpy.try_shutdown()
        sys.exit(1)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
