#!/usr/bin/env python3
"""AMCL's copy of the map with THIN walls (2026-10-19, AMCL-sliding work).

WHY
    nav2 AMCL's likelihood-field model scores every laser end point by its distance to the nearest
    OCCUPIED cell, so every point that lands INSIDE a wall gets the best score, however deep it is.
    A wall that is several cells thick (closed1: 3 cells = 15 cm; a SLAM map: 1-4 cells) makes the score
    lopsided: moving a particle TOWARDS a wall costs nothing until its end points come out of the far
    side of the wall, moving it AWAY costs as soon as they leave the near side.  Where only some walls
    tell the car's along-track position (the ends of a straight, the corner just passed), the particle
    cloud drifts into that free side and AMCL's estimate slides along the track.
    Found with an offline replica of nav2 AMCL (perfect odometry, closed1): the 3-cell walls alone cost
    ~1-2 cm of along-track bias at 1.5-6 m/s; the other, bigger cause was the motion noise (amcl_sim.yaml).
WHAT
    A wall cell stays occupied only if it touches a free cell (8 neighbours) = the wall SURFACE the
    LiDAR really sees; every other wall cell becomes unknown (like the space behind a wall in a SLAM
    map).  Free and unknown cells, origin and resolution are unchanged, so poses mean the same in both
    maps.  ONLY AMCL gets this map (/map_amcl); the EKF's scan check and the obstacle detector keep /map.
    Cells are read exactly as AMCL reads them (nav2 amcl_node convertMap): 100 = occupied,
    0 = free, anything else = unknown.
NODE amcl_map_filter
    subscribes in_topic (/map) and publishes out_topic (/map_amcl), both latched (transient local),
    once per map that arrives.  thin_walls:=false republishes the map unchanged (A/B tests).
REAL CAR
    The same node between the map_server (the saved slam_toolbox map) and AMCL.  Check the result once
    in Foxglove: every wall must stay a closed line.
"""
import numpy as np


def surface(occ, free):
    """wall cells that touch a free cell (8-neighbourhood); occ, free: 2-D boolean arrays"""
    f = np.pad(free, 1, constant_values=False)
    nb = np.zeros_like(occ)
    h, w = occ.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                nb |= f[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return occ & nb


def thin_walls(grid):
    """OccupancyGrid values (2-D int array) -> the same grid with only the wall surface occupied"""
    grid = np.asarray(grid)
    occ, free = grid == 100, grid == 0
    out = np.where(free, 0, -1).astype(np.int8)
    out[surface(occ, free)] = 100
    return out


try:
    import rclpy
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
except ImportError:
    Node = object


class AmclMapFilter(Node):

    def __init__(self):
        super().__init__('amcl_map_filter')
        self.declare_parameter('in_topic', '/map')
        self.declare_parameter('out_topic', '/map_amcl')
        self.declare_parameter('thin_walls', True)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(OccupancyGrid, p('out_topic'), qos)
        self.create_subscription(OccupancyGrid, p('in_topic'), self.map_cb, qos)
        self.get_logger().info(f"AMCL map: {p('in_topic')} -> {p('out_topic')} "
                               + ('with THIN walls (only the wall surface occupied)' if p('thin_walls')
                                  else 'unchanged (thin_walls false)'))

    def map_cb(self, msg):
        h, w = msg.info.height, msg.info.width
        grid = np.array(msg.data, dtype=np.int16).reshape(h, w)
        out = thin_walls(grid) if self.get_parameter('thin_walls').value else grid.astype(np.int8)
        m = OccupancyGrid()
        m.header = msg.header
        m.info = msg.info
        m.data = out.ravel().tolist()
        self.pub.publish(m)
        n_occ, n_keep = int((grid == 100).sum()), int((out == 100).sum())
        self.get_logger().info(f'map {w} x {h} @ {msg.info.resolution:.3f} m: {n_occ} wall cells -> {n_keep} '
                               f'for AMCL ({n_occ - n_keep} behind the wall surface now unknown)')


def main(args=None):
    rclpy.init(args=args)
    node = AmclMapFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
