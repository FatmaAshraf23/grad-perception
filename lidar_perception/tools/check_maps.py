#!/usr/bin/env python3
"""check_maps.py <car_map.yaml> <world_map.yaml> [timeout_s] -- does the car_map split work?

With sim_perception.launch.py car_map:=... the simulator draws its LiDAR scans from the WORLD
map (track + obstacles) while the car's software gets the CAR map (walls only) on /map.
Run while the simulator is up. Checks:
  1. /map is cell for cell the car map file (as map_server reads it) and ONLY car_map_server
     publishes it -- so AMCL, the EKF and the detector cannot get the world map by mistake
  2. /map_world is cell for cell the world map file
  3. the two differ only where the world has extra obstacles
Exit 0 = all good.
"""
import os
import sys
import time

import numpy as np
import rclpy
import yaml
from nav_msgs.msg import OccupancyGrid
from PIL import Image
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def grid_from_file(yaml_path):
    """The OccupancyGrid map_server makes of a map yaml (trinary): 100 / 0 / -1, row 0 = bottom."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    img = np.array(Image.open(os.path.join(os.path.dirname(os.path.abspath(yaml_path)), cfg['image'])).convert('L'))
    shade = img.astype(np.float64) / 255.0
    occ = shade if cfg.get('negate', 0) else 1.0 - shade
    g = np.full(img.shape, -1, dtype=np.int8)
    g[occ > cfg['occupied_thresh']] = 100
    g[occ < cfg['free_thresh']] = 0
    return g[::-1], cfg


def counts(g):
    return f'free {int((g == 0).sum())}, occupied {int((g == 100).sum())}, unknown {int((g == -1).sum())}'


class Grab(Node):
    def __init__(self):
        super().__init__('check_maps')
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.maps = {}
        for topic in ('/map', '/map_world'):
            self.create_subscription(OccupancyGrid, topic, lambda m, t=topic: self.maps.__setitem__(t, m), qos)


def compare(topic, msg, yaml_path):
    want, cfg = grid_from_file(yaml_path)
    info = msg.info
    got = np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width)
    ok = True
    print(f'{topic}: {info.width} x {info.height} @ {info.resolution:.3f} m, origin '
          f'({info.origin.position.x:.3f}, {info.origin.position.y:.3f}), frame {msg.header.frame_id}: {counts(got)}')
    if got.shape != want.shape or abs(info.resolution - cfg['resolution']) > 1e-9 \
            or abs(info.origin.position.x - cfg['origin'][0]) > 1e-6 or abs(info.origin.position.y - cfg['origin'][1]) > 1e-6:
        print(f'  DIFFERENT size/resolution/origin from {yaml_path}')
        return False, got
    bad = int((got != want).sum())
    print(f'  vs {os.path.basename(yaml_path)}: {bad} cells differ' + ('  OK' if bad == 0 else '  WRONG'))
    ok &= bad == 0
    return ok, got


def main():
    car_yaml, world_yaml = sys.argv[1], sys.argv[2]
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    rclpy.init()
    node = Grab()
    t0 = time.time()
    while time.time() - t0 < timeout and len(node.maps) < 2:
        rclpy.spin_once(node, timeout_sec=0.2)
    time.sleep(1.0)                                   # let discovery finish before listing publishers
    rclpy.spin_once(node, timeout_sec=0.2)
    pubs = {t: sorted(i.node_name for i in node.get_publishers_info_by_topic(t)) for t in ('/map', '/map_world')}
    ok = True
    for t in ('/map', '/map_world'):
        print(f'publishers of {t}: {pubs[t]}')
    if pubs['/map'] != ['car_map_server']:
        print('  WRONG: /map must come from car_map_server only'); ok = False
    if pubs['/map_world'] != ['map_server']:
        print("  WRONG: /map_world must come from the simulator's map_server only"); ok = False
    for t in ('/map', '/map_world'):
        if t not in node.maps:
            print(f'{t}: NO MESSAGE within {timeout:.0f} s'); ok = False
    if ok:
        ok_c, car = compare('/map', node.maps['/map'], car_yaml)
        ok_w, world = compare('/map_world', node.maps['/map_world'], world_yaml)
        ok = ok_c and ok_w
        if car.shape == world.shape:
            extra = int(((world == 100) & (car != 100)).sum())
            missing = int(((car == 100) & (world != 100)).sum())
            print(f'world vs car map: {extra} cells occupied only in the world (the obstacles), {missing} only in the car map')
            ok &= missing == 0 and extra > 0
    node.destroy_node()
    rclpy.try_shutdown()
    print('MAP SPLIT OK' if ok else 'MAP SPLIT FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
