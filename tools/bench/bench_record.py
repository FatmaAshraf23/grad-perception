#!/usr/bin/env python3
"""Record a LiDAR or IMU topic to a .npz file for the bench checks.

  python3 bench_record.py scan --topic /scan      --seconds 60   --out scan_2m.npz
  python3 bench_record.py imu  --topic /imu/data  --seconds 3600 --out imu_run1.npz

Needs ROS 2 (rclpy) sourced.  Ctrl+C stops early and still saves what was recorded.
Uses the sensor-data QoS (best effort), which is what LiDAR / IMU drivers publish with.

scan -> keys: ranges[n_scans, n_beams] (NaN = no return or padding), angle_min[n], angle_increment[n],
         range_min, range_max, stamps[n] (header time), rx[n] (time we received it)
imu  -> keys: t (header time, falls back to receive time if the driver leaves it 0),
         gx gy gz [rad/s], ax ay az [m/s^2], rx (receive time)
Why both stamps: on the real car check that header time ~ receive time (see the latency test).
"""
import argparse
import sys
import time

import numpy as np


def save_scan(path, rows):
    n = len(rows)
    width = max(len(r["ranges"]) for r in rows)
    ranges = np.full((n, width), np.nan)
    for i, r in enumerate(rows):
        x = np.asarray(r["ranges"], float)
        x[~np.isfinite(x) | (x <= 0)] = np.nan          # inf / 0 = "no return"
        ranges[i, : len(x)] = x
    np.savez(path, ranges=ranges,
             angle_min=np.array([r["angle_min"] for r in rows]),
             angle_increment=np.array([r["angle_increment"] for r in rows]),
             range_min=rows[0]["range_min"], range_max=rows[0]["range_max"],
             stamps=np.array([r["stamp"] for r in rows]), rx=np.array([r["rx"] for r in rows]))


def save_imu(path, rows):
    arr = np.array(rows, float)
    t = arr[:, 0].copy()
    if not np.all(t > 0):                                 # driver left the stamp at 0
        t = arr[:, 7]
    np.savez(path, t=t, gx=arr[:, 1], gy=arr[:, 2], gz=arr[:, 3],
             ax=arr[:, 4], ay=arr[:, 5], az=arr[:, 6], rx=arr[:, 7])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["scan", "imu"])
    ap.add_argument("--topic")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    topic = args.topic or ("/scan" if args.kind == "scan" else "/imu/data")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    rclpy.init()
    node = Node("bench_record")
    rows = []

    if args.kind == "scan":
        from sensor_msgs.msg import LaserScan

        def cb(m):
            rows.append(dict(ranges=list(m.ranges), angle_min=m.angle_min, angle_increment=m.angle_increment,
                             range_min=m.range_min, range_max=m.range_max,
                             stamp=m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec,
                             rx=time.time()))
        node.create_subscription(LaserScan, topic, cb, qos_profile_sensor_data)
    else:
        from sensor_msgs.msg import Imu

        def cb(m):
            rows.append((m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec,
                         m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z,
                         m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z,
                         time.time()))
        node.create_subscription(Imu, topic, cb, qos_profile_sensor_data)

    t0 = time.time()
    last = t0
    print(f"recording {args.kind} from {topic} for {args.seconds:.0f} s (Ctrl+C to stop early) ...")
    try:
        while rclpy.ok() and time.time() - t0 < args.seconds:
            rclpy.spin_once(node, timeout_sec=0.2)
            if time.time() - last >= 10:
                last = time.time()
                print(f"  {last - t0:5.0f} s  {len(rows)} messages  ({len(rows) / max(last - t0, 1e-9):.1f} /s)")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if not rows:
        print("NOTHING RECORDED - is the topic name right? (ros2 topic list; ros2 topic hz <topic>)")
        return 1
    (save_scan if args.kind == "scan" else save_imu)(args.out, rows)
    print(f"saved {len(rows)} messages -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
