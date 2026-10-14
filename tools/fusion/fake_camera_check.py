"""fake_camera_check.py <bag> --cones <csv> --world <yaml> --camera <yaml> [--mounts sim] [--cone-height 0.25] [--out DIR]
Score the simulator's fake camera on a recording (/camera/detections, /ego_racecar/odom = truth, /state_estimate =
estimate, /obstacles = LiDAR) and PREVIEW the camera-LiDAR association:
 1. timing: picture rate, sequence gaps, latency (bag receive time - picture stamp)
 2. detection: per picture, the cones the camera SHOULD see clearly (truth pose; in view, line of sight, box taller than
    full_box_px) -> found = a box with IoU >= 0.3 with the cone's perfect box; box centre error; boxes of no cone
 3. hidden cones: cones in front but behind a wall -> must have no box
 4. FUSION PREVIEW (what the fusion node will do on the car): every LiDAR obstacle of the latest /obstacles message
    projected into the picture with the ESTIMATED pose at the picture time (+ the camera mount): inside a camera box?
    -> share of LiDAR obstacles in view that land in a box, share of cone boxes that contain a LiDAR obstacle, and the
    pixel offset obstacle - box centre (localization + LiDAR errors; what the association gate must allow).
Run in WSL (rosbag2_py, apex_msgs, sensor_fusion)."""
import argparse
import math
import os
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from sensor_fusion import projection as pj
from sensor_fusion.fake_camera import OccupancyMap, load_cones, yaw_of


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def interp_pose(arr, t):
    if not len(arr) or t < arr[0, 0] or t > arr[-1, 0]:
        return None
    return tuple(float(np.interp(t, arr[:, 0], arr[:, k])) for k in (1, 2, 3))


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--cones', required=True)
    ap.add_argument('--world', required=True)
    ap.add_argument('--camera', required=True)
    ap.add_argument('--mounts', default='sim')
    ap.add_argument('--cone-height', type=float, default=0.25)
    ap.add_argument('--max-range', type=float, default=8.0)
    ap.add_argument('--full-box-px', type=float, default=25.0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    from apex_description.mounts import load_mounts
    cam = pj.CameraModel.from_yaml(a.camera)
    T_bo = pj.mount_transforms(load_mounts(a.mounts))['camera_optical_frame']
    cones, world = load_cones(a.cones), OccupancyMap(a.world)

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    dets, truth, est, obst = [], [], [], []
    while reader.has_next():
        topic, data, t_rx = reader.read_next()
        if topic == '/camera/detections':
            dets.append((t_rx * 1e-9, deserialize_message(data, types[topic])))
        elif topic == '/ego_racecar/odom':
            m = deserialize_message(data, types[topic])
            truth.append((stamp(m.header), m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation)))
        elif topic == '/state_estimate':
            m = deserialize_message(data, types[topic])
            est.append((stamp(m.header), m.x_m, m.y_m, m.psi_rad))
        elif topic == '/obstacles':
            m = deserialize_message(data, types[topic])
            obst.append((stamp(m.header), m.status, [(o.x_m, o.y_m, o.radius_m) for o in m.obstacles]))
    if not dets:
        print('NO /camera/detections in the recording')
        sys.exit(1)
    truth, est = np.array(sorted(truth)), np.array(sorted(est))
    for arr in (truth, est):
        if len(arr):
            arr[:, 3] = np.unwrap(arr[:, 3])
    obst.sort(key=lambda r: r[0])
    ob_t = np.array([r[0] for r in obst])

    # 1. timing
    st = np.array([stamp(m.header) for _, m in dets])
    seq = np.array([m.sequence_id for _, m in dets])
    lat = np.array([(rx - stamp(m.header)) * 1000 for rx, m in dets])
    dur = st[-1] - st[0]
    print(f'{len(dets)} pictures over {dur:.1f} s ({len(dets) / max(dur, 1e-9):.1f} Hz), source {dets[0][1].source}, '
          f'frame {dets[0][1].header.frame_id}, schema {dets[0][1].schema_version}; sequence gaps '
          f'{int(np.sum(np.diff(seq) != 1))}; latency (received - picture time) median {np.median(lat):.0f} ms, p95 '
          f'{np.percentile(lat, 95):.0f} ms; boxes per picture mean {np.mean([len(m.detections) for _, m in dets]):.2f}')

    # 2.-4.
    per = {c['label']: dict(should=0, found=0, err=[], hidden=0, hidden_boxed=0) for c in cones}
    stray = 0
    lid_in_view = lid_in_box = boxes_true = boxes_true_with_lidar = 0
    du_list, dv_list, du_loc = [], [], []
    for rx, m in dets:
        t = stamp(m.header)
        tp = interp_pose(truth, t)
        if tp is None:
            continue
        T_mo_true = pj.T_from_xyz_rpy((tp[0], tp[1], 0.0), (0.0, 0.0, tp[2])) @ T_bo
        T_om_true = pj.invert(T_mo_true)
        cxy = T_mo_true[:2, 3]
        boxes = [(d.u_min, d.v_min, d.u_max, d.v_max) for d in m.detections]
        used = set()
        clear_boxes = []
        for c in cones:
            dist = math.hypot(c['x'] - cxy[0], c['y'] - cxy[1])
            b = pj.cone_box(cam, T_om_true, (c['x'], c['y']), c['size'] / 2, a.cone_height)
            if b is None or b['visible_fraction'] < 0.6 or dist > a.max_range:
                continue
            pb = b['box']
            if not world.line_of_sight(cxy[0], cxy[1], c['x'], c['y'], c['size'] / 2 + 0.10):
                per[c['label']]['hidden'] += 1
                if any(iou(pb, bx) >= 0.3 for bx in boxes):
                    per[c['label']]['hidden_boxed'] += 1
                continue
            clear_boxes.append(pb)
            if pb[3] - pb[1] < a.full_box_px:
                continue
            per[c['label']]['should'] += 1
            best = max(((iou(pb, bx), j) for j, bx in enumerate(boxes) if j not in used), default=(0.0, -1))
            if best[0] >= 0.3:
                per[c['label']]['found'] += 1
                used.add(best[1])
                bx = boxes[best[1]]
                per[c['label']]['err'].append(math.hypot((bx[0] + bx[2] - pb[0] - pb[2]) / 2, (bx[1] + bx[3] - pb[1] - pb[3]) / 2))
        stray += sum(1 for j, bx in enumerate(boxes) if j not in used and not any(iou(bx, cb) >= 0.3 for cb in clear_boxes))
        # fusion preview with the ESTIMATED pose
        te = interp_pose(est, t)
        k = int(np.searchsorted(ob_t, t, side='right')) - 1
        if te is None or k < 0 or t - ob_t[k] > 0.3 or obst[k][1] != 0:
            continue
        T_om_est = pj.invert(pj.T_from_xyz_rpy((te[0], te[1], 0.0), (0.0, 0.0, te[2])) @ T_bo)
        uv_obs = []                                      # LiDAR obstacles in view: (u, v) with the estimated pose
        for (ox, oy, _orad) in obst[k][2]:
            uv, ok = cam.project(pj.apply(T_om_est, [ox, oy, a.cone_height / 2]))
            if ok[0] and cam.in_image(uv)[0]:
                uv_obs.append((uv[0, 0], uv[0, 1], ox, oy))

        def inside(bx, u, v, m=5.0):
            return bx[0] - m <= u <= bx[2] + m and bx[1] - m <= v <= bx[3] + m

        for (u, v, ox, oy) in uv_obs:
            lid_in_view += 1
            hit = [bx for bx in boxes if inside(bx, u, v)]
            if hit:
                lid_in_box += 1
                bx = min(hit, key=lambda b_: abs((b_[0] + b_[2]) / 2 - u))
                du_list.append(u - (bx[0] + bx[2]) / 2)
                dv_list.append(v - (bx[1] + bx[3]) / 2)
            uvt, okt = cam.project(pj.apply(T_om_true, [ox, oy, a.cone_height / 2]))
            if okt[0]:
                du_loc.append(u - uvt[0, 0])
        for j in used:
            boxes_true += 1
            if any(inside(boxes[j], u, v) for (u, v, _, _) in uv_obs):
                boxes_true_with_lidar += 1
    print('DETECTION (cones the camera should see clearly = in view, line of sight, box >= '
          f'{a.full_box_px:.0f} px tall):')
    for lab, s in per.items():
        e = np.array(s['err'])
        print(f"  {lab:>10}: found in {s['found']} of {s['should']} pictures "
              f"({100 * s['found'] / max(1, s['should']):.0f} %); box centre error median "
              f"{np.median(e) if len(e) else float('nan'):.1f} px, p95 {np.percentile(e, 95) if len(e) else float('nan'):.1f} px; "
              f"behind a wall in {s['hidden']} pictures, boxed there {s['hidden_boxed']}")
    print(f'  boxes of no cone (false): {stray} in {len(dets)} pictures')
    du, dv = np.abs(du_list), np.abs(dv_list)
    print('FUSION PREVIEW (LiDAR obstacles projected with the ESTIMATED pose into the same picture):')
    print(f'  LiDAR obstacles in view: {lid_in_view}; inside a camera box: {lid_in_box} '
          f'({100 * lid_in_box / max(1, lid_in_view):.0f} %)')
    print(f'  camera boxes of a real cone: {boxes_true}; with a LiDAR obstacle inside: {boxes_true_with_lidar} '
          f'({100 * boxes_true_with_lidar / max(1, boxes_true):.0f} %)')
    if len(du):
        print(f'  offset obstacle - box centre: horizontal median {np.median(du):.1f} px, p95 {np.percentile(du, 95):.1f} px; '
              f'vertical median {np.median(dv):.1f} px, p95 {np.percentile(dv, 95):.1f} px')
    if du_loc:
        dl = np.abs(du_loc)
        print(f'  of which the LOCALIZATION error alone (estimated vs true pose): horizontal median {np.median(dl):.1f} px, '
              f'p95 {np.percentile(dl, 95):.1f} px')
    if a.out:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(a.out, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist(du_list, bins=40, color='tab:blue', alpha=0.7, label='horizontal')
        ax[0].hist(dv_list, bins=40, color='tab:orange', alpha=0.6, label='vertical')
        ax[0].set_xlabel('LiDAR obstacle (projected) - camera box centre [px]')
        ax[0].set_title('fusion preview: how far apart the two sensors put a cone', fontsize=9)
        ax[0].legend()
        ts = [stamp(m.header) - st[0] for _, m in dets]
        ax[1].plot(ts, [len(m.detections) for _, m in dets], '.', ms=3)
        ax[1].set_xlabel('time [s]')
        ax[1].set_ylabel('boxes per picture')
        ax[1].set_title('fake camera: boxes per picture over the run', fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, 'fake_camera_check.png'), dpi=110)
        print(f"-> {os.path.join(a.out, 'fake_camera_check.png')}")


if __name__ == '__main__':
    main()
