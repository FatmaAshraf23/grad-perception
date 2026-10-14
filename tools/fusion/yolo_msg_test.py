"""yolo_msg_test.py <image dir> <out dir> [n] -- test camera_publisher yolo_node's NEW output: publish the test images of
the cone dataset (conesV1/test) one by one as /camera/image_compressed, receive /camera/detections, and compare the
boxes with the dataset's labels (YOLO txt: class cx cy w h, normalized) -> IoU matches. Proves the message carries the
right pixels, the picture's stamp and frame_id. Draws the first images with the received boxes."""
import glob
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from apex_msgs.msg import CameraDetectionArray
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def main():
    img_dir, out_dir = sys.argv[1], sys.argv[2]
    n_max = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))[:n_max]
    rclpy.init()
    node = Node('yolo_msg_test')
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
    pub = node.create_publisher(CompressedImage, '/camera/image_compressed', qos)
    got = []
    node.create_subscription(CameraDetectionArray, '/camera/detections', lambda m: got.append(m), 10)
    t0 = time.time()
    while pub.get_subscription_count() == 0 and time.time() - t0 < 60:     # wait for yolo_node (model loading)
        rclpy.spin_once(node, timeout_sec=0.2)
    print(f'yolo_node subscribed after {time.time() - t0:.1f} s; {len(files)} test pictures')
    tp = fp = fn = 0
    ious, lat, stamps_ok, n_ok = [], [], 0, 0
    for k, fn_img in enumerate(files):
        img = cv2.imread(fn_img)
        H, W = img.shape[:2]
        m = CompressedImage()
        m.header.stamp = node.get_clock().now().to_msg()
        m.header.frame_id = 'camera_optical_frame'
        m.format = 'jpeg'
        m.data = open(fn_img, 'rb').read()
        before = len(got)
        node_t = time.time()
        pub.publish(m)
        while len(got) == before and time.time() - node_t < 5.0:
            rclpy.spin_once(node, timeout_sec=0.02)
        if len(got) == before:
            print(f'  no answer for {os.path.basename(fn_img)}')
            continue
        d = got[-1]
        n_ok += 1
        stamps_ok += int(d.header.stamp == m.header.stamp and d.header.frame_id == 'camera_optical_frame'
                         and d.image_width == W and d.image_height == H)
        lat.append(d.latency_ms)
        boxes = [(b.u_min, b.v_min, b.u_max, b.v_max) for b in d.detections]
        lab = os.path.join(os.path.dirname(img_dir), 'labels', os.path.basename(fn_img)[:-4] + '.txt')
        truth = []
        if os.path.isfile(lab):
            for line in open(lab):
                v = line.split()
                if len(v) == 5:
                    cx, cy, w, h = (float(x) for x in v[1:])
                    truth.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
        used = set()
        for t in truth:
            best = max(((iou(t, b), j) for j, b in enumerate(boxes) if j not in used), default=(0.0, -1))
            if best[0] >= 0.5:
                tp += 1
                used.add(best[1])
                ious.append(best[0])
            else:
                fn += 1
        fp += len(boxes) - len(used)
        if k < 6:
            out = img.copy()
            for t in truth:
                cv2.rectangle(out, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (0, 200, 0), 2)
            for b, det in zip(boxes, d.detections):
                cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
                cv2.putText(out, f'{det.class_name} {det.confidence:.2f}', (int(b[0]), max(12, int(b[1]) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(out_dir, f'yolo_msg_{k}.jpg'), out)
    print(f'{n_ok} of {len(files)} pictures answered; header (stamp + frame_id) and picture size carried over in '
          f'{stamps_ok} of {n_ok}')
    print(f'boxes vs the dataset labels (IoU >= 0.5): {tp} found, {fn} missed, {fp} extra; mean IoU of the found '
          f'{np.mean(ious) if ious else float("nan"):.2f}; latency picture -> boxes median '
          f'{np.median(lat) if lat else float("nan"):.0f} ms (includes this test publishing)')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
