"""calib_session.py -- one camera-LiDAR calibration session on disk (written by calib_capture, read by calib_solve).

A folder with
  session.npz   the camera (K, D, size, frame), the board (pattern, square sizes), the first-guess TF at capture time
                (T_cam_from_laser; NaN when there was none), and per saved pose: the 54 corner pixels, the picture's
                time and EVERY LiDAR scan taken while the board stood still (ranges + angle_min / increment /
                range_min / range_max). Plain arrays only (no pickle).
  pose_XX.png   the camera picture of each pose (calib_solve draws its overlay check on it)
"""
import os

import numpy as np

from sensor_fusion import projection as pj


class Session:

    def __init__(self, K, D, width, height, pattern=(9, 6), square=(0.025, 0.176 / 7), camera_frame='',
                 laser_frame='', T_tf=None):
        self.K, self.D = np.asarray(K, float).reshape(3, 3), np.asarray(D, float).ravel()
        self.width, self.height = int(width), int(height)
        self.pattern, self.square = tuple(int(v) for v in pattern), tuple(float(v) for v in square)
        self.camera_frame, self.laser_frame = str(camera_frame), str(laser_frame)
        self.T_tf = np.full((4, 4), np.nan) if T_tf is None else np.asarray(T_tf, float)
        self.corners, self.pose_stamp = [], []
        self.scan_ranges, self.scan_meta, self.scan_stamp, self.scan_pose = [], [], [], []

    def add_pose(self, corners, stamp, scans):
        """scans: [(stamp, ranges, (angle_min, angle_increment, range_min, range_max)), ...] -> the pose's index."""
        k = len(self.corners)
        self.corners.append(np.asarray(corners, float).reshape(-1, 2))
        self.pose_stamp.append(float(stamp))
        for st, r, meta in scans:
            r = np.asarray(r, np.float32).ravel()
            if self.scan_ranges and r.size != self.scan_ranges[0].size:
                raise ValueError(f'scan with {r.size} beams, earlier ones had {self.scan_ranges[0].size}')
            self.scan_ranges.append(r)
            self.scan_meta.append(np.asarray(meta, float))
            self.scan_stamp.append(float(st))
            self.scan_pose.append(k)
        return k

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        n = len(self.scan_ranges)
        np.savez_compressed(
            os.path.join(folder, 'session.npz'), K=self.K, D=self.D, size=np.array([self.width, self.height]),
            pattern=np.array(self.pattern), square=np.array(self.square), camera_frame=np.array(self.camera_frame),
            laser_frame=np.array(self.laser_frame), T_tf=self.T_tf,
            corners=np.array(self.corners).reshape(len(self.corners), -1, 2),
            pose_stamp=np.array(self.pose_stamp),
            scan_ranges=np.vstack(self.scan_ranges) if n else np.zeros((0, 0), np.float32),
            scan_meta=np.vstack(self.scan_meta) if n else np.zeros((0, 4)),
            scan_stamp=np.array(self.scan_stamp), scan_pose=np.array(self.scan_pose, int))

    @classmethod
    def load(cls, folder):
        z = np.load(os.path.join(os.path.expanduser(folder), 'session.npz'), allow_pickle=False)
        s = cls(z['K'], z['D'], int(z['size'][0]), int(z['size'][1]), z['pattern'], z['square'],
                str(z['camera_frame']), str(z['laser_frame']), z['T_tf'])
        s.corners = [c for c in z['corners']]
        s.pose_stamp = list(z['pose_stamp'])
        s.scan_ranges = [r for r in z['scan_ranges']]
        s.scan_meta = [m for m in z['scan_meta']]
        s.scan_stamp = list(z['scan_stamp'])
        s.scan_pose = list(z['scan_pose'])
        return s

    def calib_poses(self, z=0.0):
        """-> [{'corners', 'scan_points' (laser frame), 'n_scans'}] for lidar_camera_calib.calibrate()."""
        pose_of = np.array(self.scan_pose, int)
        out = []
        for k, c in enumerate(self.corners):
            pts = [pj.scan_to_points(self.scan_ranges[i], *self.scan_meta[i], z=z)[0] for i in np.nonzero(pose_of == k)[0]]
            out.append({'corners': c, 'scan_points': np.vstack(pts) if pts else np.zeros((0, 3)),
                        'n_scans': len(pts)})
        return out
