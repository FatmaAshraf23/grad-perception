"""detection_packet.py -- the camera's boxes as ONE small UDP datagram (Pi camera on the host -> ROS in the container).

WHY (decision 2026-11-08): the car's CSI camera works only through Raspberry Pi's libcamera / picamera2, which run on
Raspberry Pi OS, not in our Ubuntu 22.04 ROS 2 container. So capture + YOLO run on the host (pi_camera/cam_detect.py)
and send only the detections; camera_publisher detection_bridge publishes them as apex_msgs/CameraDetectionArray.
Both share the kernel, so CLOCK_REALTIME is the same clock on both sides (the container uses --net=host: 127.0.0.1).

PACKET (JSON, UTF-8, < 2 kB for 30 boxes)
  v           1 (format version)
  seq         +1 per picture (a gap = a lost datagram or a dropped picture)
  stamp_ns    when the PICTURE was taken, CLOCK_REALTIME ns (= ROS time) -- not when the boxes were ready
  w, h        the picture the boxes refer to (the detector's input before its own letterbox)
  det_ms      detector time for this picture [ms]
  src         'picamera2' | 'folder' | 'video' + detector name, for the log
  boxes       [[class_name, confidence, u_min, v_min, u_max, v_max], ...] in pixels of w x h
No ROS here (both sides import it; tests: pi_camera/test_pi_camera.py).
"""
import json

VERSION = 1
MAX_BYTES = 60000          # a UDP datagram on localhost; ~1500 boxes -- far more than a picture ever has


def encode(seq, stamp_ns, w, h, boxes, det_ms=0.0, src=''):
    d = {'v': VERSION, 'seq': int(seq), 'stamp_ns': int(stamp_ns), 'w': int(w), 'h': int(h),
         'det_ms': round(float(det_ms), 2), 'src': str(src),
         'boxes': [[str(c), round(float(p), 4), round(float(u0), 2), round(float(v0), 2), round(float(u1), 2),
                    round(float(v1), 2)] for c, p, u0, v0, u1, v1 in boxes]}
    b = json.dumps(d, separators=(',', ':')).encode('utf-8')
    if len(b) > MAX_BYTES:
        raise ValueError(f'packet too big ({len(b)} bytes, {len(boxes)} boxes)')
    return b


def decode(data):
    """bytes -> dict (raises ValueError on anything malformed)"""
    try:
        d = json.loads(data.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f'not a detection packet: {e}') from None
    if not isinstance(d, dict) or d.get('v') != VERSION:
        raise ValueError(f'unknown packet version {d.get("v") if isinstance(d, dict) else "?"}')
    for k in ('seq', 'stamp_ns', 'w', 'h', 'boxes'):
        if k not in d:
            raise ValueError(f'packet without {k}')
    if int(d['w']) <= 0 or int(d['h']) <= 0:
        raise ValueError('bad picture size')
    boxes = []
    for b in d['boxes']:
        if len(b) != 6:
            raise ValueError('a box needs 6 values')
        boxes.append((str(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5])))
    d['boxes'] = boxes
    d.setdefault('det_ms', 0.0)
    d.setdefault('src', '')
    return d


def scale_camera_info(K, width, height, w, h):
    """a calibration made at width x height -> the same camera at w x h (a pure SHRINK of the same view): fx, cx scale
    by w / width, fy, cy by h / height, with the pixel-centre convention (as camcal_calibrate.py). D is unchanged.
    K: 9 numbers row by row -> 9 numbers."""
    sx, sy = w / width, h / height
    K = [float(v) for v in K]
    return [K[0] * sx, K[1] * sx, (K[2] + 0.5) * sx - 0.5,
            K[3], K[4] * sy, (K[5] + 0.5) * sy - 0.5,
            K[6], K[7], K[8]]


def load_camera_yaml(path):
    import yaml
    with open(path) as f:
        c = yaml.safe_load(f)
    return {'width': int(c['image_width']), 'height': int(c['image_height']), 'K': list(c['camera_matrix']['data']),
            'D': list(c['distortion_coefficients']['data']), 'model': c.get('distortion_model', 'plumb_bob'),
            'name': c.get('camera_name', '')}


def check_packet(d, now_ns, last_seq, max_age_ns):
    """-> (ok, reason, lost): ok False = drop it; lost = pictures missing before it (sequence gap)"""
    age = now_ns - int(d['stamp_ns'])
    lost = 0 if last_seq is None or d['seq'] <= last_seq else d['seq'] - last_seq - 1
    if last_seq is not None and d['seq'] <= last_seq:
        return False, 'old_seq', 0
    if age > max_age_ns:
        return False, 'too_old', lost
    if age < -50_000_000:
        return True, 'future', lost
    return True, '', lost
