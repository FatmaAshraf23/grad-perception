"""detectors.py -- the cone detector behind ONE interface, so the car can switch runtime without touching the rest
(decision 2026-11-08: CPU first, the Coral later if its runtime installs).

  det = make_detector('onnx:best.onnx:320:2')   YOLO exported to ONNX, onnxruntime on the CPU, 2 threads (the car)
  det = make_detector('ultralytics:best.pt')    the training library itself (laptop: reference for the ONNX path)
  det = make_detector('none')                   no boxes (camera timing / latency tests)
  boxes = det(bgr)   -> [(class_name, confidence, u_min, v_min, u_max, v_max)] in pixels of THAT picture

ONNX: the picture is LETTERBOXED like ultralytics does (shrunk to fit imgsz x imgsz keeping its shape, padded with
grey 114, RGB, 0..1), and the boxes are mapped back exactly (minus the padding, divided by the shrink factor).
Two output layouts, told apart by shape:
  raw        (1, 4 + classes, N)  cx, cy, w, h + one score per class for N candidates (YOLOv8 / YOLO11) -> NMS here
  end2end    (1, N, 6)            x1, y1, x2, y2, score, class, already filtered (NMS-free models, e.g. YOLO26)
No ROS. Tests: test_pi_camera.py (synthetic tensors) + the laptop job comparing ONNX with ultralytics on real pictures.
"""
import time

import numpy as np

try:
    import cv2
except ImportError:                       # tests of the pure-numpy parts only
    cv2 = None


def letterbox(img, size, stride=32):
    """-> (padded square picture size x size, scale, pad_x, pad_y): resized keeping its shape, centred, grey 114"""
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    px, py = (size - nw) / 2.0, (size - nh) / 2.0
    top, left = int(round(py - 0.1)), int(round(px - 0.1))
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, s, left, top


def nms(boxes, scores, iou=0.45):
    """boxes [N, 4] x1 y1 x2 y2, scores [N] -> indices kept, best first"""
    order = np.argsort(-scores)
    keep = []
    area = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        r = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[r, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[r, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[r, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[r, 3])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        o = inter / np.maximum(area[i] + area[r] - inter, 1e-9)
        order = r[o <= iou]
    return keep


def decode_yolo(out, scale, pad_x, pad_y, w, h, conf=0.25, iou=0.45, names=('cone',), max_det=100):
    """one ONNX output tensor -> boxes in pixels of the ORIGINAL w x h picture"""
    a = np.asarray(out, dtype=np.float32)
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f'unexpected YOLO output shape {np.shape(out)}')
    if a.shape[1] == 6 and a.shape[0] != 6:                      # end2end: [N, 6] = x1 y1 x2 y2 score class
        xyxy, score, cls = a[:, :4].copy(), a[:, 4], a[:, 5].astype(int)
        ok = score >= conf
        xyxy, score, cls = xyxy[ok], score[ok], cls[ok]
        keep = list(np.argsort(-score))[:max_det]
    else:                                                        # raw: [4 + classes, N]
        if a.shape[0] > a.shape[1]:                              # some exports give [N, 4 + classes]
            a = a.T
        nc = a.shape[0] - 4
        if nc < 1:
            raise ValueError(f'unexpected YOLO output shape {np.shape(out)}')
        cxcywh, sc = a[:4].T, a[4:].T
        cls = sc.argmax(1)
        score = sc[np.arange(sc.shape[0]), cls]
        ok = score >= conf
        cxcywh, score, cls = cxcywh[ok], score[ok], cls[ok]
        xyxy = np.column_stack([cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
                                cxcywh[:, 0] + cxcywh[:, 2] / 2, cxcywh[:, 1] + cxcywh[:, 3] / 2])
        keep = []
        for c in np.unique(cls):                                 # NMS per class, as ultralytics does
            idx = np.nonzero(cls == c)[0]
            keep += [int(idx[k]) for k in nms(xyxy[idx], score[idx], iou)]
        keep = sorted(keep, key=lambda k: -score[k])[:max_det]
    boxes = []
    for k in keep:
        x1, y1, x2, y2 = xyxy[k]
        u0 = float(np.clip((x1 - pad_x) / scale, 0, w))
        v0 = float(np.clip((y1 - pad_y) / scale, 0, h))
        u1 = float(np.clip((x2 - pad_x) / scale, 0, w))
        v1 = float(np.clip((y2 - pad_y) / scale, 0, h))
        if u1 - u0 < 1 or v1 - v0 < 1:
            continue
        name = names[cls[k]] if 0 <= cls[k] < len(names) else str(int(cls[k]))
        boxes.append((name, float(score[k]), u0, v0, u1, v1))
    return boxes


class NullDetector:
    name = 'none'

    def __call__(self, img):
        return []


class OnnxYolo:
    def __init__(self, path, imgsz=320, conf=0.25, iou=0.45, names=('cone',), threads=2):
        import onnxruntime as ort
        so = ort.SessionOptions()
        # CPU budget (job 245: the default = every core + spinning threads -> 517 % of a laptop core for a 13 ms
        # detector). On the Pi the EKF, AMCL, planning and MPC need the rest: a fixed thread count, no spinning.
        # threads 0 = onnxruntime's default (all cores).
        if threads:
            so.intra_op_num_threads = int(threads)
            so.inter_op_num_threads = 1
        so.add_session_config_entry('session.intra_op.allow_spinning', '0')
        so.add_session_config_entry('session.inter_op.allow_spinning', '0')
        self.sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0].name
        shp = self.sess.get_inputs()[0].shape
        if isinstance(shp[2], int) and shp[2] != imgsz:
            imgsz = shp[2]                                       # a fixed-size export decides
        self.imgsz, self.conf, self.iou, self.names = int(imgsz), conf, iou, tuple(names)
        self.name = f'onnx {path.split("/")[-1]} {self.imgsz}px {threads or "all"} threads'

    def __call__(self, img):
        h, w = img.shape[:2]
        lb, s, px, py = letterbox(img, self.imgsz)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0      # BGR -> RGB, CHW, 0..1
        out = self.sess.run(None, {self.inp: np.ascontiguousarray(x)})[0]
        return decode_yolo(out, s, px, py, w, h, self.conf, self.iou, self.names)


class UltralyticsYolo:
    def __init__(self, path, imgsz=640, conf=0.25, device=None):
        from ultralytics import YOLO
        self.model, self.imgsz, self.conf, self.device = YOLO(path), int(imgsz), conf, device
        self.names = self.model.names
        self.name = f'ultralytics {path.split("/")[-1]} {self.imgsz}px'

    def __call__(self, img):
        r = self.model(img, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        out = []
        if r.boxes is not None and len(r.boxes):
            for (u0, v0, u1, v1), c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                              r.boxes.cls.cpu().numpy().astype(int)):
                out.append((str(self.names.get(int(k), k)) if isinstance(self.names, dict) else str(k), float(c),
                            float(u0), float(v0), float(u1), float(v1)))
        return out


def make_detector(spec, conf=0.25, iou=0.45):
    """'none' | 'onnx:<path>[:imgsz[:threads, default 2]]' | 'ultralytics:<path>[:imgsz]'"""
    parts = spec.split(':')
    kind = parts[0]
    if kind == 'none':
        return NullDetector()
    if kind == 'onnx':
        return OnnxYolo(parts[1], int(parts[2]) if len(parts) > 2 else 320, conf, iou,
                        threads=int(parts[3]) if len(parts) > 3 else 2)
    if kind == 'ultralytics':
        return UltralyticsYolo(parts[1], int(parts[2]) if len(parts) > 2 else 640, conf)
    raise ValueError(f'unknown detector {spec!r} (none | onnx:<file>[:imgsz] | ultralytics:<file>[:imgsz])')


def timed(det, img):
    t0 = time.perf_counter()
    b = det(img)
    return b, (time.perf_counter() - t0) * 1000.0
