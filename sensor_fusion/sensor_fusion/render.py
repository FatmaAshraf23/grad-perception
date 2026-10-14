"""render.py -- a synthetic camera picture of our checkerboard (for calib_fake_session and the tests): the printed A4
page seen by a calibrated camera at a known pose, through the lens's distortion, in front of a plain room, with a
little blur and noise.
How: the page (a texture) -> the ideal (undistorted) picture with a homography; then every pixel of the real
(distorted) picture looks up where it comes from in the ideal one (projection.pixel_to_ray) -> cv2.remap."""
import numpy as np

from sensor_fusion import projection as pj

PAPER = (0.297, 0.210)                      # A4 landscape [m]


def board_texture(pattern=(9, 6), square=(0.025, 0.176 / 7), paper=PAPER, px_per_mm=1.5):
    """The printed page (grayscale) + A: texture pixel (col, row, 1) -> board frame (x, y, 1) [m]. Board frame = the
    inner-corner frame of lidar_camera_calib.board_object_points (origin at the first inner corner, x across, y down);
    the pattern is centred on the page, its top-left square black (as make_checkerboard.py prints it)."""
    cols, rows = pattern
    sx, sy = square
    nx, ny = cols + 1, rows + 1                                   # squares
    x0 = -sx - (paper[0] - nx * sx) / 2                           # the page's top-left corner in the board frame
    y0 = -sy - (paper[1] - ny * sy) / 2
    s = 1.0 / (1000.0 * px_per_mm)                                # metres per texture pixel
    W, H = int(round(paper[0] / s)), int(round(paper[1] / s))
    bx = x0 + (np.arange(W) + 0.5) * s
    by = y0 + (np.arange(H) + 0.5) * s
    k = np.floor((bx + sx) / sx)[None, :]
    r = np.floor((by + sy) / sy)[:, None]
    black = (k >= 0) & (k < nx) & (r >= 0) & (r < ny) & ((k + r) % 2 == 0)
    tex = np.full((H, W), 232, np.uint8)
    tex[black] = 28
    A = np.array([[s, 0.0, x0 + 0.5 * s], [0.0, s, y0 + 0.5 * s], [0.0, 0.0, 1.0]])
    return tex, A


class Renderer:

    def __init__(self, cam, pattern=(9, 6), square=(0.025, 0.176 / 7), seed=0, blur=0.7, noise=2.0, supersample=3):
        """supersample S: the picture is drawn S x S times finer, then averaged down (cv2.INTER_AREA) -- like a real
        sensor pixel collecting light over its whole area. Without it the 11-pixel squares of a far board come out
        jagged and the corners ~0.3 px off (test 2026-10-14)."""
        import cv2
        self.cv2, self.cam, self.S = cv2, cam, int(supersample)
        self.rng = np.random.default_rng(seed)
        self.blur, self.noise = blur, noise
        self.tex, self.A = board_texture(pattern, square)
        S, W, H = self.S, cam.width, cam.height
        self.Ks = cam.K.copy()                                    # the camera matrix of the S-times finer picture
        self.Ks[0, 0] *= S
        self.Ks[0, 1] *= S
        self.Ks[1, 1] *= S
        self.Ks[0, 2] = (cam.cx + 0.5) * S - 0.5
        self.Ks[1, 2] = (cam.cy + 0.5) * S - 0.5
        us = (np.arange(W * S, dtype=float) + 0.5) / S - 0.5      # fine pixel centres in real-pixel coordinates
        vs = (np.arange(H * S, dtype=float) + 0.5) / S - 0.5
        uu, vv = np.meshgrid(us, vs)
        rays, ok = cam.pixel_to_ray(np.column_stack([uu.ravel(), vv.ravel()]))
        x, y = rays[:, 0] / rays[:, 2], rays[:, 1] / rays[:, 2]
        mx = self.Ks[0, 0] * x + self.Ks[0, 1] * y + self.Ks[0, 2]
        my = self.Ks[1, 1] * y + self.Ks[1, 2]
        mx[~ok], my[~ok] = -1.0, -1.0
        self.map_x = mx.reshape(H * S, W * S).astype(np.float32)
        self.map_y = my.reshape(H * S, W * S).astype(np.float32)
        g = np.linspace(0.0, 1.0, H * S)[:, None]                # a plain room: lighter wall, darker floor
        self.background = (np.where(g < 0.55, 150 - 25 * g, 95 + 20 * g) * np.ones((1, W * S))).astype(np.uint8)

    def render(self, T_cam_board=None):
        """-> BGR picture (uint8) with the board at T_cam_board (None = no board in view)."""
        cv2, S = self.cv2, self.S
        ideal = self.background.copy()
        if T_cam_board is not None:
            R, t = T_cam_board[:3, :3], T_cam_board[:3, 3]
            Hm = self.Ks @ np.column_stack([R[:, 0], R[:, 1], t]) @ self.A     # texture pixel -> fine ideal pixel
            if t[2] > 0.05:
                cv2.warpPerspective(self.tex, Hm, (ideal.shape[1], ideal.shape[0]), dst=ideal,
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        img = cv2.remap(ideal, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if S > 1:
            img = cv2.resize(img, (self.cam.width, self.cam.height), interpolation=cv2.INTER_AREA)
        if self.blur > 0:
            img = cv2.GaussianBlur(img, (0, 0), self.blur)
        img = np.clip(img + self.rng.normal(0.0, self.noise, img.shape), 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
