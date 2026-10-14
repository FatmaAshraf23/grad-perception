"""lidar_camera_calib.py -- where the camera sits relative to the 2-D LiDAR (camera-LiDAR EXTRINSIC calibration),
from the checkerboard held at different angles in front of the car.
Method: Zhang & Pless (2004), "Extrinsic calibration of a camera and laser range finder (improves camera
calibration)", IROS -- + a robust Levenberg-Marquardt refinement + a bootstrap spread.

THE IDEA
  The CAMERA measures the board's PLANE: its pose from the 54 corners (solvePnP with the calibrated lens).
  The LIDAR sees a LINE of points across the same board. Every one of those points must lie on that plane:
        n_i . (R p_ij + t) = d_i
  n_i, d_i   board i's plane in the camera frame (n_i . X = d_i)
  p_ij       LiDAR point j on board i (laser frame, z = 0: a 2-D LiDAR)
  R, t       the unknown T_cam_from_laser (camera_optical_frame <- laser)
  A line on a plane = 2 independent equations per board pose -> the 6 unknowns from >= 3 poses; use 12-20.
WHY NOT POLES / CONES: they only say where a point lands LEFT-RIGHT in the picture. Sliding the LiDAR's scan plane
  up or down a vertical pole changes nothing the camera sees -> its height and pitch relative to the camera stay
  unknown.
THE BOARDS MUST BE TILTED forward / back, not only turned: with vertical boards only the same happens (height, pitch,
  roll unobservable). The bootstrap spread then shows it (flagged WEAK) instead of a silently wrong number.
OUTPUT: T_cam_from_laser and -- with the LiDAR's measured mount (T_base_laser) -- the camera's mount relative to
  base_link: x, y, z, roll, pitch, yaw = the camera section of apex_description sensor_mounts_car.yaml.
"""
import math

import numpy as np

from sensor_fusion import projection as pj

PATTERN = (9, 6)                      # inner corners (cols, rows) of our printed board
SQUARE = (0.025, 0.176 / 7)           # m (across, down): measured 250 x 176 mm (2026-10-13)
BOARD_MARGIN = 0.045                  # m beyond the inner corners: 1 square + the A4 paper's white border
# camera_link -> camera_optical_frame (columns = optical x, y, z in camera_link): the fixed optical turn
R_LINK_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
WEAK_M, WEAK_DEG = 0.01, 0.5          # bootstrap spread above this = WEAK (not determined by the data)


# ---------------------------------------------------------------- small geometry helpers
def board_object_points(pattern=PATTERN, square=SQUARE):
    """The inner corners in the board frame, in cv2.findChessboardCorners order (as camcal_calibrate.py):
    x along a row (across), y down the rows, z = 0 (z points away from the camera), origin = the first corner."""
    cols, rows = pattern
    g = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(float)
    return np.column_stack([g[:, 0] * square[0], g[:, 1] * square[1], np.zeros(len(g))])


def so3_exp(w):
    w = np.asarray(w, float)
    th = float(np.linalg.norm(w))
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + math.sin(th) / th * K + (1 - math.cos(th)) / th ** 2 * (K @ K)


def rot_angle_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))


def camera_mount(T_base_opt):
    """T_base_from_camera_optical_frame -> (x, y, z [m], roll, pitch, yaw [deg]) of camera_link = the numbers of the
    camera section of sensor_mounts_*.yaml (ROS rpy, R = Rz(yaw) Ry(pitch) Rx(roll))."""
    R = np.asarray(T_base_opt, float)[:3, :3] @ R_LINK_OPT.T
    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    pitch = math.degrees(-math.asin(max(-1.0, min(1.0, R[2, 0]))))
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    x, y, z = (float(v) for v in np.asarray(T_base_opt)[:3, 3])
    return np.array([x, y, z, roll, pitch, yaw])


def plane_from_pose(R_cb, t_cb):
    """Board pose in the camera frame -> its plane n . X = d, d >= 0 (n points from the camera towards the board)."""
    n = np.asarray(R_cb, float)[:, 2].copy()
    d = float(n @ np.asarray(t_cb, float))
    return (-n, -d) if d < 0 else (n, d)


def board_pose_pnp(corners, cam, objp):
    """Detected corner pixels [N, 2] -> board pose (R_cb, t_cb) in the camera frame + reprojection rms [px]."""
    import cv2
    corners = np.asarray(corners, float).reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(objp, corners.reshape(-1, 1, 2), cam.K, cam.D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()
    uv, okp = cam.project(objp @ R.T + t)
    if not okp.all():
        return None
    return R, t, float(np.sqrt(np.mean(np.sum((uv - corners) ** 2, 1))))


# ---------------------------------------------------------------- the solver
def _stack(obs):
    P = np.vstack([o['pts'] for o in obs])
    N = np.vstack([np.repeat(o['n'][None, :], len(o['pts']), 0) for o in obs])
    D = np.concatenate([np.full(len(o['pts']), o['d']) for o in obs])
    return P, N, D


def _huber_cost(r, delta):
    a = np.abs(r)
    return float(np.sum(np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))))


def refine(obs, T0, delta=0.01, iters=100):
    """Robust (Huber, delta [m]) Levenberg-Marquardt on every point-on-plane residual n.(R p + t) - d.
    obs: [{'n', 'd', 'pts' (laser frame [M, 3])}, ...]. -> T_cam_from_laser, info."""
    P, N, D = _stack(obs)
    R, t = np.array(T0, float)[:3, :3].copy(), np.array(T0, float)[:3, 3].copy()

    def res(R, t):
        return np.einsum('ij,ij->i', N, P @ R.T + t) - D

    r = res(R, t)
    cost, lam, it = _huber_cost(r, delta), 1e-3, 0
    for it in range(1, iters + 1):
        q = P @ R.T + t
        a = np.abs(r)
        w = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-300))
        J = np.hstack([np.cross(q, N), N])                 # d r / d(rotation, translation)
        A = J.T @ (w[:, None] * J)
        g = J.T @ (w * r)
        accepted = False
        while lam < 1e10:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A)) + 1e-12 * np.eye(6), -g)
            R2, t2 = so3_exp(step[:3]) @ R, t + step[3:]
            r2 = res(R2, t2)
            c2 = _huber_cost(r2, delta)
            if c2 <= cost:
                R, t, r, cost, lam, accepted = R2, t2, r2, c2, max(lam / 3, 1e-9), True
                break
            lam *= 4
        if not accepted or np.linalg.norm(step) < 1e-11:
            break
    per_board, k = [], 0
    for o in obs:
        m = len(o['pts'])
        per_board.append(float(np.sqrt(np.mean(r[k:k + m] ** 2))))
        k += m
    return pj._T(R, t), {'rms': float(np.sqrt(np.mean(r ** 2))), 'cost': cost, 'points': len(r), 'iterations': it,
                         'per_board_rms': per_board, 'residuals': r}


def linear_init(obs):
    """Zhang-Pless closed form: R p + t = [r1 r2 t] (x, y, 1)^T for a 2-D LiDAR (z = 0), so n^T H q = d is LINEAR in
    the 9 entries of H = [r1 r2 t]. Least squares, then the nearest rotation. -> T, condition number (huge =
    the board poses do not determine everything, e.g. no tilted boards)."""
    rows, b = [], []
    for o in obs:
        p = o['pts']
        Q = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
        rows.append(np.einsum('mj,i->mji', Q, o['n']).reshape(len(p), 9))   # kron(q, n): coefficient of H[i, j]
        b.append(np.full(len(p), o['d']))
    A, b = np.vstack(rows), np.concatenate(b)
    sv = np.linalg.svd(A, compute_uv=False)
    cond = float(sv[0] / max(sv[-1], 1e-300))
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    H = h.reshape(3, 3).T                                   # h = [column 0; column 1; column 2]
    M = np.column_stack([H[:, 0], H[:, 1], np.cross(H[:, 0], H[:, 1])])
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return pj._T(R, H[:, 2]), cond


def associate(pts_laser, R_cb, t_cb, T_cam_laser, objp, margin=BOARD_MARGIN, gate=0.05, line_tol=0.03, rng=None):
    """Which LiDAR points lie on board i? -> their indices. Using the current T_cam_laser guess, a point must be
    within `gate` of the board's plane AND inside the board's area (+ margin); then a RANSAC line through them
    in the scan plane drops stray returns (a hand, the stand)."""
    pts_laser = np.asarray(pts_laser, float).reshape(-1, 3)
    if len(pts_laser) == 0:
        return np.zeros(0, int)
    Xb = (pj.apply(T_cam_laser, pts_laser) - t_cb) @ R_cb       # in the board frame
    lo, hi = objp[:, :2].min(0) - margin, objp[:, :2].max(0) + margin
    keep = (np.abs(Xb[:, 2]) < gate) & np.all(Xb[:, :2] >= lo, 1) & np.all(Xb[:, :2] <= hi, 1)
    idx = np.nonzero(keep)[0]
    if len(idx) < 3:
        return idx
    rng = rng if rng is not None else np.random.default_rng(0)
    xy = pts_laser[idx, :2]
    best = None
    for _ in range(100):
        i, j = rng.choice(len(xy), 2, replace=False)
        dv = xy[j] - xy[i]
        L = float(np.hypot(*dv))
        if L < 1e-3:
            continue
        inl = np.abs((xy - xy[i]) @ np.array([-dv[1], dv[0]]) / L) < line_tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 2:
        return idx[:0]
    c = xy[best].mean(0)
    nrm = np.linalg.svd(xy[best] - c)[2][1]                      # the line's normal (total least squares)
    return idx[np.abs((xy - c) @ nrm) < line_tol]


def coarse_search(planes, T0, objp, margin=BOARD_MARGIN, gate=0.20, yaw=(-20.0, 20.0, 2.0), pitch=(-10.0, 10.0, 2.5),
                  shift=(-0.12, 0.12, 0.04), max_pts=400, window=0.6):
    """For a ROUGH first guess: try the camera turned (yaw, pitch) and shifted (sideways, up / down) around it and
    keep the candidate that puts the most LiDAR points on the boards (plane gate + board area). Ties -> the
    smallest change. -> T, (dyaw, dpitch [deg], dx, dy [m] in the camera frame), (poses, points)."""
    lo, hi = objp[:, :2].min(0) - margin, objp[:, :2].max(0) + margin
    S, Rs, ts, idx = [], [], [], []
    for k, pl in enumerate(planes):
        s = pl['scan']
        # the LiDAR sits within ~0.3 m of the camera: a board the camera sees at distance d is within d +- 0.6 m
        s = s[np.abs(np.hypot(s[:, 0], s[:, 1]) - np.linalg.norm(pl['t_cb'])) < window]
        if len(s) > max_pts:
            s = s[np.linspace(0, len(s) - 1, max_pts).astype(int)]
        S.append(s)
        idx.append(np.full(len(s), k))
        Rs.append(pl['R_cb'])
        ts.append(pl['t_cb'])
    S, idx, Rs, ts = np.vstack(S), np.concatenate(idx), np.array(Rs), np.array(ts)
    Rp, tp = Rs[idx], ts[idx]                                  # per point: its board's pose
    shifts = np.arange(shift[0], shift[1] + 1e-9, shift[2])
    RsT2 = np.transpose(Rs, (0, 2, 1))[:, :, :2]               # per pose: a camera-frame shift (dx, dy, 0) in the board frame
    best = None
    for dyaw in np.arange(yaw[0], yaw[1] + 1e-9, yaw[2]):
        for dpitch in np.arange(pitch[0], pitch[1] + 1e-9, pitch[2]):
            a, b = math.radians(dpitch), math.radians(dyaw)
            Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
            Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
            Rd = Rx @ Ry                                       # optical frame: x = right (pitch), y = down (yaw)
            Xb0 = np.einsum('ij,ijk->ik', S @ (Rd @ T0[:3, :3]).T + Rd @ T0[:3, 3] - tp, Rp)
            for dx in shifts:
                for dy in shifts:
                    Xb = Xb0 + (RsT2 @ np.array([dx, dy]))[idx]
                    m = (np.abs(Xb[:, 2]) < gate) & np.all(Xb[:, :2] >= lo, 1) & np.all(Xb[:, :2] <= hi, 1)
                    n_poses = int(np.count_nonzero(np.bincount(idx[m], minlength=len(planes)) >= 3))
                    score = (n_poses, int(m.sum()), -(abs(dyaw) + abs(dpitch) + 100 * (abs(dx) + abs(dy))))
                    if best is None or score > best[0]:
                        best = (score, pj._T(Rd, (dx, dy, 0.0)) @ T0, (float(dyaw), float(dpitch), float(dx), float(dy)))
    return best[1], best[2], best[0][:2]


def bootstrap(obs, T, n=200, delta=0.01, seed=1, T_base_laser=None):
    """Re-solve n times with the board poses resampled (with replacement): the spread of the answer = how well the
    data determine each number (honest: a whole pose's error moves together, like in reality)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(obs), len(obs))
        Tb, _ = refine([obs[i] for i in pick], T, delta, iters=40)
        out.append(camera_mount(T_base_laser @ pj.invert(Tb)) if T_base_laser is not None else
                   np.concatenate([Tb[:3, 3], np.degrees(np.array([Tb[2, 1], Tb[0, 2], Tb[1, 0]]))]))
    out = np.array(out)
    return {'std': out.std(0), 'samples': out}


def calibrate(poses, cam, T0_cam_laser, T_base_laser=None, objp=None, margin=BOARD_MARGIN, gates=(0.20, 0.06, 0.03),
              min_points=6, max_pnp_rms=1.0, delta=0.01, n_boot=200, seed=0, min_pose_share=0.75,
              min_pose_share_search=0.5, max_move=0.25, max_turn=25.0):
    """poses: [{'corners': [54, 2] pixels, 'scan_points': [M, 3] laser-frame returns (1 or more scans)}, ...]
    cam: projection.CameraModel. T0_cam_laser: the first guess (CAD / tape measure, sensor_mounts_car.yaml).
    T_base_laser: the LiDAR's mount (to report the camera's mount in base_link + its bootstrap spread).
    -> dict(T (T_cam_from_laser), mount, std, weak, info, obs, planes, log). Raises ValueError when it cannot work."""
    objp = board_object_points() if objp is None else objp
    rng = np.random.default_rng(seed)
    planes, log = [], []
    for k, p in enumerate(poses):
        res = board_pose_pnp(p['corners'], cam, objp)
        if res is None or res[2] > max_pnp_rms:
            log.append(f'pose {k}: board pose unusable (PnP rms {res[2] if res else float("nan"):.2f} px) -- skipped')
            continue
        R_cb, t_cb, rms = res
        n, d = plane_from_pose(R_cb, t_cb)
        planes.append(dict(pose=k, R_cb=R_cb, t_cb=t_cb, n=n, d=d, pnp_rms=rms,
                           scan=np.asarray(p['scan_points'], float).reshape(-1, 3)))
    T = np.array(T0_cam_laser, float)
    if len(planes) < 3:
        raise ValueError(f'only {len(planes)} usable board poses (need >= 3, better 12-20)')
    T, offs, score = coarse_search(planes, T, objp, margin, gates[0])
    log.append(f'coarse search around the first guess: best = turned yaw {offs[0]:+.0f} / pitch {offs[1]:+.1f} deg, '
               f'shifted {offs[2] * 100:+.0f} / {offs[3] * 100:+.0f} cm (camera frame) -> {score[0]} of {len(planes)} '
               f'poses, {score[1]} points on the boards')
    # SAFETY 1: near the first guess the boards must show up in most poses (a LiDAR yaw 180 deg off fits a few
    # poses with the WALL BEHIND the car -- job 2026-10-14)
    if score[0] < max(3, math.ceil(min_pose_share_search * len(planes))):
        raise ValueError(f'near the first guess the boards show up in the LiDAR in only {score[0]} of {len(planes)} '
                         f'poses: the first guess is too far off (check the LiDAR yaw in the mount YAML -- the '
                         f'box-straight-ahead test) or the LiDAR scan does not cross the boards')
    obs = []
    for rnd, gate in enumerate(gates):
        obs = []
        for pl in planes:
            idx = associate(pl['scan'], pl['R_cb'], pl['t_cb'], T, objp, margin, gate, max(gate / 2, 0.02), rng)
            if len(idx) >= min_points:
                obs.append(dict(pose=pl['pose'], n=pl['n'], d=pl['d'], pts=pl['scan'][idx], pnp_rms=pl['pnp_rms']))
        if len(obs) < 3:
            raise ValueError(f'only {len(obs)} board poses have >= {min_points} LiDAR points on the board (gate '
                             f'{gate} m): the first guess is too far off (check the LiDAR yaw in the mount YAML: '
                             f'box-straight-ahead test) or the scan plane does not cross the boards')
        T_ref, info = refine(obs, T, delta)
        if rnd == 0:
            T_lin, cond = linear_init(obs)
            T_lin_ref, info_lin = refine(obs, T_lin, delta)
            use_lin = info_lin['cost'] < info['cost'] * 0.999
            log.append(f'linear (Zhang-Pless) start: condition {cond:.1e}, cost {info_lin["cost"]:.3e} vs from the '
                       f'first guess {info["cost"]:.3e} -> {"linear" if use_lin else "first guess"}')
            if use_lin:
                T_ref, info = T_lin_ref, info_lin
        T = T_ref
        log.append(f'round {rnd + 1}: gate {gate:.2f} m, {len(obs)} poses, {info["points"]} LiDAR points, '
                   f'rms {info["rms"] * 1000:.1f} mm')
    # SAFETY 2: one camera position must explain (nearly) every pose
    if len(obs) < max(3, math.ceil(min_pose_share * len(planes))):
        raise ValueError(f'only {len(obs)} of {len(planes)} board poses fit one camera position (want >= '
                         f'{min_pose_share:.0%}): not trustworthy -- wrong first guess (LiDAR yaw?), a moving board, '
                         f'or the scan missing the boards')
    # SAFETY 3: the answer must be near the measured first guess (tape measure / CAD are not 25 cm or 25 deg off)
    dT = pj.invert(np.asarray(T0_cam_laser, float)) @ T
    move, turn = float(np.linalg.norm(dT[:3, 3])), rot_angle_deg(dT[:3, :3])
    log.append(f'answer vs the first guess: moved {move * 100:.1f} cm, turned {turn:.1f} deg')
    if move > max_move or turn > max_turn:
        raise ValueError(f'the answer is {move * 100:.0f} cm / {turn:.0f} deg away from the first guess (limit '
                         f'{max_move * 100:.0f} cm / {max_turn:.0f} deg): check the mount YAML and the data')
    boot = bootstrap(obs, T, n_boot, delta, seed + 1, T_base_laser) if n_boot else None
    out = dict(T=T, info=info, obs=obs, planes=planes, log=log, boot=boot, mount=None, std=None, weak=[])
    if T_base_laser is not None:
        out['mount'] = camera_mount(T_base_laser @ pj.invert(T))
        if boot is not None:
            out['std'] = boot['std']
            names = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
            out['weak'] = [nm for i, nm in enumerate(names) if boot['std'][i] > (WEAK_M if i < 3 else WEAK_DEG)]
    return out


def report(res, truth_mount=None):
    """The result as text lines (per pose, the camera mount +- spread, WEAK flags, the YAML lines)."""
    L = [f"  {'pose':>4} {'PnP rms px':>10} {'LiDAR pts':>9} {'plane dist m':>12} {'residual mm':>11}"]
    for o, rb in zip(res['obs'], res['info']['per_board_rms']):
        L.append(f"  {o['pose']:4d} {o['pnp_rms']:10.2f} {len(o['pts']):9d} {o['d']:12.2f} {rb * 1000:11.1f}")
    L.append(f"  all: {res['info']['points']} LiDAR points on {len(res['obs'])} boards, residual rms "
             f"{res['info']['rms'] * 1000:.1f} mm")
    if res['mount'] is not None:
        names = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
        units = ('m', 'm', 'm', 'deg', 'deg', 'deg')
        L.append('  camera mount (camera_link in base_link):')
        for i, nm in enumerate(names):
            s = f"    {nm:>5} {res['mount'][i]:+9.4f} {units[i]:3}"
            if res['std'] is not None:
                s += f"  +- {res['std'][i]:.4f}" + ('  WEAK -- not determined by these poses' if nm in res['weak']
                                                    else '')
            if truth_mount is not None:
                s += f"   (truth {truth_mount[i]:+.4f}, error {res['mount'][i] - truth_mount[i]:+.4f})"
            L.append(s)
        m = res['mount']
        L.append(f'  sensor_mounts_car.yaml -> camera: {{x: {m[0]:.3f}, y: {m[1]:.3f}, z: {m[2]:.3f}, '
                 f'roll: {m[3]:.2f}, pitch: {m[4]:.2f}, yaw: {m[5]:.2f}}}')
    return L
