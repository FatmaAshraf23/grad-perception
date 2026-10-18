"""The car's SIDESLIP at the CG -- one definition for ekf_node (both EKFs) and state_estimate (2026-10-18).

WHAT SIDESLIP IS
    The CG does not always move where the car points.  beta = the angle between the heading and the
    CG's velocity:  vy = vx * tan(beta).  The EKF moves its position along heading + beta, and
    StateEstimate reports vx = v cos(beta), vy = v sin(beta) to control.

beta_model 'steer'  (2026-09, the old model)
    No-slip KINEMATIC bicycle from the steering angle:   beta = atan(lr / L * tan(delta))
    Needs the steering angle (servo trim, the commanded vs the actual wheel angle) and assumes the tyres
    roll exactly where they point.

beta_model 'slip'   (2026-10-18; the DEFAULT of ekf_node / state_estimate / the launch files since then)
    From the GYRO yaw rate r and the speed vx, with the rear tyres' slip:
        vy = r * (a - K * vx^2)          beta = atan(vy / vx)
    a = lr (where the rear axle is, seen from the CG), K = slip coefficient [s^2/m].
    Why (steady cornering, linear tyres):
      * no slip at the rear axle -> its sideways velocity is 0 -> at the CG  vy = lr * r   (K = 0)
      * a tyre only makes a sideways force F when it slips by an angle alpha = F / C (cornering
        stiffness C).  The rear axle carries the share lf / L of the centripetal force m * a_y, with
        a_y = vx * r  ->  alpha_r = m * lf * vx * r / (L * C_r)  ->  the rear axle slides sideways
        (outwards) with vx * alpha_r  ->  vy = lr * r - (m * lf / (L * C_r)) * vx^2 * r.
      So K = m * lf / (L * C_r).  It grows with speed squared: invisible at walking speed, ~0.2 m/s of
      vy at 4 m/s^2 of cornering.
    Simulator (closed1, 2026-10-18): fit on the truth K = 0.0170 (physics from the gym's tyre data:
    0.0178); vy error p95 0.224 -> 0.052 m/s; offline EKF replay: position p95 16-22 -> 4-9 cm at
    3-6 m/s (= as good as the TRUE sideslip), inside 2 sigma 54-63 -> 100 %.
    Real car: K from `slip_calib` (a recording with fast corners: the K the AMCL fixes agree with
    best), stored in the calibration file (slip_file).  No file -> K = 0 (gyro, no slip) with a
    warning: at low speed as good as 'steer' and it does not need the steering calibration.
"""
import math
import os


class Sideslip:
    MODELS = ('steer', 'slip')

    def __init__(self, model='steer', lf=0.15875, lr=0.17145, slip_k=0.0, slip_a=None, v_min=0.05):
        if model not in self.MODELS:
            raise ValueError(f"beta_model must be one of {self.MODELS}, not {model!r}")
        if slip_k < 0.0 or slip_k > 0.2:
            raise ValueError(f'slip_k {slip_k} outside 0..0.2 s^2/m')
        self.model = model
        self.lf, self.lr = float(lf), float(lr)
        self.k = float(slip_k)
        self.a = float(lr if slip_a is None else slip_a)
        self.v_min = float(v_min)

    @property
    def kinematic(self):
        """True if vy comes from a no-slip model (StateEstimate FLAG_VY_KINEMATIC)."""
        return self.model == 'steer' or self.k == 0.0

    def vy(self, vx, r, steer=0.0):
        """lateral velocity of the CG [m/s] for the forward speed vx [m/s] (signed) and
        yaw rate r [rad/s] (gyro, bias removed); steer [rad] only for model 'steer'"""
        return vx * math.tan(self.beta(vx, r, steer))

    def beta(self, vx, r, steer=0.0):
        """sideslip angle [rad]; |beta| < 90 deg, so a negative vx (reversing) keeps its sign"""
        if self.model == 'steer':
            return math.atan(self.lr / (self.lf + self.lr) * math.tan(steer))
        if abs(vx) < self.v_min:
            return 0.0
        return math.atan(r * (self.a - self.k * vx * vx) / vx)

    def describe(self):
        if self.model == 'steer':
            return f'sideslip: kinematic from the steering angle (lr {self.lr:.4f}, L {self.lf + self.lr:.4f} m)'
        return (f'sideslip: from the gyro, vy = r * ({self.a:.4f} - {self.k:.5f} * vx^2)'
                + (' (K = 0: no tyre slip)' if self.k == 0.0 else ''))


def load_slip_file(path):
    """(K, sigma_K, problem) from a slip calibration file (key: value lines, written by slip_calib).
    problem = None if fine, else a message; K is None if there is no usable value."""
    path = os.path.expanduser(str(path).strip())
    if not path or path.lower() == 'none':
        return None, None, 'no slip calibration file configured'
    try:
        vals = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                if ':' in line and not line.lstrip().startswith('#'):
                    key, val = line.split(':', 1)
                    vals[key.strip()] = val.strip()
        k = float(vals['slip_k'])
        sig = float(vals.get('slip_k_sigma', 'nan'))
        if not 0.0 <= k <= 0.2:
            raise ValueError(f'slip_k {k} outside 0..0.2')
    except FileNotFoundError:
        return None, None, f'no slip calibration file {path} (create it with slip_calib)'
    except (KeyError, ValueError) as e:
        return None, None, f'bad slip calibration file {path}: {e}'
    return k, sig, None


def sideslip_from_params(model, lf, lr, slip_k, slip_file):
    """Sideslip for the node parameters + a log line.  slip_k >= 0: that value; slip_k < 0: from
    slip_file, and K = 0 (no tyre slip) if the file is missing -- then the log line is a warning."""
    if model == 'slip' and slip_k < 0.0:
        k, sig, problem = load_slip_file(slip_file)
        if k is None:
            s = Sideslip(model, lf, lr, 0.0)
            return s, f'{problem} -> K = 0 (no tyre slip; fine at low speed, too much vy in fast corners)', True
        s = Sideslip(model, lf, lr, k)
        return s, s.describe() + f' (K from {os.path.expanduser(slip_file)}, +-{sig:.4f})', False
    s = Sideslip(model, lf, lr, max(slip_k, 0.0))
    return s, s.describe(), False
