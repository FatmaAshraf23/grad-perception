"""car_config.py -- read and CHECK the real car's settings (config/car.yaml) before anything starts.

No ROS imports: the launch file, tools and tests use it.

WHY a checker
    A wrong setting on the car is worse than a crash: the estimator runs, publishes "OK" and is quietly wrong.
    So the launch stops if car.yaml has
      - an UNKNOWN key (a typo like `slip_ro` would otherwise be ignored and the default used),
      - a value of the wrong TYPE ("true" in quotes is a string, not a bool) or an impossible SIZE
        (lf: 160 = millimetres typed as metres),
      - ANY simulator / test switch (allow_test_faults, test_*, sim_*): the car's launch never passes them,
        and they cannot be added here,
      - a calibration folder that is the simulator's (~/.ros).
    Every problem is listed at once (fix them all in one go). Placeholders (names in not_measured) are allowed but
    WARNED about at every start.
"""
import math
import os

import yaml

# key -> (type, min, max) for numbers; (type, allowed values) for strings; (bool,) for switches
SCHEMA = {
    'vehicle': {
        'lf': (float, 0.05, 0.5),
        'lr': (float, 0.05, 0.5),
        'imu_yaw_deg': (float, -180.0, 180.0),
        'beta_model': (str, ('steer', 'slip')),
        'slip_k': (float, 0.0, 0.1),
    },
    'timing': {
        'wheel_delay_s': (float, -1.0, 0.2),
        'slip_lag_s': (float, 0.0, 0.2),
        'imu_gap_s': (float, 0.005, 0.2),
    },
    'estimator': {
        'slip_check': (bool,),
        'slip_use_observer': (bool,),
        'slip_r0': (float, 0.05, 1.0),
        'imu_gap_predict': (bool,),
        'imu_gap_kus': (float, -0.05, 0.05),
        'zupt_gate': (float, 0.0, 1000.0),
        'zupt_turn_max_s': (float, 0.0, 60.0),
        'map_check': (bool,),
        'amcl_corridor_sigma': (float, 0.0, 0.5),
        'amcl_min_sigma_pos': (float, 0.01, 0.5),
    },
    'lidar': {
        'serial_port': (str, None),
        'serial_baudrate': (int, (115200, 256000, 1000000)),
        'scan_mode': (str, ('Standard', 'Express', 'Boost', 'Sensitivity', 'Stability')),
        'scan_frequency': (float, 1.0, 20.0),
        'deskew_mode': (str, ('full', 'restamp', 'off')),
        'deskew_time_order': (str, ('reverse', 'forward')),
    },
}
TOP_KEYS = ('not_measured', 'calibration_dir') + tuple(SCHEMA)
ALL_NAMES = {k: sec for sec, keys in SCHEMA.items() for k in keys}
FORBIDDEN_WORDS = ('allow_test_faults', 'test_', 'sim_', 'fake_')


class CarConfigError(ValueError):
    """car.yaml cannot be used; the message lists every problem."""


def default_path():
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('apex_bringup'), 'config', 'car.yaml')


def _forbidden(key):
    k = str(key).lower()
    return any(k.startswith(w) or w in k for w in FORBIDDEN_WORDS)


def _check_value(sec, key, v, rule):
    """'' if ok, else the problem."""
    kind = rule[0]
    if kind is bool:
        return '' if isinstance(v, bool) else f'{sec}.{key}: must be true or false (got {v!r})'
    if kind is float or kind is int:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return f'{sec}.{key}: must be a number (got {v!r})'
        if kind is int:
            if v != int(v) or int(v) not in rule[1]:
                return f'{sec}.{key} = {v}: must be one of {list(rule[1])}'
            return ''
        lo, hi = rule[1], rule[2]
        if not lo <= v <= hi:
            return f'{sec}.{key} = {v}: outside {lo} .. {hi} (units: m, s, rad; *_deg in degrees)'
        return ''
    if not isinstance(v, str) or not v.strip():
        return f'{sec}.{key}: must be a text value (got {v!r})'
    if rule[1] is not None and v not in rule[1]:
        return f'{sec}.{key} = {v!r}: must be one of {list(rule[1])}'
    if key == 'serial_port' and not v.startswith('/dev/'):
        return f'{sec}.{key} = {v!r}: must be a device path (/dev/...)'
    return ''


def check(doc, path='car.yaml'):
    """(cfg, warnings) for a parsed car.yaml; raises CarConfigError listing EVERY problem."""
    if not isinstance(doc, dict):
        raise CarConfigError(f'{path}: not a YAML mapping')
    problems, warnings = [], []

    def scan_forbidden(d, where):
        for k, v in d.items():
            if _forbidden(k):
                problems.append(f'{where}{k}: a SIMULATOR / TEST setting -- never allowed on the car')
            if isinstance(v, dict):
                scan_forbidden(v, f'{where}{k}.')
    scan_forbidden(doc, '')

    for k in doc:
        if k not in TOP_KEYS and not _forbidden(k):
            problems.append(f'{k}: unknown key (typo?) -- allowed: {", ".join(TOP_KEYS)}')
    cfg = {}
    for sec, rules in SCHEMA.items():
        d = doc.get(sec)
        if not isinstance(d, dict):
            problems.append(f'{sec}: missing section')
            continue
        for k in d:
            if k not in rules and not _forbidden(k):
                problems.append(f'{sec}.{k}: unknown key (typo?)')
        for k, rule in rules.items():
            if k not in d:
                problems.append(f'{sec}.{k}: missing')
                continue
            p = _check_value(sec, k, d[k], rule)
            if p:
                problems.append(p)
            else:
                cfg[k] = float(d[k]) if rule[0] is float else d[k]

    cd = doc.get('calibration_dir')
    if not isinstance(cd, str) or not cd.strip():
        problems.append('calibration_dir: missing (a folder for THIS car\'s calibrations, e.g. ~/.ros/apex_car)')
    else:
        full = os.path.normpath(os.path.expanduser(cd.strip()))
        if full == os.path.normpath(os.path.expanduser('~/.ros')):
            problems.append('calibration_dir: ~/.ros holds the SIMULATOR\'s calibrations -- use a folder of its own '
                            '(e.g. ~/.ros/apex_car)')
        cfg['calibration_dir'] = full

    nm = doc.get('not_measured', [])
    if nm is None:
        nm = []
    if not isinstance(nm, list):
        problems.append('not_measured: must be a list, e.g. [lf, lr] or []')
        nm = []
    for n in nm:
        if n not in ALL_NAMES:
            problems.append(f'not_measured: {n!r} is not a setting name (typo?)')
    if len(set(nm)) != len(nm):
        problems.append('not_measured: a name is listed twice')
    cfg['not_measured'] = [n for n in nm if n in ALL_NAMES]

    if problems:
        raise CarConfigError(f'{path}: {len(problems)} problem(s):\n  - ' + '\n  - '.join(problems))

    # --- warnings: allowed, but must be seen at every start ---
    for n in cfg['not_measured']:
        warnings.append(f'{n} = {cfg[n]} is a PLACEHOLDER (not measured on this car)')
    if cfg['lf'] + cfg['lr'] < 0.15 or cfg['lf'] + cfg['lr'] > 0.6:
        warnings.append(f"wheelbase lf + lr = {cfg['lf'] + cfg['lr']:.3f} m is unusual for a 1:10 car")
    if cfg['beta_model'] == 'slip' and cfg['slip_k'] == 0.0:
        warnings.append('beta_model slip with slip_k 0 = the same as steer (fit slip_k with slip_calib first)')
    if cfg['imu_gap_predict']:
        warnings.append('imu_gap_predict is ON: only if control has accepted contract v0.4 (wheel-based stamps)')
    if cfg['slip_use_observer'] and 'slip_r0' in cfg['not_measured']:
        warnings.append('slip_use_observer is ON with an unmeasured slip_r0: normal tyre slip may switch the EKFs '
                        'to the IMU speed')
    if not cfg['slip_check']:
        warnings.append('slip_check is OFF: wheel spin / skid will not be flagged')
    if not cfg['map_check']:
        warnings.append('map_check is OFF: a wrong pose will not be flagged (contract v0.3)')
    return cfg, warnings


def load_car_config(path=None):
    """(cfg, warnings) from car.yaml (default: the installed apex_bringup/config/car.yaml)."""
    path = os.path.expanduser(path) if path else default_path()
    with open(path) as f:
        doc = yaml.safe_load(f)
    cfg, warnings = check(doc, path)
    cfg['_path'] = path
    return cfg, warnings


def main(argv=None):
    """check_car_config [path]: print the checked settings + warnings; exit 1 on a problem."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    try:
        cfg, warnings = load_car_config(argv[0] if argv else None)
    except (CarConfigError, OSError, yaml.YAMLError) as e:
        print(f'CAR CONFIG REFUSED\n{e}')
        return 1
    print(f"car config OK: {cfg['_path']}")
    for sec, keys in SCHEMA.items():
        print(f'  {sec}: ' + ', '.join(f'{k}={cfg[k]}' for k in keys))
    print(f"  calibration_dir: {cfg['calibration_dir']}")
    for w in warnings:
        print(f'  WARNING: {w}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
