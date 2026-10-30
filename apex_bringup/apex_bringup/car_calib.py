"""car_calib.py -- run the calibration tools FOR THE CAR: every input and output path and every car number comes from
car.yaml, so a car calibration can never land in the simulator's folder (~/.ros) or use the simulator car's lf / lr.

WHY (2026-10-30)
    state_estimation's calibration tools (wheel_calib, steer_calib, tools/slip_calib.py) default to ~/.ros/... --
    the SIMULATOR's calibrations. Run on the car with their defaults, the result would be written where the car never
    reads it (car.launch.py reads calibration_dir), and steer_calib / slip_calib would compute with the F1TENTH car's
    wheelbase. The tools stay unchanged (the simulator needs those defaults); on the car they are started through
    apex_bringup:
        ros2 launch apex_bringup car_calib.launch.py tool:=wheel true_distance:=10.0   -> ekf_speed_scale.yaml
        ros2 launch apex_bringup car_calib.launch.py tool:=steer                        -> steering_calibration.yaml
        ros2 run apex_bringup car_slip_calib <recordings> --map <track map.yaml>       -> vehicle_slip.yaml + the
                                                                                          slip_k to put in car.yaml
    Order on the real car: wheel -> wheel delay (tools/wheel_delay_calib.py, prints wheel_delay_s for car.yaml) ->
    steer -> slip.
WHERE THINGS END UP
    speed scale + steering calibration: files in calibration_dir, read automatically by both EKFs and state_estimate.
    slip constant K: car.yaml vehicle.slip_k (+ beta_model: slip) -- a tuning constant a person decides to use, so it
    lives in the checked settings file; vehicle_slip.yaml keeps the fit's details as a record.
No ROS imports at module level (tests).
"""
import os

from apex_bringup.car_nodes import calib_files


def wheel_calib_spec(cfg, true_distance):
    if not true_distance or float(true_distance) <= 0.0:
        raise ValueError('wheel calibration needs the TAPE-MEASURED distance: true_distance:=10.0 (metres)')
    return {'package': 'state_estimation', 'executable': 'wheel_calib', 'name': 'wheel_calib',
            'params': {'true_distance': float(true_distance), 'scale_file': calib_files(cfg)['scale_file'],
                       'write_file': True, 'use_sim_truth': False}}


def steer_calib_spec(cfg):
    f = calib_files(cfg)
    return {'package': 'state_estimation', 'executable': 'steer_calib', 'name': 'steer_calib',
            'params': {'lf': cfg['lf'], 'lr': cfg['lr'], 'scale_file': f['scale_file'],
                       'out_file': f['steer_calib_file'], 'use_sim_truth': False}}


def slip_calib_argv(cfg, recordings, resolution):
    """Arguments for state_estimation tools/slip_calib.py with the car's numbers (resolution = map cell [m])."""
    f = calib_files(cfg)
    return [str(r) for r in recordings] + [
        '--lf', repr(cfg['lf']), '--lr', repr(cfg['lr']), '--scale-file', f['scale_file'],
        '--amcl-offset', repr(resolution / 2.0), '--wheel-delay', repr(cfg['wheel_delay_s']),
        '--write', f['slip_file']]


def find_slip_calib(state_estimation_prefix):
    """tools/slip_calib.py is not installed: look next to the install space (<ws>/install/<pkg> -> <ws>/src/...)."""
    env = os.environ.get('APEX_SLIP_CALIB', '').strip()
    cands = [env] if env else []
    ws = os.path.dirname(os.path.dirname(os.path.abspath(state_estimation_prefix)))
    cands.append(os.path.join(ws, 'src', 'state_estimation', 'tools', 'slip_calib.py'))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def main_slip(argv=None):
    """car_slip_calib <recordings...> --map <map.yaml> [--config car.yaml] [--dry-run]"""
    import argparse
    import subprocess
    import sys

    from apex_bringup.car_config import CarConfigError, load_car_config
    from apex_bringup.car_nodes import map_resolution
    ap = argparse.ArgumentParser(prog='car_slip_calib', description=main_slip.__doc__)
    ap.add_argument('recordings', nargs='+')
    ap.add_argument('--map', required=True, help="the track's map YAML (AMCL's half-cell offset)")
    ap.add_argument('--config', default=None)
    ap.add_argument('--dry-run', action='store_true', help='only print the command')
    a = ap.parse_args(argv)
    try:
        cfg, warnings = load_car_config(a.config)
    except (CarConfigError, OSError) as e:
        print(f'CAR CONFIG REFUSED\n{e}')
        return 1
    for w in warnings:
        print(f'WARNING: {w}')
    os.makedirs(cfg['calibration_dir'], exist_ok=True)
    args = slip_calib_argv(cfg, a.recordings, map_resolution(a.map))
    from ament_index_python.packages import get_package_prefix
    tool = find_slip_calib(get_package_prefix('state_estimation'))
    cmd = [sys.executable, tool or '<state_estimation>/tools/slip_calib.py'] + args
    print('command: ' + ' '.join(cmd))
    if a.dry_run:
        return 0
    if tool is None:
        print('slip_calib.py not found (set APEX_SLIP_CALIB=<path>)')
        return 1
    rc = subprocess.call(cmd)
    if rc == 0:
        print(f"\nNEXT: if the fit is clear, set in car.yaml  vehicle.slip_k: <K above>  and  beta_model: slip  "
              f"(details kept in {calib_files(cfg)['slip_file']})")
    return rc
