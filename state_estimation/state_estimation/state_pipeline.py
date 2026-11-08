#!/usr/bin/env python3
"""ONE process for the state chain: ekf_local + ekf_global + state_estimate.

WHY (CPU, 2026-10-03, jobs 035-036)
    In Python ROS 2 the cost is mostly the MESSAGES, not the math: each node that
    receives the IMU (100 Hz) and the wheel/steering message (50 Hz) pays for
    unpacking every message and waking up its executor. As 3 processes, the same
    IMU and wheel messages were received 3 times, and /odom/ekf + /ekf/nis went
    from ekf_global to state_estimate as messages: ~70 % of one laptop core for the
    three -- and the Pi 4 must also run planning and MPC on its 4 cores.

WHAT CHANGES -- AND WHAT DOES NOT
    same    the classes (EkfNode twice: 'ekf_local' and 'ekf_global';
            StateEstimateNode: 'state_estimate'), their parameters, the math, the
            output topics and the TF -> AMCL, eval_logger and control notice nothing.
    new     ONE subscription to /imu/data and ONE to /vehicle/measured. Each
            message goes to the three as a plain function call, state_estimate
            first (its freshness check then sees the newest IMU / wheel sample).
    new     ekf_global hands every odometry it publishes, the NIS of every accepted
            AMCL fix and (contract v0.3) every map-check result DIRECTLY to
            state_estimate (all are still published too, for eval_logger and tools). /state_estimate therefore leaves
            in the same callback as the IMU sample: no extra hop between processes.
    new     one single-threaded executor: callbacks never run at the same time, so
            nothing needs a lock.
    risk    one process = one failure domain. An error inside state_estimate is
            caught and logged here so the EKFs keep running (AMCL needs ekf_local's
            TF); /state_estimate then stops and control's staleness check goes LOST,
            exactly as if a separate state_estimate process had crashed.

PARAMETERS
    From a params file with one section per node name:
        /ekf_local:      {ros__parameters: {...}}
        /ekf_global:     {ros__parameters: {...}}
        /state_estimate: {ros__parameters: {...}}   (optional)
    sim_localization.launch.py pipeline:=merged writes it. Never start this
    executable with a node-name remap (__node:=...): it would rename all three.
    Argument --no-state-estimate: only the two EKFs.

STAMP-ORDERED INPUTS (2026-11-07, input_order.py, parameter order_inputs on ekf_global, default ON):
    in a backlog the executor takes one message per queue per round, so the 50 Hz wheel queue ran
    ahead of the 100 Hz IMU queue in time and the EKFs predicted with a wheel speed from the future
    (F07: up to 3.5 m/s wrong). ON: a wheel message waits until the IMU is processed up to its stamp.

Real car: identical -- this is what should run on the Pi (inputs from the STM32
bridge, the MPU-6050 driver and rplidar instead of the simulator).
STUCK WHEEL (2026-11-08, F12, wheel_stuck.py, parameter wheel_stuck_check on ekf_global, default ON):
  a dead encoder that keeps SENDING 0 while the car drives (job 252: 55 cm after 1 s, 16 m after 10 s with a false
  "parked"). Such a 0 is DROPPED here, before the EKFs and state_estimate: to them it is missing wheel data, so the
  tested F11 path (wheel_gap.py) takes over -- IMU speed, never parked, honest uncertainty, WHEEL_STALE ->
  DEGRADED -> LOST after wheel_lost_s. Back after 3 plausible non-zero samples. false = the old behaviour (TEST ONLY).
"""
import sys
import time
import traceback

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Imu

from state_estimation.ekf_node import EkfNode
from state_estimation.input_order import InputOrder
from state_estimation.wheel_stuck import StuckWheel
from state_estimation.state_estimate_node import StateEstimateNode

try:
    from rclpy.executors import ExternalShutdownException
except ImportError:                              # older rclpy
    class ExternalShutdownException(Exception):
        pass


def guarded(node, fn, what):
    """fn(msg), but an exception is logged (the first time with its traceback), never raised."""
    errors = [0]

    def call(msg):
        try:
            fn(msg)
        except Exception:                        # noqa: BLE001 -- deliberately broad, see docstring
            errors[0] += 1
            if errors[0] == 1 or errors[0] % 1000 == 0:
                node.get_logger().error(f'{what} failed ({errors[0]} times so far):\n'
                                        + traceback.format_exc())
    return call


def main(args=None):
    rclpy.init(args=args)
    with_se = '--no-state-estimate' not in remove_ros_args(args=sys.argv)[1:]

    local = EkfNode('ekf_local', own_inputs=False)
    glob = EkfNode('ekf_global', own_inputs=False)
    nodes = [local, glob]
    se_imu = se_wheel = None
    if with_se:
        se = StateEstimateNode(own_inputs=False)
        nodes.append(se)
        glob.odom_hooks.append(guarded(se, se.odom_cb, 'state_estimate.odom_cb'))
        glob.fix_hooks.append(guarded(se, se.fix_cb, 'state_estimate.fix_cb'))
        glob.map_check_hooks.append(guarded(se, se.map_check_cb, 'state_estimate.map_check_cb'))
        glob.slip_hooks.append(guarded(se, se.slip_cb, 'state_estimate.slip_cb'))      # Fix 3, 2026-10-26
        glob.gap_hooks.append(guarded(se, se.imu_gap_cb, 'state_estimate.imu_gap_cb'))  # Fix 2, 2026-10-27
        se_imu = guarded(se, se.imu_cb, 'state_estimate.imu_cb')
        se_wheel = guarded(se, se.wheel_cb, 'state_estimate.wheel_cb')

    # TEST ONLY (2026-10-20, fault scenarios for control): extra work per IMU message = a busy CPU
    # (job 167 r60: the laptop was overloaded and this process fell 0.34 s behind). Runtime:
    # ros2 param set /ekf_global test_lag_ms 12.0   (> 10 ms per 100 Hz message -> the process falls behind)
    # LOCKED (2026-10-24): only when ekf_global's allow_test_faults is true (the simulator launch files);
    # otherwise test_lag_ms is not declared and on_imu does not even read it (no cost on the car).
    test_switches = glob.allow_test_faults
    if test_switches:
        glob.declare_parameter('test_lag_ms', 0.0)
    lag_on = [0.0]

    # STAMP-ORDERED INPUTS (estimator-stall fix A, 2026-11-07, input_order.py): in a backlog the executor
    # drains the 50 Hz wheel queue twice as fast in time as the 100 Hz IMU queue -> the EKFs predicted with a
    # wheel speed from the FUTURE (F07: speed error up to 3.5 m/s). ON: a wheel message waits until the IMU
    # has been processed up to its stamp (released anyway after order_wait_s without any IMU = a dropout).
    # Default ON since 2026-11-07 (jobs 239-240: no change in normal driving); false = the old arrival order.
    glob.declare_parameter('order_inputs', True)
    glob.declare_parameter('order_wait_s', 0.025)
    order = None
    if glob.get_parameter('order_inputs').value:
        order = InputOrder(wait_s=float(glob.get_parameter('order_wait_s').value))
        glob.get_logger().info(
            f'stamp-ordered inputs ON: a wheel message ahead of the IMU waits for it (released after '
            f'{1000 * order.wait_s:.0f} ms without any IMU message = a dropout)')
        glob.create_timer(5.0, lambda: glob.get_logger().info(order.report()))
    else:
        glob.get_logger().info('stamp-ordered inputs OFF (order_inputs false): messages in arrival order (TEST ONLY: the old behaviour)')

    def stamp_s(msg):
        return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec

    # STUCK WHEEL (F12, 2026-11-08, wheel_stuck.py): a 0 while the car drives is a dead encoder, not a stop ->
    # dropped HERE (after the stamp order: the samples arrive in time order), so the EKFs and state_estimate see
    # missing wheel data and Fix 4's wheel-lost path (wheel_gap.py, F11) takes over.
    glob.declare_parameter('wheel_stuck_check', True)
    glob.declare_parameter('wheel_stuck_v_min', 0.5)     # m/s: a 0 right after more than this = stuck
    stuck = None
    if glob.get_parameter('wheel_stuck_check').value:
        stuck = StuckWheel(v_min=float(glob.get_parameter('wheel_stuck_v_min').value))
        glob.get_logger().info(
            f'stuck-wheel check ON: a wheel speed of ~0 right after > {stuck.v_min:.1f} m/s (wheel or IMU speed) is a '
            f'dead encoder -> treated as missing wheel data (back after {stuck.k_back} plausible samples)')
    else:
        glob.get_logger().info('stuck-wheel check OFF (wheel_stuck_check false): a stuck 0 reaches the EKF (TEST ONLY)')

    def imu_speed():
        sl = glob.slip
        return sl.observer_speed if sl is not None and sl.synced and not sl.lost else None

    def wheel_now(msg):
        if stuck is not None:
            drop = stuck.check(stamp_s(msg), msg.drive.speed, imu_speed())
            for ev in stuck.pop_events():
                if ev[0] == 'stuck':
                    glob.get_logger().warn(
                        f'WHEEL STUCK: the wheel reads {msg.drive.speed:.2f} m/s right after '
                        + (f'{ev[2]:.2f} m/s' if ev[2] is not None else 'no good sample')
                        + (f' (IMU speed {ev[3]:.2f} m/s)' if ev[3] is not None else '')
                        + ' -- a dead encoder, not a stop: its samples are DROPPED (= missing wheel data)')
                else:
                    glob.get_logger().info(f'wheel back after {ev[2]:.2f} s stuck ({ev[3]} samples dropped): '
                                           f'{msg.drive.speed:.2f} m/s')
            if drop:
                return
        if se_wheel:
            se_wheel(msg)
        local.meas_cb(msg)
        glob.meas_cb(msg)

    def on_imu(msg):                             # 100 Hz (sim) / 200 Hz (real car)
        if test_switches:
            lag = float(glob.get_parameter('test_lag_ms').value)
            if lag != lag_on[0]:
                lag_on[0] = lag
                glob.get_logger().warn(f'TEST: estimator lag {lag:.1f} ms extra work per IMU message')
            if lag > 0.0:
                time.sleep(lag / 1000.0)
        if order is not None:                    # waiting wheel messages up to this IMU stamp go FIRST
            for w in order.imu(stamp_s(msg), time.monotonic()):
                wheel_now(w)
        if se_imu:
            se_imu(msg)                          # freshness first
        local.imu_cb(msg)                        # -> TF odom -> est/base_link (for AMCL)
        glob.imu_cb(msg)                         # -> /odom/ekf -> state_estimate -> /state_estimate

    def on_wheel(msg):                           # 50 Hz
        if order is None:
            wheel_now(msg)
            return
        for w in order.wheel(stamp_s(msg), msg, time.monotonic()):
            wheel_now(w)

    # the two shared subscriptions live on ekf_global (any of the three nodes would do)
    glob.create_subscription(Imu, '/imu/data', on_imu, 100)
    glob.create_subscription(AckermannDriveStamped, '/vehicle/measured', on_wheel, 50)
    glob.get_logger().info(
        'state_pipeline: ekf_local + ekf_global' + (' + state_estimate' if with_se else '')
        + ' in ONE process (one /imu/data and one /vehicle/measured subscription)')

    executor = SingleThreadedExecutor()
    for n in nodes:
        executor.add_node(n)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        glob.save_scale()                        # last learned speed scale, as ekf_node does
    for n in nodes:
        n.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
