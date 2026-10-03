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
    new     ekf_global hands every odometry it publishes, and the NIS of every
            accepted AMCL fix, DIRECTLY to state_estimate (both are still published
            too, for eval_logger and other tools). /state_estimate therefore leaves
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

Real car: identical -- this is what should run on the Pi (inputs from the STM32
bridge, the MPU-6050 driver and rplidar instead of the simulator).
"""
import sys
import traceback

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Imu

from state_estimation.ekf_node import EkfNode
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
        se_imu = guarded(se, se.imu_cb, 'state_estimate.imu_cb')
        se_wheel = guarded(se, se.wheel_cb, 'state_estimate.wheel_cb')

    def on_imu(msg):                             # 100 Hz (sim) / 200 Hz (real car)
        if se_imu:
            se_imu(msg)                          # freshness first
        local.imu_cb(msg)                        # -> TF odom -> est/base_link (for AMCL)
        glob.imu_cb(msg)                         # -> /odom/ekf -> state_estimate -> /state_estimate

    def on_wheel(msg):                           # 50 Hz
        if se_wheel:
            se_wheel(msg)
        local.meas_cb(msg)
        glob.meas_cb(msg)

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
