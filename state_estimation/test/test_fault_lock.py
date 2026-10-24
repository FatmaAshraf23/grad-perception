#!/usr/bin/env python3
"""The TEST-ONLY fault switches of the estimator are locked unless allow_test_faults is true (2026-10-24).
Needs rclpy + the built workspace. Creates ekf_global twice (locked / unlocked) and pokes the switches.
Run: python3 test_fault_lock.py"""
import sys

import rclpy
from rclpy.parameter import Parameter

from state_estimation.ekf_node import EkfNode

RESULTS = []


def check(name, cond):
    RESULTS.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}")


def make(allow, test_fault=''):
    args = ['--ros-args', '-p', f"allow_test_faults:={'true' if allow else 'false'}"]
    if test_fault:
        args += ['-p', f'test_fault:={test_fault}']
    rclpy.init(args=args)
    return EkfNode('ekf_global', own_inputs=False)


def refused(node, name, value):
    """True if setting the parameter is refused (not declared -> exception, or an unsuccessful result)."""
    try:
        res = node.set_parameters([Parameter(name, value=value)])
    except Exception:                          # noqa: BLE001 -- ParameterNotDeclaredException & co.
        return True
    return not res[0].successful


def main():
    n = make(False, test_fault='1.0,0.5,0.0,0.0')
    check('locked by default: allow_test_faults is false', n.allow_test_faults is False)
    check('locked: test_kick is not declared at all', not n.has_parameter('test_kick'))
    check('locked: setting test_kick is refused', refused(n, 'test_kick', '0.5,0.0,0.0'))
    check('locked: a test_fault given at start-up is ignored', n.fault is None)
    check('allow_test_faults cannot be switched on at runtime (read-only)', refused(n, 'allow_test_faults', True))
    n.destroy_node()
    rclpy.shutdown()

    n = make(True)
    check('unlocked: test_kick is declared', n.has_parameter('test_kick'))
    ok = not refused(n, 'test_kick', '0.5,0.0,0.0')
    check('unlocked: a kick is accepted and armed', ok and n.fault == [0.0, 0.5, 0.0, 0.0])
    n.destroy_node()
    rclpy.shutdown()

    n = make(True, test_fault='1.0,0.5,0.0,0.0')
    check('unlocked: a test_fault given at start-up is armed', n.fault == [1.0, 0.5, 0.0, 0.0])
    n.destroy_node()
    rclpy.shutdown()
    print(f'\n{sum(RESULTS)} of {len(RESULTS)} passed')
    return 0 if all(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
