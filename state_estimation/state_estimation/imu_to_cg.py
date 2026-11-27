"""Step 1 of ax / ay for Control: move the IMU reading to the car's centre of gravity (CG).

Not handled yet: sensor offset (bias + tilt), computing r_dot, noise filtering.
Car axes: x forward, y left, yaw positive counter-clockwise (ROS REP 103).
"""
import math


def imu_to_cg(ax_imu, ay_imu, r, r_dot, mount_yaw, px, py):
    """Return (ax, ay) at the CG in car axes, in m/s^2.

    ax_imu, ay_imu : IMU reading in the sensor's own axes (m/s^2)
    r, r_dot       : yaw rate (rad/s) and how fast it changes (rad/s^2)
    mount_yaw      : angle of the sensor's x axis from car forward (rad)
    px, py         : sensor position from the CG: forward, left (m)
    """
    # 1. Turn the sensor axes into car axes.
    fx = math.cos(mount_yaw) * ax_imu - math.sin(mount_yaw) * ay_imu
    fy = math.sin(mount_yaw) * ax_imu + math.cos(mount_yaw) * ay_imu

    # 2. Remove the extra acceleration felt because the sensor is away from the CG.
    ax = fx + r_dot * py + r ** 2 * px
    ay = fy - r_dot * px + r ** 2 * py
    return ax, ay
