import math
import unittest

from state_estimation.imu_to_cg import imu_to_cg


def rotate(x, y, angle):
    return (math.cos(angle) * x - math.sin(angle) * y,
            math.sin(angle) * x + math.cos(angle) * y)


class TestImuToCg(unittest.TestCase):
    def check(self, got, want):
        self.assertAlmostEqual(got[0], want[0], places=5)
        self.assertAlmostEqual(got[1], want[1], places=5)

    def test_sensor_at_cg_and_aligned_changes_nothing(self):
        self.check(imu_to_cg(1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0), (1.0, 2.0))

    def test_sensor_mounted_turned_90_degrees(self):
        # Sensor x points left, so forward acceleration appears on sensor -y.
        self.check(imu_to_cg(0.0, -1.0, 0.0, 0.0, math.pi / 2, 0.0, 0.0), (1.0, 0.0))

    def test_steady_circle_sensor_ahead_of_cg(self):
        # v = 2 m/s, radius 1 m: CG has ax = 0, ay = 4. Sensor 10 cm ahead reads ax = -0.4.
        self.check(imu_to_cg(-0.4, 4.0, 2.0, 0.0, 0.0, 0.10, 0.0), (0.0, 4.0))

    def test_against_a_simulated_moving_sensor(self):
        """Independent check: get accelerations by differentiating positions, not from the formula."""
        px, py, mount_yaw = 0.12, -0.05, 0.3

        def heading(t):
            return 0.4 + 1.5 * t + 0.8 * t * t       # r = 1.5 + 1.6 t, r_dot = 1.6

        def cg(t):
            return 1.5 * math.sin(1.2 * t), 0.8 * math.cos(0.9 * t)

        def sensor(t):
            dx, dy = rotate(px, py, heading(t))
            return cg(t)[0] + dx, cg(t)[1] + dy

        def accel(path, t, h=1e-4):
            a, b, c = path(t - h), path(t), path(t + h)
            return tuple((a[i] - 2 * b[i] + c[i]) / h ** 2 for i in range(2))

        for t in (0.0, 0.3, 1.1, 2.0):
            psi = heading(t)
            reading = rotate(*accel(sensor, t), -psi - mount_yaw)  # world -> sensor axes
            truth = rotate(*accel(cg, t), -psi)                    # world -> car axes
            got = imu_to_cg(*reading, 1.5 + 1.6 * t, 1.6, mount_yaw, px, py)
            self.check(got, truth)


if __name__ == '__main__':
    unittest.main()
