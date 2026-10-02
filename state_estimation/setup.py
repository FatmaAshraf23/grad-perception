import os
from glob import glob

from setuptools import setup

package_name = 'state_estimation'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Fatma',
    maintainer_email='fatmaashraf54321@gmail.com',
    description='Vehicle state estimation: wheel odometry, IMU, EKF',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fake_vehicle_sensors = state_estimation.fake_vehicle_sensors:main',
            'wheel_odometry = state_estimation.wheel_odometry:main',
            'fake_imu = state_estimation.fake_imu:main',
            'imu_heading_check = state_estimation.imu_heading_check:main',
            'ekf_node = state_estimation.ekf_node:main',
            'sim_scan_relay = state_estimation.sim_scan_relay:main',
            'test_driver = state_estimation.test_driver:main',
            'eval_logger = state_estimation.eval_logger:main',
            'wheel_calib = state_estimation.wheel_calib:main',
        ],
    },
)
