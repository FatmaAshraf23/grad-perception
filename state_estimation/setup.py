from setuptools import setup

package_name = 'state_estimation'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
        ],
    },
)
