from glob import glob

from setuptools import find_packages, setup

package_name = 'apex_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fatma',
    maintainer_email='fatmaashraf54321@gmail.com',
    description='Start the REAL car: car settings (checked), RPLIDAR, de-skew, sensor frames, localization',
    license='MIT',
    entry_points={
        'console_scripts': [
            'check_car_config = apex_bringup.car_config:main',
            'car_slip_calib = apex_bringup.car_calib:main_slip',
        ],
    },
)
