from glob import glob

from setuptools import find_packages, setup

package_name = 'sensor_fusion'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fatma',
    maintainer_email='fatma@todo.todo',
    description='Camera-LiDAR fusion: projection (LiDAR point <-> pixel), camera-LiDAR extrinsic calibration + demos',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'projection_demo = sensor_fusion.projection_demo:main',
            'calib_demo = sensor_fusion.calib_demo:main',
            'calib_capture = sensor_fusion.calib_capture:main',
            'calib_solve = sensor_fusion.calib_solve:main',
            'calib_fake_session = sensor_fusion.calib_fake_session:main',
            'fake_camera = sensor_fusion.fake_camera:main',
        ],
    },
)
