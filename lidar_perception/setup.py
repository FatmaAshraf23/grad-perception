import os
from glob import glob

from setuptools import setup

package_name = 'lidar_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Fatma',
    maintainer_email='fatmaashraf54321@gmail.com',
    description='2D LiDAR perception: obstacle list (map difference + tracking), clustering, cone detection',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scan_cluster_node = lidar_perception.scan_cluster_node:main',
            'cone_map_node = lidar_perception.cone_map_node:main',
            'obstacle_node = lidar_perception.obstacle_node:main',
        ],
    },
)
