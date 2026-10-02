"""Works two ways:
  ROS 2:    put the folder in ~/ros2_ws/src, `colcon build --packages-select apex_track`
  no ROS:   `pip install -e .` inside this folder (Windows, a laptop, a notebook)
"""
import os
from glob import glob

from setuptools import setup

package_name = 'apex_track'

# every track folder (tracks/<name>/track.yaml + centerline.csv) is installed to
# share/apex_track/tracks/<name>/ so ROS nodes can find it by package name
track_files = []
for d in sorted(glob('tracks/*')):
    if os.path.isdir(d):
        track_files.append((os.path.join('share', package_name, d),
                            sorted(glob(os.path.join(d, '*.*')))))

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + track_files,
    install_requires=['setuptools', 'numpy', 'pyyaml'],
    zip_safe=True,
    maintainer='Fatma Ashraf',
    maintainer_email='fatmaashraf54321@gmail.com',
    description='APEX shared track geometry (global <-> Frenet, track_id, track_hash)',
    license='MIT',
    tests_require=['pytest'],
)
