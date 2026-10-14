from glob import glob

from setuptools import find_packages, setup

package_name = 'apex_description'

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
    maintainer_email='fatma@todo.todo',
    description='Sensor mounts -> static TFs (sensor_frames) + check_frames',
    license='MIT',
    entry_points={
        'console_scripts': [
            'sensor_frames = apex_description.sensor_frames:main',
            'check_frames = apex_description.check_frames:main',
        ],
    },
)
