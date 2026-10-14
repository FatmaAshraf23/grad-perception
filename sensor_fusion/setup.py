from setuptools import find_packages, setup

package_name = 'sensor_fusion'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fatma',
    maintainer_email='fatma@todo.todo',
    description='Camera-LiDAR fusion: projection (LiDAR point <-> pixel) + demo',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'projection_demo = sensor_fusion.projection_demo:main',
        ],
    },
)
