import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'limo_gz_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jay',
    maintainer_email='you@example.com',
    description='Gazebo Fortress simulation and control suite for LIMO',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pid_tracker = limo_gz_sim.pid_tracker_node:main',
            'mpc_tracker = limo_gz_sim.mpc_tracker_node:main',
            'lqr_tracker = limo_gz_sim.lqr_tracker_node:main',
            'nmpc_tracker = limo_gz_sim.nmpc_tracker_node:main',
            'live_plotter = limo_gz_sim.live_plotter_node:main',
        ],
    },
)
