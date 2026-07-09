import os
from glob import glob
from setuptools import setup

package_name = 'abaja_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Purahan',
    maintainer_email='you@example.com',
    description='Object detection and classification perception stack for aBAJA SAEINDIA 2026',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = abaja_perception.perception_node:main',
            'webcam_publisher = abaja_perception.webcam_publisher:main',
        ],
    },
)
