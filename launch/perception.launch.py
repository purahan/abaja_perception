import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory('abaja_perception'),
        'config', 'classes.yaml')

    return LaunchDescription([
        Node(
            package='abaja_perception',
            executable='perception_node',
            name='abaja_perception_node',
            output='screen',
            parameters=[{'config_path': config_path}],
        )
    ])
