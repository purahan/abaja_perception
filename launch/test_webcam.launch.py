import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory('abaja_perception'),
        'config', 'classes.yaml')

    device_id_arg = DeclareLaunchArgument(
        'device_id', default_value='0',
        description='Webcam device index (0 = default laptop/USB camera)')

    return LaunchDescription([
        device_id_arg,
        Node(
            package='abaja_perception',
            executable='webcam_publisher',
            name='webcam_publisher',
            output='screen',
            parameters=[{'device_id': LaunchConfiguration('device_id')}],
        ),
        Node(
            package='abaja_perception',
            executable='perception_node',
            name='abaja_perception_node',
            output='screen',
            parameters=[{'config_path': config_path}],
        ),
    ])
