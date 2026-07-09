"""
webcam_publisher

Publishes frames from a USB/laptop webcam as sensor_msgs/Image on
/camera/image_raw, matching what perception_node.py subscribes to.

This exists purely to de-risk development: it lets you prove the full
pipeline (capture -> inference -> tracker -> overlay -> recording) works
correctly before you know the actual vehicle's camera/RADAR hardware.
Swapping this out later for the real camera driver is a one-line topic
change in perception_node.py, nothing else in the pipeline needs to
change.

Usage:
    ros2 run abaja_perception webcam_publisher --ros-args -p device_id:=0
"""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from .image_utils import bgr8_to_imgmsg


class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')

        self.declare_parameter('device_id', 0)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('topic', '/camera/image_raw')

        device_id = self.get_parameter('device_id').get_parameter_value().integer_value
        fps = self.get_parameter('fps').get_parameter_value().double_value
        topic = self.get_parameter('topic').get_parameter_value().string_value

        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            self.get_logger().error(
                f'Could not open camera device {device_id}. Try a different '
                f'device_id (0, 1, 2...) or check `ls /dev/video*` on Linux.')
            raise RuntimeError(f'Camera device {device_id} not available')

        self.pub = self.create_publisher(Image, topic, 10)
        self.timer = self.create_timer(1.0 / max(fps, 1.0), self.on_timer)

        self.get_logger().info(
            f'Publishing webcam device {device_id} on {topic} at {fps} fps')

    def on_timer(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame from camera')
            return
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_frame'
        msg = bgr8_to_imgmsg(frame, header=header)
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
