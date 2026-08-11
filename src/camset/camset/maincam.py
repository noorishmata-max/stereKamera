#!/usr/bin/env python3

import cv2
import rclpy

from rclpy.node import Node

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from cv_bridge import CvBridge


class StereoSplitter(Node):

    def __init__(self):

        super().__init__("stereo_splitter")

        # ============================
        # Parameter
        # ============================

        self.declare_parameter("device", "/dev/video2")
        self.declare_parameter("fps", 30)

        device = self.get_parameter("device").value
        fps = self.get_parameter("fps").value

        # ============================
        # Camera
        # ============================

        self.cap = cv2.VideoCapture(device)

        if not self.cap.isOpened():
            self.get_logger().error(f"Tidak dapat membuka {device}")
            raise RuntimeError("Camera open failed")

        # MJPEG
        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )

        # Coba resolusi tinggi
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.cap.set(cv2.CAP_PROP_FPS, fps)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.get_logger().info(
            f"Camera Resolution : {width} x {height}"
        )

        self.bridge = CvBridge()

        # ============================
        # Publisher
        # ============================

        self.left_pub = self.create_publisher(
            Image,
            "/my_stereo/left/image_raw",
            10
        )

        self.right_pub = self.create_publisher(
            Image,
            "/my_stereo/right/image_raw",
            10
        )

        self.left_info_pub = self.create_publisher(
            CameraInfo,
            "/my_stereo/left/camera_info",
            10
        )

        self.right_info_pub = self.create_publisher(
            CameraInfo,
            "/my_stereo/right/camera_info",
            10
        )

        self.timer = self.create_timer(
            1.0 / fps,
            self.callback
        )

        self.get_logger().info("Stereo Splitter Started")

    # ========================================

    def callback(self):

        ret, frame = self.cap.read()

        if not ret:
            return

        h, w = frame.shape[:2]

        half = w // 2

        left = frame[:, :half]

        right = frame[:, half:]

        stamp = self.get_clock().now().to_msg()

        left_msg = self.bridge.cv2_to_imgmsg(
            left,
            encoding="bgr8"
        )

        right_msg = self.bridge.cv2_to_imgmsg(
            right,
            encoding="bgr8"
        )

        left_msg.header.stamp = stamp
        right_msg.header.stamp = stamp

        left_msg.header.frame_id = "left_camera"

        right_msg.header.frame_id = "right_camera"

        # ============================
        # Camera Info
        # ============================

        left_info = CameraInfo()

        left_info.header = left_msg.header
        left_info.width = left.shape[1]
        left_info.height = left.shape[0]

        right_info = CameraInfo()

        right_info.header = right_msg.header
        right_info.width = right.shape[1]
        right_info.height = right.shape[0]

        # publish

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)

        self.left_info_pub.publish(left_info)
        self.right_info_pub.publish(right_info)

        # preview

        cv2.imshow("LEFT", left)
        cv2.imshow("RIGHT", right)

        cv2.waitKey(1)

    # ========================================

    def destroy_node(self):

        self.cap.release()

        cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = StereoSplitter()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()