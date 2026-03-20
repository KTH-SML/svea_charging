#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from svea_core import rosonic as rx
from svea_core.interfaces import ActuationInterface


class line_follower(rx.Node):
    image_topic = rx.Parameter("/image_raw")
    target_velocity = rx.Parameter(0.4)
    max_velocity = rx.Parameter(0.7)
    loop_hz = rx.Parameter(20.0)
    stop_on_lost_line = rx.Parameter(True)
    publish_debug_image = rx.Parameter(True)
    debug_image_topic = rx.Parameter("line_follower/debug_image")

    lower_h = rx.Parameter(100)
    lower_s = rx.Parameter(50)
    lower_v = rx.Parameter(50)
    upper_h = rx.Parameter(130)
    upper_s = rx.Parameter(255)
    upper_v = rx.Parameter(255)

    crop_start_ratio = rx.Parameter(0.55)
    min_contour_area = rx.Parameter(120)
    steering_gain = rx.Parameter(0.55)
    steering_limit_rad = rx.Parameter(0.6)
    lost_line_steering_rad = rx.Parameter(0.0)
    velocity_scale_from_error = rx.Parameter(True)
    use_aruco_stop = rx.Parameter(True)
    aruco_distance_topic = rx.Parameter("aruco/distance_m")
    aruco_slowdown_distance_m = rx.Parameter(0.8)
    aruco_stop_distance_m = rx.Parameter(0.25)
    aruco_min_velocity = rx.Parameter(0.1)

    actuation = ActuationInterface()

    line_error_pub = rx.Publisher(Float32, "line_follower/error_px")
    status_pub = rx.Publisher(String, "line_follower/status")
    centroid_pub = rx.Publisher(Point, "line_follower/centroid")
    debug_image_pub = rx.Publisher(Image, debug_image_topic)

    @rx.Subscriber(Float32, aruco_distance_topic)
    def _aruco_distance_callback(self, msg: Float32):
        self.aruco_distance = float(msg.data)

    def on_startup(self):
        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_mask = None
        self.latest_centroid = None
        self.line_detected = False
        self.aruco_distance = -1.0

        self.create_subscription(
            Image,
            str(self.image_topic),
            self._image_callback,
            10,
        )

        period = 1.0 / max(float(self.loop_hz), 1.0)
        self.create_timer(period, self.loop)
        self.get_logger().info(
            f"Line follower started on image_topic={self.image_topic}"
        )

    def on_shutdown(self):
        self.actuation.send_control(0.0, 0.0)

    def _image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert image: {exc}")
            return

        centroid, mask = self._extract_line_centroid(frame)
        self.latest_frame = frame
        self.latest_mask = mask
        self.latest_centroid = centroid
        self.line_detected = centroid is not None

    def _extract_line_centroid(self, frame):
        height, width = frame.shape[:2]
        crop_start = int(np.clip(float(self.crop_start_ratio), 0.0, 0.95) * height)
        roi = frame[crop_start:, :]

        hsv_image = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array(
            [int(self.lower_h), int(self.lower_s), int(self.lower_v)],
            dtype=np.uint8,
        )
        upper = np.array(
            [int(self.upper_h), int(self.upper_s), int(self.upper_v)],
            dtype=np.uint8,
        )
        mask = cv2.inRange(hsv_image, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = float(self.min_contour_area)
        best_contour = None
        best_area = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area and area > best_area:
                best_contour = contour
                best_area = area

        if best_contour is None:
            return None, mask

        moments = cv2.moments(best_contour)
        if moments["m00"] <= 0.0:
            return None, mask

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"]) + crop_start
        return (cx, cy), mask

    def loop(self):
        frame = self.latest_frame
        if frame is None:
            return

        height, width = frame.shape[:2]
        image_center_x = width / 2.0

        if self.latest_centroid is None:
            self._publish_status("line_lost")
            if bool(self.stop_on_lost_line):
                self.actuation.send_control(
                    float(self.lost_line_steering_rad),
                    0.0,
                )
            self._publish_debug_image(frame, None, None)
            return

        cx, cy = self.latest_centroid
        error_px = cx - image_center_x
        normalized_error = error_px / max(image_center_x, 1.0)

        steering = -float(self.steering_gain) * float(normalized_error)
        steering = float(
            np.clip(
                steering,
                -float(self.steering_limit_rad),
                float(self.steering_limit_rad),
            )
        )

        if bool(self.velocity_scale_from_error):
            speed_scale = max(0.25, 1.0 - min(abs(normalized_error), 1.0))
        else:
            speed_scale = 1.0

        velocity = float(self.target_velocity) * speed_scale
        velocity = self._apply_aruco_stop_logic(velocity)
        velocity = float(np.clip(velocity, 0.0, float(self.max_velocity)))

        self.actuation.send_control(steering, velocity)

        self.line_error_pub.publish(Float32(data=float(error_px)))
        self._publish_status(self._get_status_text(velocity))

        centroid_msg = Point()
        centroid_msg.x = float(cx)
        centroid_msg.y = float(cy)
        centroid_msg.z = 0.0
        self.centroid_pub.publish(centroid_msg)

        self._publish_debug_image(frame, (cx, cy), error_px)

    def _publish_status(self, text: str):
        self.status_pub.publish(String(data=text))

    def _get_status_text(self, velocity: float) -> str:
        if not bool(self.use_aruco_stop) or self.aruco_distance <= 0.0:
            return "tracking"

        if velocity <= 0.0:
            return "stopped_at_aruco"

        if self.aruco_distance <= float(self.aruco_slowdown_distance_m):
            return "slowing_for_aruco"

        return "tracking"

    def _apply_aruco_stop_logic(self, velocity: float) -> float:
        if not bool(self.use_aruco_stop):
            return velocity

        distance = float(self.aruco_distance)
        if distance <= 0.0:
            return velocity

        stop_distance = float(self.aruco_stop_distance_m)
        slowdown_distance = max(float(self.aruco_slowdown_distance_m), stop_distance)
        min_velocity = max(0.0, float(self.aruco_min_velocity))

        if distance <= stop_distance:
            return 0.0

        if distance >= slowdown_distance:
            return velocity

        span = max(slowdown_distance - stop_distance, 1e-6)
        ratio = (distance - stop_distance) / span
        regulated_velocity = min_velocity + ratio * max(velocity - min_velocity, 0.0)
        return min(velocity, regulated_velocity)

    def _publish_debug_image(self, frame, centroid, error_px):
        if not bool(self.publish_debug_image):
            return

        debug = frame.copy()
        height, width = debug.shape[:2]
        center_x = width // 2

        cv2.line(debug, (center_x, 0), (center_x, height), (0, 255, 255), 2)

        crop_start = int(np.clip(float(self.crop_start_ratio), 0.0, 0.95) * height)
        cv2.line(debug, (0, crop_start), (width, crop_start), (255, 255, 0), 2)

        if centroid is not None:
            cv2.circle(debug, centroid, 8, (0, 0, 255), -1)
            cv2.putText(
                debug,
                f"error_px={error_px:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if bool(self.use_aruco_stop) and self.aruco_distance > 0.0:
                cv2.putText(
                    debug,
                    f"aruco_d={self.aruco_distance:.2f} m",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 220, 0),
                    2,
                    cv2.LINE_AA,
                )
        else:
            cv2.putText(
                debug,
                "line lost",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.debug_image_pub.publish(msg)


if __name__ == "__main__":
    line_follower.main()
