#!/usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from svea_core import rosonic as rx
from svea_core.interfaces import ActuationInterface


class line_follower(rx.Node):
    dt = rx.Parameter(0.05)
    image_topic = rx.Parameter("camera1/camera/image_raw")
    target_velocity = rx.Parameter(0.4)
    max_velocity = rx.Parameter(0.7)
    stop_on_lost_line = rx.Parameter(True)

    publish_debug_image = rx.Parameter(True)
    debug_image_topic = rx.Parameter("line_follower/debug_image")
    debug_publish_every_n = rx.Parameter(3)

    lower_h = rx.Parameter(20)
    lower_s = rx.Parameter(100)
    lower_v = rx.Parameter(100)
    upper_h = rx.Parameter(35)
    upper_s = rx.Parameter(255)
    upper_v = rx.Parameter(255)


    crop_start_ratio = rx.Parameter(0.55)
    min_contour_area = rx.Parameter(120)
    steering_kp = rx.Parameter(0.55)
    steering_ki = rx.Parameter(0.1)
    steering_kd = rx.Parameter(0.0)
    steering_limit_rad = rx.Parameter(0.6)
    lost_line_steering_rad = rx.Parameter(0.0)
    velocity_scale_from_error = rx.Parameter(True)

    use_aruco_stop = rx.Parameter(True)
    aruco_distance_topic = rx.Parameter("aruco/distance_m")
    aruco_stop_distance_m = rx.Parameter(0.25)
    aruco_velocity_gain = rx.Parameter(1.0)

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
        self.debug_publish_counter = 0
        
        self.steering_error_prev = 0.0
        self.steering_error_integral = 0.0

        self.create_subscription(
            Image,
            str(self.image_topic),
            self._image_callback,
            1,
        )

        self.dt_s = max(float(self.dt), 1e-3)
        self.create_timer(self.dt_s, self.loop)
        self.get_logger().info(
            f"Line follower started on image_topic={self.image_topic}, dt={self.dt_s:.3f}s"
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
        height, _ = frame.shape[:2]
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

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        min_area = float(self.min_contour_area)
        line = None

        for contour in contours:
            moments = cv2.moments(contour)
            if moments["m00"] > min_area:
                line = (
                    int(moments["m10"] / moments["m00"]),
                    int(moments["m01"] / moments["m00"]) + crop_start,
                )

        return line, mask

    def loop(self):
        frame = self.latest_frame
        if frame is None:
            return

        _, width = frame.shape[:2]
        image_center_x = width / 2.0

        if self.latest_centroid is None:
            self._publish_status("line_lost")
            self.steering_error_prev = 0.0
            self.steering_error_integral = 0.0
            if bool(self.stop_on_lost_line):
                self.actuation.send_control(float(self.lost_line_steering_rad), 0.0)
            self._publish_debug_image(frame, None, None)
            return

        cx, cy = self.latest_centroid
        error_px = cx - image_center_x
        normalized_error = error_px / max(image_center_x, 1.0)

        dt = self.dt_s
        error_i = (normalized_error + self.steering_error_prev) / 2.0 * dt
        error_d = (normalized_error - self.steering_error_prev) / max(dt, 1e-6)
        self.steering_error_integral += error_i
        self.steering_error_integral = float(
            np.clip(
                self.steering_error_integral,
                -float(self.steering_limit_rad) * 1.5,
                float(self.steering_limit_rad) * 1.5,
            )
        )
        self.steering_error_prev = float(normalized_error)

        steering = -(
            float(self.steering_kp) * normalized_error
            + float(self.steering_ki) * self.steering_error_integral
            + float(self.steering_kd) * error_d
        )
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

        return "approaching_aruco"

    def _apply_aruco_stop_logic(self, velocity: float) -> float:
        if not bool(self.use_aruco_stop):
            return velocity

        distance = float(self.aruco_distance)
        if distance <= 0.0:
            return velocity

        stop_distance = float(self.aruco_stop_distance_m)

        if distance <= stop_distance:
            return 0.0

        regulated_velocity = float(self.aruco_velocity_gain) * (distance - stop_distance)
        return min(velocity, max(0.0, regulated_velocity))

    def _publish_debug_image(self, frame, centroid, error_px):
        if not bool(self.publish_debug_image):
            return

        self.debug_publish_counter += 1
        if self.debug_publish_counter % max(int(self.debug_publish_every_n), 1) != 0:
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
