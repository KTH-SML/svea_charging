#!/usr/bin/env python3

from std_msgs.msg import Float32, String

from svea_core import rosonic as rx


class mission_manager(rx.Node):
    publish_period_s = rx.Parameter(0.2)
    start_controller = rx.Parameter("stanley")
    active_controller_topic = rx.Parameter("control/active_controller")
    stanley_dist_topic = rx.Parameter("dist_to_goal")
    aruco_distance_topic = rx.Parameter("aruco/distance_m")
    switch_dist_to_goal_m = rx.Parameter(1.8)
    switch_aruco_distance_m = rx.Parameter(2.5)
    allow_switch_back = rx.Parameter(False)

    active_controller_pub = rx.Publisher(String, active_controller_topic)
    mode_reason_pub = rx.Publisher(String, "control/mode_reason")

    @rx.Subscriber(Float32, stanley_dist_topic)
    def _stanley_dist_cb(self, msg: Float32):
        self.dist_to_goal = float(msg.data)

    @rx.Subscriber(Float32, aruco_distance_topic)
    def _aruco_distance_cb(self, msg: Float32):
        self.aruco_distance = float(msg.data)

    def on_startup(self):
        self.dist_to_goal = -1.0
        self.aruco_distance = -1.0
        self.active_controller = str(self.start_controller)
        self.last_reason = "startup"
        self.create_timer(max(float(self.publish_period_s), 0.05), self.loop)
        self._publish_mode()

    def loop(self):
        next_controller = self.active_controller
        reason = self.last_reason

        if self.active_controller == "stanley":
            if 0.0 < self.dist_to_goal <= float(self.switch_dist_to_goal_m):
                next_controller = "line_follower"
                reason = f"dist_to_goal<={float(self.switch_dist_to_goal_m):.2f}m"
            elif 0.0 < self.aruco_distance <= float(self.switch_aruco_distance_m):
                next_controller = "line_follower"
                reason = f"aruco_distance<={float(self.switch_aruco_distance_m):.2f}m"
        elif bool(self.allow_switch_back):
            if self.aruco_distance < 0.0 and self.dist_to_goal > float(self.switch_dist_to_goal_m):
                next_controller = "stanley"
                reason = "switch_back_enabled"

        if next_controller != self.active_controller:
            self.active_controller = next_controller
            self.last_reason = reason
            self.get_logger().info(f"Active controller switched to {self.active_controller}: {reason}")

        self._publish_mode()

    def _publish_mode(self):
        self.active_controller_pub.publish(String(data=self.active_controller))
        self.mode_reason_pub.publish(String(data=self.last_reason))


if __name__ == "__main__":
    mission_manager.main()
