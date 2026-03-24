#!/usr/bin/env python3

from geometry_msgs.msg import Twist
from std_msgs.msg import String

from svea_core import rosonic as rx
from svea_core.interfaces import ActuationInterface


class control_mux(rx.Node):
    publish_rate_hz = rx.Parameter(20.0)
    command_timeout_s = rx.Parameter(0.3)
    active_controller_topic = rx.Parameter("control/active_controller")
    stanley_cmd_topic = rx.Parameter("controller_cmd/stanley")
    line_follower_cmd_topic = rx.Parameter("controller_cmd/line_follower")

    actuation = ActuationInterface()

    @rx.Subscriber(String, active_controller_topic)
    def _active_controller_cb(self, msg: String):
        self.active_controller = str(msg.data).strip() or self.default_controller

    @rx.Subscriber(Twist, stanley_cmd_topic)
    def _stanley_cmd_cb(self, msg: Twist):
        self.stanley_cmd = msg
        self.stanley_cmd_time = self._now_s()

    @rx.Subscriber(Twist, line_follower_cmd_topic)
    def _line_follower_cmd_cb(self, msg: Twist):
        self.line_follower_cmd = msg
        self.line_follower_cmd_time = self._now_s()

    def on_startup(self):
        self.default_controller = "stanley"
        self.active_controller = self.default_controller
        self.stanley_cmd = None
        self.line_follower_cmd = None
        self.stanley_cmd_time = 0.0
        self.line_follower_cmd_time = 0.0
        self.last_sent = (None, None)
        period_s = 1.0 / max(float(self.publish_rate_hz), 1.0)
        self.create_timer(period_s, self.loop)

    def on_shutdown(self):
        self.actuation.send_control(0.0, 0.0)

    def loop(self):
        cmd, is_fresh = self._get_active_command()

        if not is_fresh or cmd is None:
            steering = 0.0
            velocity = 0.0
        else:
            steering = float(cmd.angular.z)
            velocity = float(cmd.linear.x)

        if self.last_sent != (steering, velocity):
            self.actuation.send_control(steering, velocity)
            self.last_sent = (steering, velocity)

    def _get_active_command(self):
        timeout_s = max(float(self.command_timeout_s), 0.05)
        now_s = self._now_s()

        if self.active_controller == "line_follower":
            return self.line_follower_cmd, (now_s - self.line_follower_cmd_time) <= timeout_s

        return self.stanley_cmd, (now_s - self.stanley_cmd_time) <= timeout_s

    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9


if __name__ == "__main__":
    control_mux.main()
