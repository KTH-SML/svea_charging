#!/usr/bin/env python3

from std_msgs.msg import Bool, Int8, String

from svea_core import rosonic as rx


class control_mux(rx.Node):
    publish_hz = rx.Parameter(20.0)
    active_controller_topic = rx.Parameter("mission/active_controller")

    stanley_steering_topic = rx.Parameter("controllers/stanley/steering")
    stanley_throttle_topic = rx.Parameter("controllers/stanley/throttle")
    stanley_highgear_topic = rx.Parameter("controllers/stanley/highgear")
    stanley_diff_topic = rx.Parameter("controllers/stanley/diff")

    line_steering_topic = rx.Parameter("controllers/line_follower/steering")
    line_throttle_topic = rx.Parameter("controllers/line_follower/throttle")
    line_highgear_topic = rx.Parameter("controllers/line_follower/highgear")
    line_diff_topic = rx.Parameter("controllers/line_follower/diff")

    output_steering_topic = rx.Parameter("lli/ctrl/steering")
    output_throttle_topic = rx.Parameter("lli/ctrl/throttle")
    output_highgear_topic = rx.Parameter("lli/ctrl/highgear")
    output_diff_topic = rx.Parameter("lli/ctrl/diff")

    steering_pub = rx.Publisher(Int8, output_steering_topic)
    throttle_pub = rx.Publisher(Int8, output_throttle_topic)
    highgear_pub = rx.Publisher(Bool, output_highgear_topic)
    diff_pub = rx.Publisher(Bool, output_diff_topic)

    @rx.Subscriber(String, active_controller_topic)
    def _active_controller_cb(self, msg: String):
        self.active_controller = str(msg.data)

    @rx.Subscriber(Int8, stanley_steering_topic)
    def _stanley_steering_cb(self, msg: Int8):
        self.stanley_cmd["steering"] = msg

    @rx.Subscriber(Int8, stanley_throttle_topic)
    def _stanley_throttle_cb(self, msg: Int8):
        self.stanley_cmd["throttle"] = msg

    @rx.Subscriber(Bool, stanley_highgear_topic)
    def _stanley_highgear_cb(self, msg: Bool):
        self.stanley_cmd["highgear"] = msg

    @rx.Subscriber(Bool, stanley_diff_topic)
    def _stanley_diff_cb(self, msg: Bool):
        self.stanley_cmd["diff"] = msg

    @rx.Subscriber(Int8, line_steering_topic)
    def _line_steering_cb(self, msg: Int8):
        self.line_cmd["steering"] = msg

    @rx.Subscriber(Int8, line_throttle_topic)
    def _line_throttle_cb(self, msg: Int8):
        self.line_cmd["throttle"] = msg

    @rx.Subscriber(Bool, line_highgear_topic)
    def _line_highgear_cb(self, msg: Bool):
        self.line_cmd["highgear"] = msg

    @rx.Subscriber(Bool, line_diff_topic)
    def _line_diff_cb(self, msg: Bool):
        self.line_cmd["diff"] = msg

    def on_startup(self):
        self.active_controller = "idle"
        self._last_published_mode = None
        self.stanley_cmd = self._neutral_command()
        self.line_cmd = self._neutral_command()

        period = 1.0 / max(float(self.publish_hz), 1e-3)
        self.create_timer(period, self._publish_selected_command)
        self.get_logger().info("Control mux started. Waiting for mission-selected controller.")

    def _neutral_command(self):
        return {
            "steering": Int8(data=0),
            "throttle": Int8(data=0),
            "highgear": Bool(data=False),
            "diff": Bool(data=False),
        }

    def _selected_command(self):
        if self.active_controller == "stanley":
            return self.stanley_cmd
        if self.active_controller == "line_follower":
            return self.line_cmd
        return self._neutral_command()

    def _publish_selected_command(self):
        command = self._selected_command()

        # Publish neutral commands while idle or after a mode change so stale
        # commands do not persist on the low-level interface.
        if self.active_controller != self._last_published_mode:
            self.steering_pub.publish(Int8(data=0))
            self.throttle_pub.publish(Int8(data=0))

        self.steering_pub.publish(command["steering"])
        self.throttle_pub.publish(command["throttle"])
        self.highgear_pub.publish(command["highgear"])
        self.diff_pub.publish(command["diff"])
        self._last_published_mode = self.active_controller


if __name__ == "__main__":
    control_mux.main()
