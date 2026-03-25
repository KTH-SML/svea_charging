#!/usr/bin/env python3

from std_msgs.msg import Float32, String

from svea_core import rosonic as rx
from svea_charging.behaviourTree.behaviourTree import ChargingMissionTree, MissionBlackboard


class mission_manager(rx.Node):
    tick_hz = rx.Parameter(10.0)
    switch_distance_m = rx.Parameter(4.0)
    dock_distance_m = rx.Parameter(1.6)

    dist_to_station_topic = rx.Parameter("dist_to_goal")
    aruco_distance_topic = rx.Parameter("aruco/distance_m")
    line_status_topic = rx.Parameter("line_follower/status")

    controller_mode_topic = rx.Parameter("mission/active_controller")
    mission_phase_topic = rx.Parameter("mission/phase")
    tree_status_topic = rx.Parameter("mission/tree_status")
    running_node_topic = rx.Parameter("mission/running_node")

    controller_mode_pub = rx.Publisher(String, controller_mode_topic)
    mission_phase_pub = rx.Publisher(String, mission_phase_topic)
    tree_status_pub = rx.Publisher(String, tree_status_topic)
    running_node_pub = rx.Publisher(String, running_node_topic)

    @rx.Subscriber(Float32, dist_to_station_topic)
    def _dist_to_station_cb(self, msg: Float32):
        self.blackboard.dist_to_station = float(msg.data)

    @rx.Subscriber(Float32, aruco_distance_topic)
    def _aruco_distance_cb(self, msg: Float32):
        distance = float(msg.data)
        self.blackboard.aruco_distance = distance
        self.blackboard.charger_visible = distance > 0.0

    @rx.Subscriber(String, line_status_topic)
    def _line_status_cb(self, msg: String):
        self.line_status = str(msg.data)
        self.blackboard.line_visible = self.line_status != "line_lost"

    def on_startup(self):
        self.blackboard = MissionBlackboard(
            switch_distance_m=float(self.switch_distance_m),
            dock_distance_m=float(self.dock_distance_m),
        )
        self.tree = ChargingMissionTree(self.blackboard)
        self.line_status = "unknown"

        period = 1.0 / max(float(self.tick_hz), 1e-3)
        self.create_timer(period, self._tick_tree)
        self.get_logger().info(
            "Mission manager started. BT owns controller selection between Stanley and line follower."
        )

    def _tick_tree(self):
        status = self.tree.tick()
        self.controller_mode_pub.publish(String(data=self.blackboard.active_controller))
        self.mission_phase_pub.publish(String(data=self.blackboard.mission_phase))
        self.tree_status_pub.publish(String(data=status))
        self.running_node_pub.publish(String(data=self.tree.state))


if __name__ == "__main__":
    mission_manager.main()
