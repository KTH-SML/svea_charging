#! /usr/bin/env python3

import numpy as np
import ast

from geometry_msgs.msg import PoseWithCovarianceStamped

from svea_core.interfaces import LocalizationInterface
from svea_charging.controllers.stanleyController import StanleyController
from svea_core.interfaces import ActuationInterface
from svea_core import rosonic as rx
from svea_core.interfaces import ShowPath
from visualization_msgs.msg import Marker

class stanley_control(rx.Node):
    DELTA_TIME = 0.1

    endPoint = rx.Parameter('[2.0, 3.0]')
    target_velocity = rx.Parameter(0.5)
    
    # Interfaces
    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    # Goal Visualization
    mark = Marker()
    # Path Visualization
    path = ShowPath()


    def on_startup(self):

        self.controller = StanleyController()
        self.controller.target_velocity = self.target_velocity

        state = self.localizer.get_state()
        x, y, yaw, vel = state

        self.goal = eval(self.endPoint)
        mx = 0.5*(x + self.goal[0])
        my = 0.5*(y + self.goal[1])
        self.waypoints = [[x, y], [mx, my], self.goal]

        self.mark.marker('goal','blue',self.goal)
        self.controller.update_traj(state, self.waypoints)

        self.create_timer(self.DELTA_TIME, self.loop)

    def loop(self):
        """
        Main loop of the Stanley controller. 
        """
        state = self.localizer.get_state()
        x, y, yaw, vel = state

        self.update_goal()
        self.controller.update_traj(state, self.waypoints)

        steering, velocity = self.controller.compute_control(state)
        self.get_logger().info(f"Steering: {steering}, Velocity: {velocity}")
        self.actuation.send_control(steering, velocity)

    def update_goal(self):

        self.curr += 1
        self.curr %= len(self._points)
        self.goal = self._points[self.curr]
        self.controller.is_finished = False
        # Mark the goal
        self.publish_goal_marker()
        self.mark.marker('goal','blue',self.goal)

    
    def publish_goal_marker(self):
        
        #Publish current goal as a blue sphere Marker in the map frame.
        
        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.ns = "stanley_goal"
        msg.id = 0
        msg.type = Marker.SPHERE
        msg.action = Marker.ADD
            
        msg.pose.position.x = float(self.endPoint[0])
        msg.pose.position.y = float(self.endPoint[1])

        msg.pose.position.z = 0.1
        msg.pose.orientation.w = 1.0

        msg.scale.x = 0.6
        msg.scale.y = 0.6
        msg.scale.z = 0.6

        msg.color.r = 0.0
        msg.color.g = 0.0
        msg.color.b = 1.0
        msg.color.a = 1.0


if __name__ == '__main__':
    stanley_control.main()
