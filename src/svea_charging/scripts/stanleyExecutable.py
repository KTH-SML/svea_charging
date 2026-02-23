#! /usr/bin/env python3

import numpy as np
import ast
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from svea_core.interfaces import LocalizationInterface
from svea_charging.controllers.stanleyController import StanleyController
from svea_core.interfaces import ActuationInterface
from svea_core import rosonic as rx
from svea_core.interfaces import ShowPath

class stanley_control(rx.Node):
    DELTA_TIME = 0.1

    endPoint = rx.Parameter('[2.0, -2.5]') #0.5, -1.2 irl
    target_velocity = rx.Parameter(0.4)
    
    # Interfaces
    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    goal_tolerance = rx.Parameter(0.2) #m


    def on_startup(self):
        self.reached_goal = False
        self.counter = 0

        #create publishers
        self.goal_pub = self.create_publisher(Marker, 'goal_marker', 10)
        self.path_pub = self.create_publisher(Marker, 'path_marker', 10)
        self.traj_pub = self.create_publisher(Marker, 'traj_marker', 10)

        self.controller = StanleyController()
        self.controller.target_velocity = self.target_velocity

        import time
        time.sleep(8) # wait for localization to start up and get first state
        state = self.localizer.get_state()
        x, y, yaw, vel = state

        self.goal = eval(self.endPoint)
        mx = 0.5*(x + self.goal[0])
        my = 0.5*(y + self.goal[1])
        self.waypoints = [[x, y], [mx, my], self.goal]
        
        #publish goal and waypoints
        #self.publish_goal_marker(self.goal)
        #self.publish_waypoints_marker(self.waypoints)

        self.controller.update_traj(state, self.waypoints)
        self.create_timer(self.DELTA_TIME, self.loop)


    def loop(self):
        """
        Main loop of the Stanley controller. 
        """
        state = self.localizer.get_state()
        x, y, yaw, vel = state

        dist = self.distance_to_goal(state)
        if dist <= self.goal_tolerance:
            if not self.reached_goal:
                self.get_logger().info("Reached goal!")
                self.reached_goal = True

        #self.update_goal()
        self.controller.update_traj(state, self.waypoints)

        if not self.reached_goal:
            steering, velocity = self.controller.compute_control(state)
            self.get_logger().info(f"Steering: {steering}, Velocity: {velocity}")
        else:
            steering, velocity = 0.0, 0.0

        self.actuation.send_control(steering, velocity)
        

        if self.counter % 10 == 0: # publish markers every 1 seconds
            self.publish_goal_marker(self.goal)
            self.publish_waypoints_marker(self.waypoints)
            self.publish_trajectory_marker(self.controller.cx, self.controller.cy)


    def distance_to_goal(self, state):
        x, y, _, _ = state
        goal_x, goal_y = self.goal
        return np.sqrt((goal_x - x)**2 + (goal_y - y)**2)

    def update_goal(self):

        self.curr += 1
        self.curr %= len(self._points)
        self.goal = self._points[self.curr]
        self.controller.is_finished = False
        # Mark the goal
        self.publish_goal_marker()
    
    def publish_goal_marker(self, goal_xy):
        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = "stanley_goal"
        msg.id = 0
        msg.type = Marker.SPHERE
        msg.action = Marker.ADD

        msg.pose.position.x = float(goal_xy[0])
        msg.pose.position.y = float(goal_xy[1])
        msg.pose.position.z = 0.2
        msg.pose.orientation.w = 1.0

        msg.scale.x = 0.4
        msg.scale.y = 0.4
        msg.scale.z = 0.4

        msg.color.r = 0.0
        msg.color.g = 0.0
        msg.color.b = 1.0
        msg.color.a = 1.0

        self.goal_pub.publish(msg)

    def publish_waypoints_marker(self, waypoints):
        # Draw straight segments between waypoints
        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = "stanley_waypoints"
        msg.id = 0
        msg.type = Marker.LINE_STRIP
        msg.action = Marker.ADD

        msg.scale.x = 0.02  # line width

        msg.color.r = 1.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.color.a = 1.0

        msg.points = []
        for wp in waypoints:
            p = Point()
            p.x = float(wp[0])
            p.y = float(wp[1])
            p.z = 0.05
            msg.points.append(p)

        self.path_pub.publish(msg)

    def publish_trajectory_marker(self, cx, cy):
        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = "stanley_traj"
        msg.id = 0
        msg.type = Marker.LINE_STRIP
        msg.action = Marker.ADD

        msg.scale.x = 0.05
        msg.color.r = 0.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.color.a = 1.0

        msg.points = []
        for x, y in zip(cx, cy):
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.03
            msg.points.append(p)

        self.traj_pub.publish(msg)


if __name__ == '__main__':
    stanley_control.main()
