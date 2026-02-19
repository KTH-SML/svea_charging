#! /usr/bin/env python3

import numpy as np

from geometry_msgs.msg import PoseWithCovarianceStamped

from svea_core.svea_core.interfaces import LocalizationInterface
from stanleyController import StanleyController
from svea_core.svea_core.interfaces import ActuationInterface
from svea_core import rosonic as rx
from svea_core.svea_core.utils import PlaceMarker, ShowPath

class stanley_control(rx.Node):
    DELTA_TIME = 0.1
    TRAJ_LEN = 20

    points = rx.Parameter(['[-2.3, -7.1]', '[10.5, 11.7]', '[5.7, 15.0]', '[-7.0, -4.0]']) #we should change these points.
    state = rx.Parameter([-7.4, -15.3, 0.9, 0.0])  # x, y, yaw, vel (we should change initial state.)
    target_velocity = rx.Parameter(0.6)
    
    # Interfaces
    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    # Goal Visualization
    mark = PlaceMarker()
    # Path Visualization
    path = ShowPath()


    def on_startup(self):
        # Convert POINTS to numerical lists if loaded as strings
        if isinstance(self.points[0], str):
            self._points = [eval(point) for point in self.points]

        self.controller = StanleyController()
        self.controller.target_velocity = self.target_velocity

        state = self.localizer.get_state()
        x, y, yaw, vel = state

        self.curr = 0
        self.goal = self._points[self.curr]
        self.mark.marker('goal','blue',self.goal)
        self.update_traj(x, y)

        self.create_timer(self.DELTA_TIME, self.loop)

    def loop(self):
        """
        Main loop of the Stanley controller. 
        """
        state = self.localizer.get_state()
        x, y, yaw, vel = state

        if self.controller.is_finished:
            self.update_goal()
            self.controller.update_traj(x, y)

        steering, velocity = self.controller.compute_control(state)
        self.get_logger().info(f"Steering: {steering}, Velocity: {velocity}")
        self.actuation.send_control(steering, velocity)

    def update_goal(self):

        self.curr += 1
        self.curr %= len(self._points)
        self.goal = self._points[self.curr]
        self.controller.is_finished = False
        # Mark the goal
        self.mark.marker('goal','blue',self.goal)


if __name__ == '__main__':
    stanley_control.main()
