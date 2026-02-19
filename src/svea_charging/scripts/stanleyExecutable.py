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

    endPoint = rx.Parameter('[0, 0]')
    target_velocity = rx.Parameter(0.5)
    
    # Interfaces
    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    # Goal Visualization
    mark = PlaceMarker()
    # Path Visualization
    path = ShowPath()


    def on_startup(self):

        self.controller = StanleyController()
        self.controller.target_velocity = self.target_velocity

        state = self.localizer.get_state()
        x, y, yaw, vel = state

        self.goal = eval(self.endPoint)
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
            #self.update_goal()
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
