#!/usr/bin/env python3
from better_launch import BetterLaunch, launch_this

def toargs(**kwds):
    args = []
    for k, v in kwds.items():
        args.append(f"--{k.replace('_', '-')}")
        args.append(str(v))
    return args

@launch_this
def main(
    name: str = 'svea',
    use_gps: bool = True,
    use_lidar: bool = True,
    map_frame: str = 'map',
    odom_frame: str = 'odom',
    base_frame: str = 'base_link',
    # Router mounting offset w.r.t. base_link [m, rad]. Measure on the car.
    router_x: float = 0.0,
    router_y: float = 0.0,
    router_z: float = 0.0,
    router_yaw: float = 0.0,
):
    bl = BetterLaunch()

    # map -> odom and odom -> base_link are dynamic transforms. They are
    # published by the global and local localization nodes respectively.
    bl.node("tf2_ros", "static_transform_publisher",
            name="broadcaster_imu",
            cmd_args=toargs(x=0.10, y=-0.047, z=0.17, yaw=0, pitch=0, roll=0,
                            frame_id=base_frame, child_frame_id=f"{name}/imu"))

    if use_gps:
        bl.node("tf2_ros", "static_transform_publisher",
                name="broadcaster_gps",
                cmd_args=toargs(x=0.04, y=0.103, z=0.140, yaw=0, pitch=0, roll=0,
                                frame_id=base_frame, child_frame_id=f"{name}/gps"))

    if use_lidar:
        bl.node("tf2_ros", "static_transform_publisher",
                name="broadcaster_lidar",
                cmd_args=toargs(x=0.385, y=0, z=0.15, yaw=0, pitch=0, roll=0,
                                frame_id=base_frame, child_frame_id=f"{name}/laser"))
    bl.node("tf2_ros", "static_transform_publisher",
                name="broadcaster_router",
                cmd_args=toargs(x=router_x, y=router_y, z=router_z, yaw=router_yaw, pitch=0, roll=0,
                                frame_id=base_frame, child_frame_id=f"{name}/router"))
