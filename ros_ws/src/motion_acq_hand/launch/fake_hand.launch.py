"""Fake STEP 3 chain: fake Nova 2 -> hand_node -> fake RH56F1 (no hardware).

    ros2 launch motion_acq_hand fake_hand.launch.py side:=right calibration:=<yaml>
"""

from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    side = LaunchConfiguration("side")
    args = [
        DeclareLaunchArgument("side", default_value="right"),
        DeclareLaunchArgument("calibration"),
        DeclareLaunchArgument("glove_mode", default_value="cycle"),
        DeclareLaunchArgument("dropout_every_s", default_value="0.0"),
        DeclareLaunchArgument("dropout_s", default_value="0.0"),
        DeclareLaunchArgument("enable_on_start", default_value="true"),
        DeclareLaunchArgument("udp_target", default_value=""),
        DeclareLaunchArgument("fake_domain", default_value="177"),
    ]
    # Fake nodes publish the real driver topic names: keep them localhost-only on
    # the dedicated fake domain (the fake nodes refuse anything else).
    isolation = [
        SetEnvironmentVariable("ROS_DOMAIN_ID", LaunchConfiguration("fake_domain")),
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
    ]
    return LaunchDescription(args + isolation + [
        Node(package="motion_acq_hand", executable="fake_glove", name="fake_glove", output="screen",
             parameters=[{"side": side, "mode": LaunchConfiguration("glove_mode"),
                          "dropout_every_s": LaunchConfiguration("dropout_every_s"),
                          "dropout_s": LaunchConfiguration("dropout_s")}]),
        Node(package="motion_acq_hand", executable="fake_rh56f1", name="fake_rh56f1", output="screen",
             parameters=[{"side": side}]),
        Node(package="motion_acq_hand", executable="hand_node", name="motion_acq_hand", output="screen",
             parameters=[{"side": side, "calibration": LaunchConfiguration("calibration"),
                          "enable_on_start": LaunchConfiguration("enable_on_start"),
                          "udp_target": LaunchConfiguration("udp_target")}]),
    ])
