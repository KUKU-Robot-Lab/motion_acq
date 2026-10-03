"""Fake Nova 2: publishes SenseGloveState like senseglove_ros, no glove needed.

    ros2 run motion_acq_hand fake_glove --ros-args -p side:=right -p mode:=cycle

mode cycle: curl, thumb bend and thumb opposition follow slow sines.
mode pose:  hold a calibration pose (open | fist | thumb_opposed); switch it by
            publishing a pose name on /motion_acq/fake_glove/<side>/pose.
dropout_every_s / dropout_s: stop publishing periodically (glove loss).
"""

# ruff: noqa: I001  -- motion_acq_hand.common must be imported first (it puts motion_acq on sys.path)
from __future__ import annotations

import math
import time

from motion_acq_hand.common import spin_node, check_side, glove_qos, glove_topic, require_fake_isolation

from geometry_msgs.msg import Quaternion
from rclpy.node import Node
from senseglove_msgs.msg import KinematicsVect3D, SenseGloveState
from std_msgs.msg import String

from motion_acq.hand.synthetic import POSE_ANGLES, synthetic_angles, synthetic_state


class FakeGlove(Node):
    def __init__(self) -> None:
        super().__init__("fake_glove")
        self.side = check_side(self.declare_parameter("side", "right").value)
        serial = str(self.declare_parameter("serial", "0").value)
        self.mode = str(self.declare_parameter("mode", "cycle").value)
        self.pose = str(self.declare_parameter("pose", "open").value)
        self.period_s = float(self.declare_parameter("period_s", 6.0).value)
        self.dropout_every_s = float(self.declare_parameter("dropout_every_s", 0.0).value)
        self.dropout_s = float(self.declare_parameter("dropout_s", 0.0).value)
        rate_hz = float(self.declare_parameter("rate_hz", 60.0).value)
        if self.mode not in ("cycle", "pose") or self.pose not in POSE_ANGLES:
            raise SystemExit(f"mode cycle|pose and pose in {sorted(POSE_ANGLES)}")
        self.pub = self.create_publisher(SenseGloveState, glove_topic(serial, self.side), glove_qos())
        self.create_subscription(String, f"/motion_acq/fake_glove/{self.side}/pose", self._on_pose, 10)
        self.t0 = time.monotonic()
        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(f"fake {self.side} glove on {glove_topic(serial, self.side)} ({self.mode})")

    def _on_pose(self, msg: String) -> None:
        if msg.data in POSE_ANGLES:
            self.mode, self.pose = "pose", msg.data
            self.get_logger().info(f"fake glove pose -> {msg.data}")
        elif msg.data == "cycle":
            self.mode = "cycle"

    def _angles(self, t: float) -> dict[str, float]:
        if self.mode == "pose":
            return POSE_ANGLES[self.pose]
        phase = 2.0 * math.pi * t / self.period_s
        curl = 0.5 - 0.5 * math.cos(phase)
        bend = 0.5 - 0.5 * math.cos(phase + 0.7)
        opposition = 0.5 - 0.5 * math.cos(0.5 * phase)
        return synthetic_angles(curl, bend, opposition)

    def _tick(self) -> None:
        t = time.monotonic() - self.t0
        if self.dropout_every_s > 0 and (t % self.dropout_every_s) >= self.dropout_every_s - self.dropout_s:
            return
        names, positions = synthetic_state(self.side, self._angles(t))
        msg = SenseGloveState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.joint_names = names
        msg.position = [float(p) for p in positions]
        msg.absolute_velocity = [0.0] * len(names)
        msg.hand_position = [KinematicsVect3D() for _ in range(20)]
        msg.finger_tip_position = [KinematicsVect3D() for _ in range(5)]
        msg.imu_orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.pub.publish(msg)


def main() -> None:
    require_fake_isolation("fake glove")
    spin_node(FakeGlove)


if __name__ == "__main__":
    main()
