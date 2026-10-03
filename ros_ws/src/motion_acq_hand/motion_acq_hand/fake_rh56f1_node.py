"""Fake RH56F1 driver: same topics as the EtherCAT node, registers move at a fixed rate.

    ros2 run motion_acq_hand fake_rh56f1 --ros-args -p side:=right

Subscribes /hand_<side>/angle_set (SetAngle1, -1 = leave the axis), speed_set,
force_set; publishes /hand_<side>/angle_actual (GetAngleAct1).
"""

# ruff: noqa: I001  -- motion_acq_hand.common must be imported first (it puts motion_acq on sys.path)
from __future__ import annotations

from motion_acq_hand.common import spin_node, check_side, hand_ns, require_fake_isolation

from rclpy.node import Node
from rh56f1_interfaces.msg import GetAngleAct1, SetAngle1, SetForce1, SetSpeed1

from motion_acq.hand.retarget import load_hand_retarget_config
from motion_acq.hand.rh56f1 import N_SLOTS, load_rh56f1_map

SLOT_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"]


class FakeRh56f1(Node):
    def __init__(self) -> None:
        super().__init__("fake_rh56f1")
        self.side = check_side(self.declare_parameter("side", "right").value)
        rate_hz = float(self.declare_parameter("rate_hz", 100.0).value)
        # ~135 deg/s measured at speed 2000 (sim2real 09.30) ~ 1350 registers/s.
        self.reg_per_s = float(self.declare_parameter("registers_per_s", 1350.0).value)
        home = load_rh56f1_map().to_registers(load_hand_retarget_config().home_rad, side=self.side)
        self.present = [float(v) for v in home]
        self.target = list(self.present)
        self.speed: list[int] | None = None
        self.force: list[int] | None = None
        ns = hand_ns(self.side)
        self.pub = self.create_publisher(GetAngleAct1, f"{ns}/angle_actual", 10)
        self.create_subscription(SetAngle1, f"{ns}/angle_set", self._on_angle, 10)
        self.create_subscription(SetSpeed1, f"{ns}/speed_set", self._on_speed, 10)
        self.create_subscription(SetForce1, f"{ns}/force_set", self._on_force, 10)
        self.dt = 1.0 / rate_hz
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(f"fake RH56F1 {self.side} on {ns}")

    def _on_angle(self, msg: SetAngle1) -> None:
        for i, value in enumerate(list(msg.joint_values)[:N_SLOTS]):
            if value != -1:
                self.target[i] = float(value)

    def _on_speed(self, msg: SetSpeed1) -> None:
        self.speed = list(msg.joint_values)

    def _on_force(self, msg: SetForce1) -> None:
        self.force = list(msg.joint_values)

    def _tick(self) -> None:
        step = self.reg_per_s * self.dt
        for i in range(N_SLOTS):
            err = self.target[i] - self.present[i]
            self.present[i] += max(-step, min(step, err))
        msg = GetAngleAct1()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.hand_id = 1
        msg.joint_values = [int(round(v)) for v in self.present]
        msg.joint_names = SLOT_NAMES
        self.pub.publish(msg)


def main() -> None:
    require_fake_isolation("fake RH56F1")
    spin_node(FakeRh56f1)


if __name__ == "__main__":
    main()
