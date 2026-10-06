"""Fake RH56F1 driver: same topics as the EtherCAT node, registers move at a fixed rate.

    ros2 run motion_acq_hand fake_rh56f1 --ros-args -p side:=right

Subscribes /hand_<side>/angle_set (SetAngle1, -1 = leave the axis) and angle_target (same + the driver's
per-finger admittance), speed_set,
force_set; publishes /hand_<side>/angle_actual (GetAngleAct1), force_actual
(GetForceAct1) and touch_data (TouchData1).

-p object_index_reg:=1300 puts an object in the index finger's way: it stops
at that register (closing lowers it) and then reports a tip force of 2.5 N
and a joint force of 400, as a grasp would (tests the glove feedback).
"""

# ruff: noqa: I001  -- motion_acq_hand.common must be imported first (it puts motion_acq on sys.path)
from __future__ import annotations

from motion_acq_hand.common import check_side, declare, hand_ns, require_fake_isolation, spin_node

from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from rh56f1_interfaces.msg import GetAngleAct1, GetCurrentAct1, GetForceAct1, SetAngle1, SetForce1, SetSpeed1, TouchData1

import os
import sys
from pathlib import Path

from motion_acq.hand.retarget import load_hand_retarget_config
from motion_acq.hand.rh56f1 import N_SLOTS, load_rh56f1_map



def _hand_contract():
    """robot_control.rh56f1_hand: the driver's canonical admittance (contract components/rh56f1.yaml)."""
    src = Path(os.environ.get("RL_WS", Path.home() / "rl_ws")) / "robot_control" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import robot_control.rh56f1_hand as hand  # noqa: PLC0415

    return hand


SLOT_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"]
SLOT_JOINTS = ["pinky_1", "ring_1", "middle_1", "index_1", "thumb_2", "thumb_1"]  # EtherCAT node joint_names
INDEX_SLOT = 3
CONTACT_TIP_COUNTS = 250  # 2.5 N
CONTACT_JOINT_FORCE = 400  # g at first touch; grows with how far the target is past the object
PRESS_G_PER_REG = 8.0  # like the real hand pushing towards a target inside the cup (10.06: up to 1715 g)
MA_PER_G = 0.75  # 10.06 cup grasp: ~1300 mA at ~1700 g


class FakeRh56f1(Node):
    def __init__(self) -> None:
        super().__init__("fake_rh56f1")
        self.side = check_side(str(declare(self, "side", "right")))
        rate_hz = float(declare(self, "rate_hz", 100.0))
        # ~135 deg/s measured at speed 2000 (sim2real 09.30) ~ 1350 registers/s.
        self.reg_per_s = float(declare(self, "registers_per_s", 1350.0))
        # Like the vendor driver: commands for another Hand_ID are ignored (0 = broadcast).
        self.hand_id = int(declare(self, "hand_id", 1))
        self.ignored = 0
        self.object_index_reg = int(declare(self, "object_index_reg", -1))
        self.ticks = 0
        home = load_rh56f1_map().to_registers(load_hand_retarget_config().home_rad, side=self.side)
        self.present = [float(v) for v in home]
        self.target = list(self.present)
        self.speed: list[int] | None = None
        self.force: list[int] | None = None
        ns = hand_ns(self.side)
        self.pub = self.create_publisher(GetAngleAct1, f"{ns}/angle_actual", 10)
        self.force_pub = self.create_publisher(GetForceAct1, f"{ns}/force_actual", 10)
        self.touch_pub = self.create_publisher(TouchData1, f"{ns}/touch_data", 10)
        self.current_pub = self.create_publisher(GetCurrentAct1, f"{ns}/current_actual", 10)
        prefix = "r" if self.side == "right" else "l"
        self.joint_names = [f"{prefix}_hj_{j}" for j in SLOT_JOINTS]
        self.create_subscription(SetAngle1, f"{ns}/angle_set", self._on_angle, 10)
        # angle_target: the driver's per-finger admittance (sim2real rh56f1_admittance.h, mirrored below)
        self.create_subscription(SetAngle1, f"{ns}/angle_target", self._on_target, 10)
        self.adm_on = [False] * N_SLOTS
        self.hand = _hand_contract()
        self.adm_params = self.hand.load_admittance()
        self.adm_state = self.hand.AdmState()
        self.adm_state.bias = [10.0] * N_SLOTS
        self.last_tips = [0] * 5
        self.last_force = [0.0] * N_SLOTS
        self.adm_pub = self.create_publisher(Int32MultiArray, f"{ns}/admittance_offset", 10)
        self.create_subscription(SetSpeed1, f"{ns}/speed_set", self._on_speed, 10)
        self.create_subscription(SetForce1, f"{ns}/force_set", self._on_force, 10)
        self.dt = 1.0 / rate_hz
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(f"fake RH56F1 {self.side} on {ns}")

    def _accept(self, msg) -> bool:
        if msg.hand_id in (0, self.hand_id):
            return True
        self.ignored += 1
        self.get_logger().warning(
            f"ignoring hand_id={msg.hand_id} (this hand is {self.hand_id})", throttle_duration_sec=2.0
        )
        return False

    def _on_angle(self, msg: SetAngle1) -> None:
        if not self._accept(msg):
            return
        for i, value in enumerate(list(msg.joint_values)[:N_SLOTS]):
            if value != -1:
                self.target[i] = float(value)
                self.adm_on[i] = False
                self.adm_state.y[i], self.adm_state.limiting[i] = 0.0, 0

    def _on_target(self, msg: SetAngle1) -> None:
        if not self._accept(msg):
            return
        for i, value in enumerate(list(msg.joint_values)[:N_SLOTS]):
            if value != -1:
                self.target[i] = float(value)
                self.adm_on[i] = i < 5  # thumb_1 stays position (as the driver default)

    def _on_speed(self, msg: SetSpeed1) -> None:
        if self._accept(msg):
            self.speed = list(msg.joint_values)

    def _on_force(self, msg: SetForce1) -> None:
        if self._accept(msg):
            self.force = list(msg.joint_values)

    def _sent(self, i: int) -> float:
        """The register the driver would send: target, or the driver's admittance (robot_control.rh56f1_hand,
        contract values; rest bias 10 g as this fake's free reading)."""
        if not self.adm_on[i]:
            return self.target[i]
        tip = float(self.last_tips[i]) if i < 5 else -1.0
        return self.hand.adm_step(self.adm_params, self.adm_state, i, self.dt, self.target[i], self.present[i],
                                  self.last_force[i], tip)

    def _tick(self) -> None:
        step = self.reg_per_s * self.dt
        sent = [self._sent(i) for i in range(N_SLOTS)]
        for i in range(N_SLOTS):
            err = sent[i] - self.present[i]
            self.present[i] += max(-step, min(step, err))
        pressing = False
        if self.object_index_reg >= 0 and self.present[INDEX_SLOT] < self.object_index_reg:
            self.present[INDEX_SLOT] = float(self.object_index_reg)
            pressing = sent[INDEX_SLOT] < self.object_index_reg
        stamp = self.get_clock().now().to_msg()
        force = GetForceAct1()
        force.header.stamp = stamp
        force.hand_id = self.hand_id
        self.ticks += 1
        noise = self.ticks % 3  # live readings are never exactly still (hand_node faults a frozen hand)
        press = CONTACT_JOINT_FORCE + PRESS_G_PER_REG * (self.object_index_reg - sent[INDEX_SLOT])
        values = [int(min(press, 1800)) + noise if pressing and i == INDEX_SLOT else 10 + noise for i in range(N_SLOTS)]
        force.joint_values = values
        self.last_force = [float(v) for v in values]
        force.joint_names = self.joint_names
        self.force_pub.publish(force)
        current = GetCurrentAct1()
        current.header.stamp = stamp
        current.hand_id = self.hand_id
        current.joint_values = [int(v * MA_PER_G) if v > 100 else 60 + noise for v in values]
        current.joint_names = self.joint_names
        self.current_pub.publish(current)
        touch = TouchData1()
        touch.header.stamp = stamp
        touch.finger_forces = [CONTACT_TIP_COUNTS if pressing and f == "index" else 0
                               for f in ("pinky", "ring", "middle", "index", "thumb")]
        self.last_tips = list(touch.finger_forces)
        touch.palm_data = [0] * 9
        self.touch_pub.publish(touch)
        self.adm_pub.publish(Int32MultiArray(data=[int(round(sent[i] - self.target[i])) if self.adm_on[i] else 0
                                                   for i in range(N_SLOTS)]))
        msg = GetAngleAct1()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.hand_id = self.hand_id
        msg.joint_values = [int(round(v)) for v in self.present]
        msg.joint_names = SLOT_NAMES
        self.pub.publish(msg)


def main() -> None:
    require_fake_isolation("fake RH56F1")
    spin_node(FakeRh56f1)


if __name__ == "__main__":
    main()
