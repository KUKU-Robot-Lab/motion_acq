"""Nova 2 glove -> RH56F1 hand node (one side). Own process; arms and head never wait on it.

    ros2 run motion_acq_hand hand_node --ros-args -p side:=right \\
        -p calibration:=<repo>/configs/hands/calibration/<user>_right.yaml

Disabled at start unless enable_on_start is true; enable/disable with
std_msgs/Bool on /motion_acq/hand_<side>/enable. Enabling needs a fresh
angle_actual from the driver (the hand starts from its measured pose), sends
speed_set/force_set once, then streams angle_set at rate_hz. A glove sample
older than stale_s holds the hand (nothing is published). Every cycle is
logged to <log_dir>/hand_<side>_<time>.jsonl and, if udp_target is set, sent
as one JSON datagram (recorder sidecar).
"""

# ruff: noqa: I001  -- motion_acq_hand.common must be imported first (it puts motion_acq on sys.path)
from __future__ import annotations

import json
import socket
import time
from pathlib import Path

from motion_acq_hand.common import spin_node, REPO_ROOT, check_side, glove_qos, glove_topic, hand_ns

from rclpy.node import Node
from rh56f1_interfaces.msg import GetAngleAct1, SetAngle1, SetForce1, SetSpeed1
from senseglove_msgs.msg import SenseGloveState
from std_msgs.msg import Bool, String

from motion_acq.hand.calibration import HandCalibration
from motion_acq.hand.nova2 import GloveDataError, angles_from_state
from motion_acq.hand.retarget import (
    DEFAULT_RETARGET,
    HandRetargeter,
    HandState,
    load_hand_retarget_config,
)
from motion_acq.hand.rh56f1 import DEFAULT_MAP, N_SLOTS, load_rh56f1_map

MEASURED_FRESH_S = 0.5


class HandNode(Node):
    def __init__(self) -> None:
        super().__init__("motion_acq_hand")
        p = self.declare_parameter
        self.side = check_side(p("side", "right").value)
        serial = str(p("glove_serial", "0").value)
        topic = str(p("glove_topic", "").value) or glove_topic(serial, self.side)
        calibration = str(p("calibration", "").value)
        retarget_path = Path(str(p("retarget_config", str(DEFAULT_RETARGET)).value))
        map_path = Path(str(p("hand_map", str(DEFAULT_MAP)).value))
        enable_on_start = bool(p("enable_on_start", False).value)
        log_dir = str(p("log_dir", str(REPO_ROOT / "logs" / "hand")).value)
        udp_target = str(p("udp_target", "").value)
        if not calibration:
            raise SystemExit("calibration:=<file> is required (run motion_acq_hand calibrate first)")

        self.config = load_hand_retarget_config(retarget_path)
        self.hand_map = load_rh56f1_map(map_path)
        self.retargeter = HandRetargeter(
            self.config, HandCalibration.load(Path(calibration), side=self.side), self.hand_map, self.side
        )
        self.enabled = False
        self.want_enable = enable_on_start
        self.glove: tuple[dict[str, float], float] | None = None
        self.glove_errors = 0
        self.measured: tuple[list[int], float] | None = None
        self.last_published: list[int] | None = None

        ns = hand_ns(self.side)
        self.angle_pub = self.create_publisher(SetAngle1, f"{ns}/angle_set", 10)
        self.speed_pub = self.create_publisher(SetSpeed1, f"{ns}/speed_set", 10)
        self.force_pub = self.create_publisher(SetForce1, f"{ns}/force_set", 10)
        self.status_pub = self.create_publisher(String, f"/motion_acq/hand_{self.side}/status", 10)
        self.create_subscription(SenseGloveState, topic, self._on_glove, glove_qos())
        self.create_subscription(GetAngleAct1, f"{ns}/angle_actual", self._on_actual, 10)
        self.create_subscription(Bool, f"/motion_acq/hand_{self.side}/enable", self._on_enable, 10)

        self.log_file = self._open_log(Path(log_dir))
        self.udp = None
        if udp_target:
            host, port = udp_target.rsplit(":", 1)
            self.udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port)))
        self.create_timer(1.0 / self.config.rate_hz, self._tick)
        self.get_logger().info(
            f"hand {self.side}: glove {topic} -> {ns}/angle_set at {self.config.rate_hz:g} Hz "
            f"({'enable on start' if enable_on_start else 'disabled until enabled'})"
        )

    def _open_log(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"hand_{self.side}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.get_logger().info(f"hand log: {path}")
        return path.open("w", encoding="utf-8", buffering=1)

    def _on_glove(self, msg: SenseGloveState) -> None:
        try:
            angles = angles_from_state(list(msg.joint_names), list(msg.position), self.side)
        except GloveDataError as exc:
            self.glove_errors += 1
            self.get_logger().warning(f"bad glove sample: {exc}", throttle_duration_sec=2.0)
            return
        self.glove = (angles, time.monotonic())

    def _on_actual(self, msg: GetAngleAct1) -> None:
        values = [int(v) for v in msg.joint_values]  # numpy int32 -> int (JSON log)
        if len(values) == N_SLOTS:
            self.measured = (values, time.monotonic())

    def _on_enable(self, msg: Bool) -> None:
        self.want_enable = bool(msg.data)
        if not msg.data and self.enabled:
            self.enabled = False
            self.retargeter.state = HandState.IDLE
            self.get_logger().info("hand disabled (the hand keeps its last command)")

    def _try_enable(self, now: float) -> None:
        if self.measured is None or now - self.measured[1] > MEASURED_FRESH_S:
            self.get_logger().warning(
                "cannot enable: no fresh angle_actual from the RH56F1 driver", throttle_duration_sec=2.0
            )
            return
        measured_rad = self.hand_map.to_rad(self.measured[0], side=self.side)
        self.retargeter.start(measured_rad, now)
        speed, force = SetSpeed1(), SetForce1()
        speed.hand_id = force.hand_id = 1
        speed.joint_values = [self.config.driver_speed] * N_SLOTS
        force.joint_values = [self.config.driver_force] * N_SLOTS
        self.speed_pub.publish(speed)
        self.force_pub.publish(force)
        self.enabled = True
        self.get_logger().info(
            f"hand enabled from measured {self.measured[0]} "
            f"(speed {self.config.driver_speed}, force {self.config.driver_force})"
        )

    def _tick(self) -> None:
        now = time.monotonic()
        if self.want_enable and not self.enabled:
            self._try_enable(now)
        glove = self.glove
        glove_age = None if glove is None else now - glove[1]
        angles = glove[0] if glove is not None and now - glove[1] <= self.config.stale_s else None
        step = self.retargeter.step(angles, now)
        if step.state is HandState.RUNNING and step.registers is not None:
            msg = SetAngle1()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.hand_id = 1
            msg.joint_values = list(step.registers)
            self.angle_pub.publish(msg)
            self.last_published = list(step.registers)
        record = {
            "t_mono_s": round(now, 6),
            "side": self.side,
            "state": step.state.value,
            "enabled": self.enabled,
            "glove_age_s": None if glove_age is None else round(glove_age, 4),
            "features": step.features,
            "normalized": step.normalized,
            "q_target_rad": step.q_target,
            "q_command_rad": step.q_command,
            "registers": step.registers if step.state is HandState.RUNNING else None,
            "measured_registers": None if self.measured is None else self.measured[0],
        }
        line = json.dumps(record)
        self.log_file.write(line + "\n")
        if self.udp is not None:
            self.udp[0].sendto(line.encode("utf-8"), self.udp[1])
        self.status_pub.publish(String(data=json.dumps(
            {k: record[k] for k in ("state", "enabled", "glove_age_s", "registers", "measured_registers")}
        )))

    def destroy_node(self) -> None:
        self.log_file.close()
        super().destroy_node()


def main() -> None:
    spin_node(HandNode)


if __name__ == "__main__":
    main()
