"""Nova 2 glove -> RH56F1 hand node (one side). Own process; arms and head never wait on it.

    ros2 run motion_acq_hand hand_node --ros-args -p side:=right \\
        -p calibration:=<repo>/configs/hands/calibration/<user>_right.yaml \\
        -p amplitude:=0.3 -p driver_speed:=1000 -p driver_force:=300

Disabled at start; enable/disable with std_msgs/Bool on
/motion_acq/hand_<side>/enable. The enable/fault rules live in
motion_acq.hand.controller (fresh, plausible angle_actual; all driver topics
subscribed; hand_id taken from angle_actual; speed/force re-sent every second;
angle_actual loss -> latched FAULT, the hand stays put). Start and end pose is
home (hand open): enable walks home before following the glove; disable, Ctrl+C
and SIGTERM walk home before stopping (a second signal skips the return).
enable_on_start is only accepted on the isolated fake domain. Every cycle is
logged to <log_dir>/hand_<side>_<time>.jsonl and, if udp_target is set, sent
as one JSON datagram to each HOST:PORT of that comma-separated list (recorder
sidecar, macq console).

Feedback (10.05): the RH56F1 tip / palm forces (touch_data) and joint forces
(force_actual) drive the Nova 2 brakes, strap and vibration through the glove
driver's haptics_controller (forward_command_controller, percent per joint,
motion_acq_hand config/nova2_<side>_controllers.yaml); motion_acq.hand.feedback
has the rules. haptics:=false turns it off. All off when not following the
glove and when the node exits.
"""

# ruff: noqa: I001  -- motion_acq_hand.common must be imported first (it puts motion_acq on sys.path)
from __future__ import annotations

import json
import socket
import time
from pathlib import Path

from motion_acq_hand.common import (
    REPO_ROOT,
    check_side,
    declare,
    declare_glove_topic,
    fake_isolated,
    glove_qos,
    hand_ns,
    keep_off_rt,
)

import rclpy
from rclpy.node import Node
from rh56f1_interfaces.msg import GetAngleAct1, GetForceAct1, SetAngle1, SetForce1, SetSpeed1, TouchData1
from senseglove_msgs.msg import SenseGloveState
from std_msgs.msg import Bool, Float64MultiArray, String

from motion_acq.hand.calibration import HandCalibration
from motion_acq.hand.controller import ControllerConfig, HandController
from motion_acq.hand.feedback import HAPTICS_BEAT_S, HAPTICS_MIN_PERIOD_S, OFF, haptics_topic_for, heartbeat
from motion_acq.hand.nova2 import GloveDataError, angles_from_state, tip_signals
from motion_acq.hand.retarget import DEFAULT_RETARGET, HandRetargeter, load_hand_retarget_config
from motion_acq.hand.rh56f1 import DEFAULT_MAP, load_rh56f1_map
from motion_acq.sidecar import parse_udp_targets


class HandNode(Node):
    def __init__(self) -> None:
        super().__init__("motion_acq_hand")
        self.side = check_side(str(declare(self, "side", "right")))
        topic = declare_glove_topic(self, self.side)
        calibration = str(declare(self, "calibration", ""))
        retarget_path = Path(str(declare(self, "retarget_config", str(DEFAULT_RETARGET))))
        map_path = Path(str(declare(self, "hand_map", str(DEFAULT_MAP))))
        enable_on_start = bool(declare(self, "enable_on_start", False))
        amplitude = float(declare(self, "amplitude", 1.0))
        log_dir = Path(str(declare(self, "log_dir", str(REPO_ROOT / "logs" / "hand"))))
        udp_target = str(declare(self, "udp_target", ""))
        haptics = bool(declare(self, "haptics", True))
        haptics_topic = str(declare(self, "haptics_topic", haptics_topic_for(topic)))
        if not calibration:
            raise SystemExit("calibration:=<file> is required (run motion_acq_hand calibrate first)")
        if enable_on_start and not fake_isolated():
            raise SystemExit("enable_on_start is only allowed on the isolated fake domain")

        config = load_hand_retarget_config(retarget_path)
        speed = int(declare(self, "driver_speed", config.driver_speed))
        force = int(declare(self, "driver_force", config.driver_force))
        hand_map = load_rh56f1_map(map_path)
        retargeter = HandRetargeter(
            config, HandCalibration.load(Path(calibration), side=self.side), hand_map, self.side,
            amplitude=amplitude,
        )
        self.controller = HandController(retargeter, hand_map, self.side, ControllerConfig(
            glove_stale_s=config.stale_s, driver_speed=speed, driver_force=force,
        ))
        self.controller.request_enable(enable_on_start)

        ns = hand_ns(self.side)
        self.angle_pub = self.create_publisher(SetAngle1, f"{ns}/angle_set", 10)
        self.speed_pub = self.create_publisher(SetSpeed1, f"{ns}/speed_set", 10)
        self.force_pub = self.create_publisher(SetForce1, f"{ns}/force_set", 10)
        self.status_pub = self.create_publisher(String, f"/motion_acq/hand_{self.side}/status", 10)
        self.create_subscription(SenseGloveState, topic, self._on_glove, glove_qos())
        self.create_subscription(GetAngleAct1, f"{ns}/angle_actual", self._on_actual, 10)
        self.create_subscription(Bool, f"/motion_acq/hand_{self.side}/enable", self._on_enable, 10)
        self.create_subscription(TouchData1, f"{ns}/touch_data", self._on_touch, 10)
        self.create_subscription(GetForceAct1, f"{ns}/force_actual", self._on_force, 10)
        self.create_subscription(String, f"{ns}/ecat_status", self._on_status, 10)
        self.haptics_pub = self.create_publisher(Float64MultiArray, haptics_topic, 10) if haptics else None
        self._haptics_sent: tuple[list[float], float] | None = None
        self._haptics_beat = False

        self.log_file = self._open_log(log_dir)
        # recorder sidecar and the console: comma-separated HOST:PORT list
        self.udp_targets = parse_udp_targets(udp_target)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.udp_targets else None
        self._last_mode = self.controller.mode
        self.create_timer(1.0 / config.rate_hz, self._tick)
        self.get_logger().info(
            f"hand {self.side}: glove {topic} -> {ns}/angle_set at {config.rate_hz:g} Hz, "
            f"amplitude {amplitude:g}, speed {speed}, force {force}, "
            f"haptics {haptics_topic if haptics else 'off'} "
            f"({'enable on start (fake)' if enable_on_start else 'disabled until enabled'})"
        )

    def _open_log(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"hand_{self.side}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.get_logger().info(f"hand log: {path}")
        return path.open("w", encoding="utf-8", buffering=1)

    def _on_glove(self, msg: SenseGloveState) -> None:
        try:
            angles = angles_from_state(list(msg.joint_names), list(msg.position), self.side)
            angles.update(tip_signals([(p.x, p.y, p.z) for p in msg.finger_tip_position]))
        except GloveDataError as exc:
            self.controller.on_glove_error()
            self.get_logger().warning(f"bad glove sample: {exc}", throttle_duration_sec=2.0)
            return
        self.controller.on_glove(angles, time.monotonic())

    def _on_actual(self, msg: GetAngleAct1) -> None:
        self.controller.on_measured([int(v) for v in msg.joint_values], int(msg.hand_id), time.monotonic())

    def _on_touch(self, msg: TouchData1) -> None:
        try:
            self.controller.on_touch(list(msg.finger_forces), list(msg.palm_data), time.monotonic())
        except ValueError as exc:
            self.get_logger().warning(f"bad touch_data: {exc}", throttle_duration_sec=2.0)

    def _on_status(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
        except ValueError:
            status = {"raw": msg.data}
        self.controller.on_status(status if isinstance(status, dict) else {"raw": status})

    def _on_force(self, msg: GetForceAct1) -> None:
        self.controller.on_joint_force(list(msg.joint_names), list(msg.joint_values), time.monotonic())

    def send_haptics(self, efforts: list[float], t: float, *, force: bool = False) -> None:
        """Glove haptics at most at the glove rate (60 Hz). While any level is on it is re-sent every
        HAPTICS_BEAT_S with the on levels alternately 0.01 % lower: the patched glove driver releases
        a command that has not changed for 1 s, so the glove lets go if this node dies."""
        if self.haptics_pub is None:
            return
        last = self._haptics_sent
        changed = last is None or efforts != last[0]
        if not force and last is not None:
            if t - last[1] < HAPTICS_MIN_PERIOD_S:
                return
            if not changed and (not any(efforts) or t - last[1] < HAPTICS_BEAT_S):
                return
        self._haptics_beat = not self._haptics_beat
        data = heartbeat(efforts, self._haptics_beat)
        self.haptics_pub.publish(Float64MultiArray(data=data))
        self._haptics_sent = (list(efforts), t)

    def _on_enable(self, msg: Bool) -> None:
        self.get_logger().info(f"enable request: {bool(msg.data)}")
        self.controller.request_enable(bool(msg.data))

    def _subscribers_ready(self) -> bool:
        return all(p.get_subscription_count() > 0 for p in (self.angle_pub, self.speed_pub, self.force_pub))

    def _publish(self, publisher, msg_type, hand_id: int, values: list[int]) -> None:
        msg = msg_type()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.hand_id = hand_id
        msg.joint_values = values
        publisher.publish(msg)

    def _tick(self) -> None:
        out = self.controller.tick(time.monotonic(), subscribers_ready=self._subscribers_ready())
        if out.hand_id is not None:
            if out.speed is not None:
                self._publish(self.speed_pub, SetSpeed1, out.hand_id, out.speed)
            if out.force is not None:
                self._publish(self.force_pub, SetForce1, out.hand_id, out.force)
            if out.angle is not None:
                self._publish(self.angle_pub, SetAngle1, out.hand_id, out.angle)
        self.send_haptics(out.haptics.efforts(), time.monotonic())
        mode = self.controller.mode
        if mode is not self._last_mode:
            self._last_mode = mode
            detail = self.controller.fault_reason or ""
            # one call site per severity: rclpy refuses a site that changes severity (crashed the node
            # when the fake driver stopped first and the hand faulted after an info here)
            if detail:
                self.get_logger().error(f"hand {self.side} -> {mode.value} {detail}")
            else:
                self.get_logger().info(f"hand {self.side} -> {mode.value}")
        if out.record.get("refusal"):
            self.get_logger().warning(f"enable refused: {out.record['refusal']}", throttle_duration_sec=2.0)
        line = json.dumps(out.record)
        try:
            self.log_file.write(line + "\n")
            if self.udp is not None:
                for target in self.udp_targets:
                    self.udp.sendto(line.encode("utf-8"), target)
        except OSError as exc:  # logging must never stop the control loop
            self.get_logger().warning(f"hand log/udp write failed: {exc}", throttle_duration_sec=5.0)
        self.status_pub.publish(String(data=json.dumps({
            k: out.record[k] for k in ("mode", "state", "fault", "refusal", "glove_age_s", "glove_frozen", "glove_errors",
                                       "registers", "measured_registers")
        })))

    def return_home(self, interrupted) -> None:
        """End pose = home: disable walks the hand home; spin until it is there."""
        if not self.controller.want_enable and not self.controller.busy:
            return
        self.get_logger().info(f"hand {self.side}: returning home before exit")
        self.controller.request_enable(False)
        deadline = time.monotonic() + self.controller.config.home_timeout_s + 1.0
        while self.controller.busy and time.monotonic() < deadline:
            if interrupted():
                self.get_logger().warning(f"hand {self.side}: second stop, leaving the hand where it is")
                return
            rclpy.spin_once(self, timeout_sec=0.02)
        state = "at home" if not self.controller.busy else "home not reached"
        self.get_logger().info(f"hand {self.side}: {state}")

    def destroy_node(self) -> None:
        try:
            for _ in range(3):  # the glove controller holds the last command: leave it off
                self.send_haptics(OFF.efforts(), time.monotonic(), force=True)
            self.log_file.close()
        finally:
            super().destroy_node()


def main() -> None:
    """Spin until SIGINT/SIGTERM, then walk the hand home before exiting.

    rclpy's own signal handler would shut the context down at the first
    Ctrl+C, leaving no way to publish the return; signals are handled here
    instead. A second signal skips the return (the hand stays where it is).
    """
    import signal
    import threading

    from rclpy.signals import SignalHandlerOptions

    print(keep_off_rt(), flush=True)  # off the RH56F1 EtherCAT cores (sim2real cpu plan)
    stops = threading.Semaphore(0)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stops.release())
    node = None
    try:
        node = HandNode()
        while not stops.acquire(blocking=False):
            rclpy.spin_once(node, timeout_sec=0.05)
        node.return_home(lambda: stops.acquire(blocking=False))
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
