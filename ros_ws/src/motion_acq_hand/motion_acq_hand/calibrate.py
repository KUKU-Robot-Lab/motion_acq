"""Calibrate one glove side from example poses, or re-zero a saved calibration.

    ros2 run motion_acq_hand calibrate --side right --user op1            # full: every example pose
    ros2 run motion_acq_hand calibrate --side right --user op1 --rezero   # 2 s open hand, keeps the map
    ros2 run motion_acq_hand calibrate --side right --user fake --fake --yes   # fake glove

Full: for each example pose of configs/hands/nova2_to_rh56f1.yaml (Korean prompt) the
operator holds still and presses Enter; the tool records --seconds of glove signals. A
pose that moved, or two poses the glove cannot tell apart, are asked again
(motion_acq.hand.calibration.run_session). Then the glove -> RH56F1 map is fitted and
saved to configs/hands/calibration/<user>_<side>.yaml, which every later session loads.

--rezero (after a SenseCom restart or a glove power cycle, when the hand does not open or
close fully): one open-hand pose shifts the saved map's inputs.

While recording, the glove strap squeezes at feedback.strap_hold (the same as in use, so
the glove sits on the palm the same way). Enters typed while a pose records are dropped.
"""

# ruff: noqa: I001  -- motion_acq_hand.common must be imported first (it puts motion_acq on sys.path)
from __future__ import annotations

import argparse
import sys
import termios
import time
from pathlib import Path

from motion_acq_hand.common import (
    REPO_ROOT,
    check_side,
    glove_qos,
    glove_topic,
    load_gloves,
    require_fake_isolation,
)

import rclpy
from rclpy.node import Node
from senseglove_msgs.msg import SenseGloveState
from std_msgs.msg import Float64MultiArray, String

from motion_acq.hand.calibration import OPEN_POSE, CalibrationError, HandCalibration, fit_error, rezero, run_session
from motion_acq.hand.feedback import HAPTICS_BEAT_S, OFF, haptics_topic_for, heartbeat, strap_only
from motion_acq.hand.nova2 import GloveDataError, angles_from_state, tip_signals
from motion_acq.hand.retarget import DEFAULT_RETARGET, load_hand_retarget_config


class Recorder(Node):
    def __init__(self, topic: str, side: str, strap_level: float) -> None:
        super().__init__("motion_acq_hand_calibrate")
        self.side = side
        self.samples: list[dict[str, float]] | None = None
        self.last_rx = 0.0
        self.create_subscription(SenseGloveState, topic, self._on_glove, glove_qos())
        self.pose_pub = self.create_publisher(String, f"/motion_acq/fake_glove/{side}/pose", 10)
        self.haptics_pub = self.create_publisher(Float64MultiArray, haptics_topic_for(topic), 10)
        self.strap = strap_only(strap_level) if strap_level > 0.0 else None
        self._beat = False
        if self.strap is not None:
            self.create_timer(HAPTICS_BEAT_S, self._hold_strap)

    def _hold_strap(self) -> None:
        self._beat = not self._beat
        self.haptics_pub.publish(Float64MultiArray(data=heartbeat(self.strap, self._beat)))

    def release_strap(self) -> None:
        if self.strap is not None:
            for _ in range(3):
                self.haptics_pub.publish(Float64MultiArray(data=OFF.efforts()))

    def _on_glove(self, msg: SenseGloveState) -> None:
        self.last_rx = time.monotonic()
        if self.samples is None:
            return
        try:
            signals = angles_from_state(list(msg.joint_names), list(msg.position), self.side)
            signals.update(tip_signals([(p.x, p.y, p.z) for p in msg.finger_tip_position]))
        except GloveDataError:
            return
        self.samples.append(signals)


def spin_for(node: Node, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.02)


def record_pose(node: Recorder, seconds: float) -> list[dict[str, float]]:
    node.samples = []
    spin_for(node, seconds)
    samples, node.samples = node.samples, None
    if len(samples) < 10:
        raise CalibrationError(f"{seconds} 초에 장갑 샘플 {len(samples)} 개뿐: 장갑 드라이버가 도는지 확인")
    return samples


def ask_enter(text: str) -> None:
    """Wait for a fresh Enter: drop Enters typed earlier (e.g. during a recording)."""
    if sys.stdin.isatty():
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    input(text)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--glove-serial", default=None,
                    help="default: the side's glove in configs/hands/nova2_gloves.yaml (fake: 0)")
    ap.add_argument("--glove-topic", default="")
    ap.add_argument("--retarget-config", type=Path, default=DEFAULT_RETARGET)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--rezero", action="store_true", help="only the open hand: shift the saved map's inputs")
    ap.add_argument("--fake", action="store_true", help="drive the fake glove to each pose")
    ap.add_argument("--yes", action="store_true", help="no prompts (fake runs)")
    args = ap.parse_args(argv)
    side = check_side(args.side)
    config = load_hand_retarget_config(args.retarget_config)
    out = args.out or REPO_ROOT / "configs" / "hands" / "calibration" / f"{args.user}_{side}.yaml"
    serial = args.glove_serial or ("0" if args.fake else load_gloves()[side].serial)
    topic = args.glove_topic or glove_topic(serial, side)
    if args.fake:
        require_fake_isolation("calibrate --fake")
    examples = {p: (e.prompt, e.target) for p, e in config.examples.items()}
    ask = (lambda text: None) if args.yes else ask_enter

    def say(text: str) -> None:
        print(text, flush=True)

    rclpy.init()
    node = Recorder(topic, side, 0.0 if args.fake else config.feedback.strap_hold)
    try:
        spin_for(node, 1.0)
        if time.monotonic() - node.last_rx > 0.5:
            raise SystemExit(f"no glove samples on {topic}")

        def record(pose: str) -> list[dict[str, float]]:
            if args.fake:
                node.pose_pub.publish(String(data=pose))
                spin_for(node, 0.5)
            return record_pose(node, args.seconds)

        if args.rezero:
            saved = HandCalibration.load(out, side=side)
            ask(f"[편 손 맞춤] {config.examples[OPEN_POSE].prompt}. 자세를 잡고 멈춘 뒤 Enter 를 누르세요 ")
            say("  기록 중: 그대로 멈춰 있으세요")
            calibration = rezero(saved, record(OPEN_POSE))
            calibration.save(out)
            shift = ", ".join(f"{n} {v:+.2f}" for m in calibration.models.values()
                              for n, v in zip(m.inputs, m.offset) if abs(v) > 0.02)
            print(f"편 손 맞춤 저장됨: {out} ({shift or '거의 그대로'})")
            return
        calibration = run_session(side=side, user=args.user, groups=config.groups,
                                  examples=examples, ask=ask, record=record, say=say)
        calibration.save(out)
        print(f"보정 저장됨: {out}")
        print(f"  예시 자세 {len(examples)}개, 예시에서의 최대 오차 {fit_error(calibration, examples):.3f} rad")
    except CalibrationError as exc:
        raise SystemExit(f"calibration failed: {exc}") from exc
    finally:
        node.release_strap()
        if args.fake:
            node.pose_pub.publish(String(data="cycle"))
        spin_for(node, 0.2)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
