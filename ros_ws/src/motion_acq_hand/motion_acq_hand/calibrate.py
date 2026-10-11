"""Calibrate one glove side from example poses, or re-zero a saved calibration.

    ros2 run motion_acq_hand calibrate --side right --user op1            # full: every example pose
    ros2 run motion_acq_hand calibrate --side right --user op1 --rezero   # 2 s open hand, keeps the map
    ros2 run motion_acq_hand calibrate --side right --user op1 --fingers  # one finger at a time (10.11)
    ros2 run motion_acq_hand calibrate --side right --user fake --fake --yes   # fake glove

Full: for each example pose of configs/hands/nova2_to_rh56f1.yaml (Korean prompt) the
operator holds still and presses Enter; the tool records --seconds of glove signals. A
pose that moved, or two poses the glove cannot tell apart, are asked again
(motion_acq.hand.calibration.run_session). Then the glove -> RH56F1 map is fitted and
saved to configs/hands/calibration/<user>_<side>.yaml, which every later session loads.

--rezero (after a SenseCom restart or a glove power cycle, when the hand does not open or
close fully): one open-hand pose shifts the saved map's inputs.

--fingers ([손가락별 맞춤], 10.11): thumb, index, middle, ring+pinky, then the fist, one at a time: the
open hand still for a second, then that finger bent and straightened a few times. Each robot finger
then goes 0..1 with that finger alone and the glove's crosstalk between fingers is removed
(motion_acq.hand.sweeps); stored in the saved calibration (finger_sweeps), which it needs first.

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

from motion_acq.hand.calibration import (
    OPEN_POSE, CalibrationError, HandCalibration, fit_error, rezero, run_session, thumb_of)
from motion_acq.hand.feedback import HAPTICS_BEAT_S, OFF, haptics_topic_for, heartbeat, strap_only
from motion_acq.hand.kinematic import glove_points
from motion_acq.hand.nova2 import GloveDataError, angles_from_state, tip_signals
from motion_acq.hand.retarget import DEFAULT_RETARGET, load_hand_retarget_config
from motion_acq.hand.sweeps import MIN_MOVE, SOURCE_KO, SWEEPS, SweepError, fit_sweeps, report, travel, unsteady

# fake glove poses a sweep alternates between (after the open hand)
FAKE_SWEEP_POSES = {"thumb": ("thumb_bend", "thumb_opposed"), "index": ("index",), "middle": ("middle",),
                    "ring": ("ring",), "fist": ("fist",)}


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
            signals.update(glove_points([(p.x, p.y, p.z) for p in msg.hand_position],
                                     [(p.x, p.y, p.z) for p in msg.finger_tip_position]))
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


def record_sweep(node: Recorder, seconds: float, fake_poses: tuple[str, ...] | None) -> list[dict[str, float]]:
    """The moving part of a sweep; a fake glove steps through flat and the sweep's poses each 0.5 s."""
    node.samples = []
    end, step = time.monotonic() + seconds, 0
    while time.monotonic() < end:
        if fake_poses is not None:
            cycle = ("flat",) + fake_poses
            node.pose_pub.publish(String(data=cycle[step % len(cycle)]))
            step += 1
        spin_for(node, min(0.5, max(end - time.monotonic(), 0.0)))
    samples, node.samples = node.samples, None
    if len(samples) < 20:
        raise CalibrationError(f"{seconds} 초에 장갑 샘플 {len(samples)} 개뿐: 장갑 드라이버가 도는지 확인")
    return samples


def finger_sweeps(node: Recorder, side: str, ask, say, *, fake: bool, rest_s: float, move_s: float,
                  max_tries: int = 3):
    """[손가락별 맞춤]: the five sweeps, each retried when the open hand moved or the finger hardly did."""
    side_ko = "오른손" if side == "right" else "왼손"
    recordings = {}
    for i, kind in enumerate(SWEEPS, 1):
        for attempt in range(1, max_tries + 1):
            ask(f"[{side_ko} 손가락별 {i}/{len(SWEEPS)}] {SWEEPS[kind]}. '시작' 이 나오면 {move_s:.0f}초 동안 "
                "천천히 3번. 손을 편 채 Enter 를 누르세요 ")
            say(f"  편 손 기록 중: 그대로 멈춰 있으세요 ({rest_s:.0f}초)")
            if fake:
                node.pose_pub.publish(String(data="flat"))
                spin_for(node, 0.3)
            try:
                rest = record_pose(node, rest_s)
                restless = unsteady(rest)
                if restless:
                    names = ", ".join(f"{n} {v:.2f}" for n, v in restless.items())
                    say(f"  편 손이 움직였습니다 ({names}): 다시 합니다 ({attempt}/{max_tries})")
                    continue
                say(f"  시작: {SOURCE_KO[kind]} 천천히 3번 굽혔다 펴기")
                moving = record_sweep(node, move_s, FAKE_SWEEP_POSES[kind] if fake else None)
                moved = travel(kind, (rest, moving))
            except (CalibrationError, SweepError) as exc:
                say(f"  기록 실패 ({exc}): 다시 합니다 ({attempt}/{max_tries})")
                continue
            if moved < MIN_MOVE[kind]:
                say(f"  {SOURCE_KO[kind]}이 거의 움직이지 않았습니다 ({moved:.2f} < {MIN_MOVE[kind]}): "
                    f"다시 합니다 ({attempt}/{max_tries})")
                continue
            say(f"  기록됨 ({len(moving)} 샘플, 움직임 {moved:.2f})")
            recordings[kind] = (rest, moving)
            break
        else:
            raise CalibrationError(f"{SOURCE_KO[kind]} 를 {max_tries}번 해도 기록하지 못했습니다")
    try:
        return fit_sweeps(recordings, recorded=time.strftime("%Y-%m-%dT%H:%M:%S"))
    except SweepError as exc:
        raise CalibrationError(str(exc)) from exc


def alignment_line(calibration: HandCalibration) -> str:
    a = calibration.alignment
    if a is None:
        return "  리타겟팅 정렬 없음: 장갑 손 모델(hand_position)이 오지 않는다 → 예시 자세 방식으로 따라간다"
    import numpy as np

    mirror = "거울 좌표" if np.linalg.det(a.rotation) < 0 else "일반 좌표"
    return f"  리타겟팅 정렬: 크기 비 {a.scale * 1000:.2f} (로봇/사람), 장갑 {mirror}"


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
    ap.add_argument("--fingers", action="store_true",
                    help="one finger at a time: per-finger ranges and crosstalk (needs a saved calibration)")
    ap.add_argument("--sweep-seconds", type=float, default=6.0, help="--fingers: moving part of each sweep")
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

        if args.fingers:
            saved = HandCalibration.load(out, side=side)
            sweeps = finger_sweeps(node, side, ask, say, fake=args.fake, rest_s=1.0, move_s=args.sweep_seconds)
            saved.with_sweeps(sweeps).save(out)
            print(f"손가락별 맞춤 저장됨: {out}")
            for line in report(sweeps):
                print(line)
            return
        if args.rezero:
            saved = HandCalibration.load(out, side=side)
            ask(f"[편 손 맞춤] {config.examples[OPEN_POSE].prompt}. 자세를 잡고 멈춘 뒤 Enter 를 누르세요 ")
            say("  기록 중: 그대로 멈춰 있으세요")
            calibration = rezero(saved, record(OPEN_POSE), OPEN_POSE, thumb_of(config.examples[OPEN_POSE].target))
            calibration.save(out)
            shift = ", ".join(f"{n} {v:+.2f}" for m in calibration.models.values()
                              for n, v in zip(m.inputs, m.offset) if abs(v) > 0.02)
            print(f"편 손 맞춤 저장됨: {out} ({shift or '거의 그대로'})")
            print(alignment_line(calibration))
            return
        calibration = run_session(side=side, user=args.user, groups=config.groups,
                                  examples=examples, ask=ask, record=record, say=say)
        calibration.save(out)
        print(f"보정 저장됨: {out}")
        print("  손가락별 맞춤은 새 보정에 없다: 필요하면 [손가락별 맞춤]을 다시 한다")
        print(f"  예시 자세 {len(examples)}개, 예시에서의 최대 오차 {fit_error(calibration, examples):.3f} rad")
        print(alignment_line(calibration))
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
