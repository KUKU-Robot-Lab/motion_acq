#!/usr/bin/env python3
"""Run live HandUMI teleop on a registered real-robot backend.

The controller TCP pose is anchored at the robot home TCP, then relative
controller motion drives the IK target. The IK solution ``q`` is the source of
truth; the selected backend converts it to hardware commands and streams them
over CAN.

Safety behavior:

* controller->TCP calibration is required;
* the robot homes before teleop starts;
* opening a HandUMI gripper activates and anchors that arm;
* a HandUMI gripper held fully closed for two seconds parks that arm at home;
* reopening the corresponding HandUMI gripper re-anchors that arm for use.

Examples:

    handumi teleop-real --device pico --robot piper
    handumi teleop-real --device pico --robot piper --space-start
"""

import argparse
import logging
import os
import sys
import threading
import time

import numpy as np

# Also cover direct execution instead of the ``handumi`` command router.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from motion_acq.feetech import FeetechGripperSampler
from motion_acq.feetech.setup import list_feetech_serial_ports
from motion_acq.real.registry import make_real_backend
from motion_acq.robots.registry import load_embodiment, resolve_home_q
from motion_acq.teleop.common import (
    BestEffortPeriodicWorker,
    JointMotionDiagnostics,
    KeyboardSpaceListener,
    TeleopLoopTimer,
    return_home_on_exit,
)
from motion_acq.teleop.common import (
    enabled_sides as _enabled_sides,
)
from motion_acq.teleop.common import (
    enabled_tracking_ok as _enabled_tracking_ok,
)
from motion_acq.teleop.common import (
    latest_widths as _latest_widths,
)
from motion_acq.teleop.common import (
    tracking_world_map as _tracking_world_map,
)
from motion_acq.teleop.core import TeleopController
from motion_acq.teleop.dls import (
    make_real_teleop_dls_solver,
    resolve_real_teleop_timing,
)
from motion_acq.teleop.hardware import (
    load_required_controller_tcp_calibration as _load_required_calibration,
)
from motion_acq.teleop.hardware import (
    validate_feetech_ports_exist,
)
from motion_acq.teleop.hardware import (
    validate_feetech_ready as _validate_feetech_ready,
)
from motion_acq.teleop.motion import (
    TeleopMotionConfig,
    validate_teleop_motion_args,
)
from motion_acq.teleop.physical import (
    add_physical_teleop_arguments,
    validate_physical_teleop_args,
)
from motion_acq.teleop.session import TeleopSession
from motion_acq.teleop.standby import GripperHomeStandby
from motion_acq.teleop.tracking import LatestTrackingSampler, TrackingRecoveryPolicy
from motion_acq.utils.speech import log_say
from motion_acq.visualize import LiveCameraViews

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
real_log = logging.getLogger("handumi.teleop_real")

def _parse_real_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    show_advanced = "--help-advanced" in raw_argv
    raw_argv = [value for value in raw_argv if value != "--help-advanced"]
    parser = argparse.ArgumentParser(
        description="Teleoperate a supported physical robot with HandUMI."
    )
    parser.add_argument(
        "--help-advanced", action="store_true", help="Show expert hardware options."
    )
    add_physical_teleop_arguments(parser)
    parser.add_argument(
        "--duration-s", type=float, default=0.0, help="0 means run until Ctrl+C."
    )
    parser.add_argument(
        "--joint-debug",
        action="store_true",
        help="Log per-arm IK, filtered, and streamed joint positions once per second.",
    )
    parser.add_argument(
        "--ik-solver",
        choices=("dls", "lm"),
        default="dls",
        help=(
            "IK follower: incremental singularity-aware DLS (default), or the "
            "legacy global LM solver."
        ),
    )
    # teleop-real can follow the headset faster with the inexpensive DLS step.
    # ``None`` lets us retain the historical 30 Hz default for explicit LM.
    parser.set_defaults(fps=None, trajectory_delay_ms=None)
    if not show_advanced:
        normal = {
            "help",
            "help_advanced",
            "device",
            "robot",
            "side",
            "space_start",
            "no_sounds",
            "cameras",
            "skip_cameras",
            "no_rerun",
            "joint_debug",
            "ik_solver",
            "fps",
        }
        for action in parser._actions:
            if action.dest not in normal:
                action.help = argparse.SUPPRESS
    else:
        parser.print_help()
        raise SystemExit(0)
    args = parser.parse_args(raw_argv)
    args.fps, args.trajectory_delay_ms = resolve_real_teleop_timing(
        args.ik_solver,
        input_rate_hz=args.fps,
        trajectory_delay_ms=args.trajectory_delay_ms,
    )
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _parse_real_args(argv)


def _validate_real_args(args: argparse.Namespace) -> None:
    validate_teleop_motion_args(args)
    validate_physical_teleop_args(args)
    if args.duration_s < 0.0:
        raise SystemExit("--duration-s must be >= 0.")


def _validate_feetech_ports_exist(feetech_config, *, robot: str = "openarmv1") -> None:
    return validate_feetech_ports_exist(
        feetech_config,
        robot=robot,
        list_ports=list_feetech_serial_ports,
    )


class _LatestJointFeedback:
    """Cache robot feedback off the latency-sensitive teleop loop."""

    def __init__(self, read, base_q: np.ndarray) -> None:
        self._read = read
        self._base_q = np.asarray(base_q, dtype=np.float32).copy()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None

    def update(self) -> None:
        value = np.asarray(self._read(base_q=self._base_q), dtype=np.float32)
        with self._lock:
            self._latest = value.copy()

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()


def _motion_joint_indices(runtime, enabled_sides: tuple[str, ...]) -> tuple[int, ...]:
    finger_indices = {
        finger.index
        for fingers in (runtime.finger_joints or {}).values()
        for finger in fingers
    }
    return tuple(
        dict.fromkeys(
            index
            for side in enabled_sides
            for index in runtime.arm_joint_indices(side)
            if index not in finger_indices
        )
    )


def _log_joint_debug(
    runtime,
    enabled_sides: tuple[str, ...],
    raw_q: np.ndarray,
    filtered_q: np.ndarray,
    streamed_q: np.ndarray | None,
    feedback_q: np.ndarray | None,
) -> None:
    finger_indices = {
        finger.index
        for fingers in (runtime.finger_joints or {}).values()
        for finger in fingers
    }
    streamed = filtered_q if streamed_q is None else streamed_q
    for side in enabled_sides:
        indices = [
            index
            for index in runtime.arm_joint_indices(side)
            if index not in finger_indices
        ]
        names = [runtime.joint_names[index] for index in indices]
        raw_deg = np.round(np.rad2deg(raw_q[indices]), 1).tolist()
        filtered_deg = np.round(np.rad2deg(filtered_q[indices]), 1).tolist()
        streamed_deg = np.round(np.rad2deg(streamed[indices]), 1).tolist()
        feedback_deg = (
            None
            if feedback_q is None
            else np.round(np.rad2deg(feedback_q[indices]), 1).tolist()
        )
        real_log.info(
            "Joint debug %s %s (deg): IK=%s filtered=%s streamed=%s feedback=%s",
            side,
            names,
            raw_deg,
            filtered_deg,
            streamed_deg,
            feedback_deg,
        )


def _run_real() -> None:
    args = _parse_real_args()
    _validate_real_args(args)
    from motion_acq.scripts.record import build_tracker, connect_feetech

    real_log.info("Loading %s kinematics.", args.robot)
    runtime = load_embodiment(args.robot)
    import jax

    real_log.info("IK compute platform: %s.", jax.default_backend())
    try:
        home_pose_name, home_q = resolve_home_q(
            runtime, rig_config=args.rig_config, explicit_name=args.home_pose
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    controller = TeleopController(
        runtime,
        home_q=home_q,
        enabled_sides=_enabled_sides(args.side),
        source_world_to_robot_world=_tracking_world_map(args.device),
        translation_scale=args.translation_scale,
    )
    if args.ik_solver == "dls":
        controller.solver = make_real_teleop_dls_solver(
            runtime,
            controller.solver,
            home_q,
            input_rate_hz=float(args.fps),
        )
        for side in _enabled_sides(args.side):
            limits = controller.solver.side_joint_speed_limits_rad_s[side]
            real_log.info(
                "DLS %s joint-speed limits resolved automatically: %.1f..%.1f deg/s.",
                side,
                float(np.rad2deg(np.min(limits))),
                float(np.rad2deg(np.max(limits))),
            )
    q = home_q.copy()
    real_log.info("Selected home pose: %s", home_pose_name)
    real_log.info(
        "Warming %s IK solver before touching hardware.", args.ik_solver.upper()
    )
    controller.warmup()
    _validate_feetech_ready(args)

    calibration = _load_required_calibration(args)
    tracker = build_tracker(args, calibration, reset_workspace_on_x=False)
    gripper_pair = None
    grippers = None
    enabled_sides = _enabled_sides(args.side)
    real_env = make_real_backend(
        args.robot,
        runtime=runtime,
        rig_config=args.rig_config,
        active_sides=enabled_sides,
        fake=args.fake_robot,
    )
    space_listener = KeyboardSpaceListener(enabled=args.space_start)
    tracker_started = False
    tracking_sampler: LatestTrackingSampler | None = None
    camera_views: LiveCameraViews | None = None
    camera_worker: BestEffortPeriodicWorker | None = None
    feedback_sampler: _LatestJointFeedback | None = None
    feedback_worker: BestEffortPeriodicWorker | None = None

    # Match teleop-record: with Feetech enabled, every arm starts parked and
    # can only be anchored by opening its corresponding HandUMI gripper.
    # --space-start remains the fallback for --skip-feetech.
    home_standby = GripperHomeStandby(
        hold_s=args.gripper_park_hold_s,
        initial_standby=not args.skip_feetech,
    )
    play_sounds = not args.no_sounds
    motion_config = TeleopMotionConfig.from_args(args)
    loop_timer = TeleopLoopTimer(motion_config.input_rate_hz)
    motion_joint_indices = _motion_joint_indices(runtime, enabled_sides)
    joint_filter = motion_config.make_joint_filter(
        filtered_indices=motion_joint_indices
    )
    teleop_session = TeleopSession(controller, joint_filter)
    joint_diagnostics = JointMotionDiagnostics(
        runtime.joint_names, motion_joint_indices
    )
    command_stream = motion_config.make_command_stream(real_env.write)
    episode_start: float | None = None
    last_processed_tracking_time_ns: int | None = None
    tracking_recovery = TrackingRecoveryPolicy()
    timing_next_log_s = time.perf_counter() + 5.0
    timing_window_start_s = time.perf_counter()
    timing_ik_total_s = 0.0
    timing_ik_max_s = 0.0
    timing_ik_samples = 0
    timing_tracking_age_max_s = 0.0
    timing_discarded_ik = 0
    timing_playback_counts = (0, 0, 0)
    joint_debug_next_log_s = time.perf_counter() + 1.0
    feedback_error_reported = False
    arms_homed = False  # the start pose was reached; the end pose is home again

    try:
        real_log.info("Starting tracking before moving real arms.")
        tracker.start()
        tracker_started = True
        tracking_sampler = LatestTrackingSampler(
            tracker.latest,
            sample_rate_hz=motion_config.input_rate_hz,
        )
        tracking_sampler.start()
        camera_views = LiveCameraViews.from_args(
            args, application_id="handumi_teleop_real"
        )
        if camera_views is not None:
            camera_worker = BestEffortPeriodicWorker(
                camera_views.update,
                rate_hz=args.cam_fps,
                thread_name="handumi-live-camera-preview",
            )
            camera_worker.start()
        gripper_pair = connect_feetech(args)
        if gripper_pair is not None:
            # Keep serial retries off the real-time control loop.  In
            # particular, a transient Feetech timeout must not abort teleop:
            # the sampler retains the last valid widths while it retries in
            # the background, just like the recording paths do.
            grippers = FeetechGripperSampler(
                gripper_pair,
                sample_hz=max(100.0, motion_config.input_rate_hz),
            )
            grippers.start()

        real_env.setup(repair=not args.skip_can_repair)
        real_env.connect()
        real_env.home(home_q)
        arms_homed = True
        if args.joint_debug:
            feedback_sampler = _LatestJointFeedback(real_env.read, home_q)
            feedback_worker = BestEffortPeriodicWorker(
                feedback_sampler.update,
                rate_hz=10.0,
                thread_name="handumi-joint-feedback-debug",
            )
            feedback_worker.start()
        joint_filter.reset(home_q)
        startup_widths = _latest_widths(grippers)
        command_stream.submit(
            home_q,
            {
                "left": float(startup_widths.left_normalized),
                "right": float(startup_widths.right_normalized),
            },
            time_s=time.perf_counter(),
            active=True,
            new_epoch=True,
        )
        real_log.info(
            "Continuous gripper synchronization started; arms remain at home "
            "until their HandUMI gripper opens."
        )
        real_log.info(
            "Tracking/IK target rate: %.1f Hz (%s); joint trajectory playback: "
            "%.1f Hz, %.0f ms delay, "
            "%.0f ms max bridge, %.0f ms EMA; adaptive IK filter "
            "cutoff=%.1f Hz + %.1f*|dq/dt|.",
            args.fps,
            args.ik_solver.upper(),
            args.command_rate_hz,
            args.trajectory_delay_ms,
            args.max_extrapolation_ms,
            args.motion_smoothing_time_constant_s * 1000.0,
            args.joint_filter_min_cutoff_hz,
            args.joint_filter_velocity_coefficient,
        )

        space_listener.start()
        if args.space_start:
            real_log.info(
                "Real %s is at home. Open a HandUMI gripper to start that arm; "
                "Space starts idle arms when Feetech is disabled.",
                args.robot,
            )
        else:
            real_log.info(
                "Real %s is at home. Open a HandUMI gripper to start that arm.",
                args.robot,
            )

        while True:
            loop_start, _ = loop_timer.tick()
            if episode_start is not None and args.duration_s > 0.0:
                if loop_start - episode_start >= args.duration_s:
                    break

            tracking_snapshot = tracking_sampler.latest()
            if tracking_snapshot is None:
                widths = _latest_widths(grippers)
                command_stream.update_openings(
                    {
                        "left": float(widths.left_normalized),
                        "right": float(widths.right_normalized),
                    }
                )
                if args.space_start:
                    space_listener.consume_space()
                loop_timer.sleep(loop_start)
                continue
            sample = tracking_snapshot.sample
            tracking_age_s = tracking_snapshot.age_s(loop_start)
            timing_tracking_age_max_s = max(
                timing_tracking_age_max_s, tracking_age_s
            )
            tracking_stale = tracking_age_s > args.tracking_stale_ms / 1000.0
            side_tracked = {"left": sample.left_tracked, "right": sample.right_tracked}
            tracking_ok = (
                not tracking_stale
                and _enabled_tracking_ok(side_tracked, enabled_sides)
            )

            # Tracker startup and a fresh SDK reconnect can briefly expose an
            # empty sample.  Before an operator anchors an arm, there is no
            # robot motion to cancel, so do not hold the robot or restart the
            # PICO service for that transient state.
            if not controller.active and not tracking_ok:
                tracking_recovery.reset()
                widths = _latest_widths(grippers)
                command_stream.update_openings(
                    {
                        "left": float(widths.left_normalized),
                        "right": float(widths.right_normalized),
                    }
                )
                if args.space_start:
                    space_listener.consume_space()
                loop_timer.sleep(loop_start)
                continue

            if not tracking_ok:
                widths = _latest_widths(grippers)
                openings = {
                    "left": float(widths.left_normalized),
                    "right": float(widths.right_normalized),
                }
                if tracking_recovery.note_missing(
                    loop_start,
                    observed_since=(
                        tracking_snapshot.fresh_at_s if tracking_stale else None
                    ),
                ):
                    command_stream.stop()
                    held = real_env.hold(q)
                    controller.tracking_lost(held)
                    home_standby.enter_standby(enabled_sides)
                    joint_filter.reset(held)
                    q = held
                    command_stream.submit(
                        held,
                        openings,
                        time_s=loop_start,
                        active=True,
                        new_epoch=True,
                    )
                    real_log.warning(
                        "Tracking lost%s; pending motion cancelled at the current "
                        "robot command. Re-anchor after recovery.",
                        (
                            f" (sample age {tracking_age_s * 1000.0:.0f} ms)"
                            if tracking_stale
                            else ""
                        ),
                    )
                    log_say("tracking lost", play_sounds=play_sounds)
                recover = getattr(tracker, "recover", None)
                if callable(recover) and tracking_recovery.should_recover(loop_start):
                    acquired, recovered = tracking_sampler.try_source_call(recover)
                    if acquired and recovered:
                        real_log.info(
                            "Tracking recovered; open the HandUMI gripper to re-anchor."
                        )
                        log_say("tracking recovered", play_sounds=play_sounds)
                command_stream.update_openings(openings)
                loop_timer.sleep(loop_start)
                continue
            if tracking_recovery.lost:
                real_log.info(
                    "Tracking stream is valid again; waiting for a fresh anchor."
                )
            tracking_recovery.reset()

            widths = _latest_widths(grippers)
            inputs = teleop_session.inputs(sample, widths)
            start_sides: tuple[str, ...] = ()
            if grippers is not None:
                park_sides, wake_sides = home_standby.update(
                    inputs.openings,
                    loop_start,
                    enabled_sides,
                )
            else:
                park_sides, wake_sides = (), ()
            if park_sides:
                parked = controller.park(park_sides)
                if parked:
                    park_indices = tuple(
                        index
                        for index in motion_joint_indices
                        if any(
                            index in controller.side_indices[side] for side in parked
                        )
                    )
                    command_stream.limit_joint_rates(
                        park_indices,
                        np.deg2rad(args.park_max_joint_speed_deg_s),
                    )
                    real_log.info(
                        "HandUMI %s gripper remained fully closed for %.1fs; "
                        "arm returning home "
                        "and entering standby.",
                        "/".join(parked),
                        args.gripper_park_hold_s,
                    )
                    for side in parked:
                        log_say(f"{side} arm standby", play_sounds=play_sounds)
            if wake_sides:
                wake_indices = tuple(
                    index
                    for index in motion_joint_indices
                    if any(
                        index in controller.side_indices[side] for side in wake_sides
                    )
                )
                command_stream.clear_joint_rate_limits(wake_indices)
                start_sides = wake_sides
                real_log.info(
                    "HandUMI %s gripper reopened; waking and re-anchoring arm.",
                    "/".join(wake_sides),
                )
            if args.space_start and space_listener.consume_space():
                space_sides = tuple(
                    side
                    for side in controller.idle_sides()
                    if side not in start_sides and not home_standby.is_standby(side)
                )
                start_sides += space_sides
                if space_sides:
                    real_log.info("Space pressed; starting %s.", "/".join(space_sides))

            fresh_tracking = (
                last_processed_tracking_time_ns is None
                or tracking_snapshot.source_time_ns
                != last_processed_tracking_time_ns
            )
            if not fresh_tracking and not start_sides and not park_sides:
                command_stream.update_openings(inputs.openings)
                real_env.check_health()
                loop_timer.sleep(loop_start)
                continue

            controller_q_before_ik = controller.q.copy()
            filter_before_ik = teleop_session.snapshot_filter()
            set_solver_timestep = getattr(controller.solver, "set_timestep", None)
            if callable(set_solver_timestep):
                source_dt_s = (
                    1.0 / motion_config.input_rate_hz
                    if last_processed_tracking_time_ns is None
                    else (
                        tracking_snapshot.source_time_ns
                        - last_processed_tracking_time_ns
                    )
                    / 1e9
                )
                set_solver_timestep(source_dt_s)
            ik_start_s = time.perf_counter()
            teleop_frame = teleop_session.advance(
                inputs,
                now_s=loop_start,
                start_sides=start_sides,
                exact_sides=park_sides,
            )
            ik_elapsed_s = time.perf_counter() - ik_start_s
            timing_ik_total_s += ik_elapsed_s
            timing_ik_max_s = max(timing_ik_max_s, ik_elapsed_s)
            timing_ik_samples += 1
            newest_after_ik = tracking_sampler.latest()
            if (
                newest_after_ik is not None
                and newest_after_ik.source_time_ns
                != tracking_snapshot.source_time_ns
                and not start_sides
                and not park_sides
            ):
                # Never publish an IK result for a frame superseded while the
                # solve was running. It would be a visibly delayed correction.
                controller.q = controller_q_before_ik
                teleop_session.restore_filter(filter_before_ik)
                timing_discarded_ik += 1
                real_env.check_health()
                continue
            anchored_sides = teleop_frame.anchored_sides
            anchored_this_frame = bool(anchored_sides)
            for side in anchored_sides:
                real_log.info("%s arm anchored; real robot follows from home.", side)
                log_say(f"{side} anchored", play_sounds=play_sounds)

            if episode_start is None and anchored_this_frame:
                episode_start = loop_start
                timing_next_log_s = time.perf_counter() + 5.0
                timing_window_start_s = time.perf_counter()
                timing_ik_total_s = 0.0
                timing_ik_max_s = 0.0
                timing_ik_samples = 0
                timing_tracking_age_max_s = 0.0
                timing_discarded_ik = 0
                timing_playback_counts = (0, 0, 0)
                joint_diagnostics.reset()
                real_log.info("Teleop timer started.")

            q = teleop_frame.q
            joint_diagnostics.observe(teleop_frame.step.q, q)
            command_stream.submit(
                q,
                inputs.openings,
                time_s=tracking_snapshot.fresh_at_s,
                # A park transition must publish its home target even when it
                # just deactivated the final active arm.
                active=controller.active or bool(park_sides),
                new_epoch=anchored_this_frame,
            )
            last_processed_tracking_time_ns = tracking_snapshot.source_time_ns
            real_env.check_health()

            timing_now_s = time.perf_counter()
            if args.joint_debug and timing_now_s >= joint_debug_next_log_s:
                if (
                    feedback_worker is not None
                    and feedback_worker.error is not None
                    and not feedback_error_reported
                ):
                    real_log.warning(
                        "Joint feedback debug sampler stopped: %s",
                        feedback_worker.error,
                    )
                    feedback_error_reported = True
                latest_command = command_stream.latest()
                _log_joint_debug(
                    runtime,
                    enabled_sides,
                    teleop_frame.step.q,
                    q,
                    None if latest_command is None else latest_command[0],
                    None if feedback_sampler is None else feedback_sampler.latest(),
                )
                joint_debug_next_log_s = timing_now_s + 1.0
            if command_stream.running and timing_now_s >= timing_next_log_s:
                output_stats = command_stream.stats()
                playback_counts = (
                    output_stats.interpolated_commands,
                    output_stats.extrapolated_commands,
                    output_stats.held_commands,
                )
                playback_window = tuple(
                    max(0, current - previous)
                    for current, previous in zip(
                        playback_counts, timing_playback_counts, strict=True
                    )
                )
                real_log.info(
                    "Control timing: output=%.1f Hz, missed=%d, "
                    "tracking_age_max=%.0f ms, IK_rate=%.1f Hz, "
                    "IK_avg/max=%.1f/%.1f ms, "
                    "backend_write_max=%.1f ms, output_lateness_max=%.1f ms, "
                    "target_age=%.0f ms, playback(i/x/h)=%d/%d/%d, "
                    "stale_IK_discarded=%d; %s.",
                    output_stats.effective_rate_hz,
                    output_stats.missed_deadlines,
                    timing_tracking_age_max_s * 1000.0,
                    timing_ik_samples
                    / max(timing_now_s - timing_window_start_s, 1e-6),
                    (
                        timing_ik_total_s / timing_ik_samples * 1000.0
                        if timing_ik_samples
                        else 0.0
                    ),
                    timing_ik_max_s * 1000.0,
                    output_stats.max_write_ms,
                    output_stats.max_lateness_ms,
                    output_stats.latest_target_age_ms,
                    *playback_window,
                    timing_discarded_ik,
                    joint_diagnostics.summary(),
                )
                timing_next_log_s = timing_now_s + 5.0
                timing_window_start_s = timing_now_s
                timing_ik_total_s = 0.0
                timing_ik_max_s = 0.0
                timing_ik_samples = 0
                timing_tracking_age_max_s = 0.0
                timing_discarded_ik = 0
                timing_playback_counts = playback_counts
                joint_diagnostics.reset()

            loop_timer.sleep(loop_start)
    except KeyboardInterrupt:
        real_log.info("Stopping.")
    finally:
        space_listener.close()
        try:
            command_stream.stop()
        finally:
            try:
                if feedback_worker is not None:
                    feedback_worker.close()
            finally:
                try:
                    if arms_homed:
                        return_home_on_exit(real_env, home_q, real_log)
                    real_env.disconnect()
                finally:
                    if grippers is not None:
                        grippers.stop()
                    if gripper_pair is not None:
                        gripper_pair.close()
                    if tracker_started:
                        if tracking_sampler is not None:
                            tracking_sampler.stop()
                        tracker.stop()
                    if camera_worker is not None:
                        camera_worker.close()
                    if camera_views is not None:
                        camera_views.close()


def main() -> None:
    _run_real()


if __name__ == "__main__":
    main()
