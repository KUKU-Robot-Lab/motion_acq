from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

from motion_acq.scripts.setup import calibrate_tcp_offset


def _pose_csv(path, *, side: str, offset: np.ndarray) -> None:
    rotations = Rotation.from_euler(
        "xyz",
        np.column_stack(
            (
                np.linspace(-70, 65, 40),
                np.linspace(55, -60, 40),
                np.linspace(-45, 75, 40),
            )
        ),
        degrees=True,
    )
    matrices = rotations.as_matrix()
    pivot = np.array([0.25, -0.1, 0.4])
    positions = pivot - np.einsum("nij,j->ni", matrices, offset)
    values = np.column_stack((positions, rotations.as_quat()))
    frame = pd.DataFrame(values, columns=["x", "y", "z", "qx", "qy", "qz", "qw"])
    frame["side"] = side
    frame.to_csv(path, index=False)


def test_pivot_defaults_to_vr_only_live_capture() -> None:
    args = calibrate_tcp_offset.build_parser().parse_args(
        ["pivot", "--side", "left", "--time-s", "12"]
    )

    assert args.dataset is None
    assert args.csv is None
    assert args.time_s == 12
    assert calibrate_tcp_offset._capture_output_dir(args).as_posix() == (
        "outputs/tcp_pivot_left"
    )


def test_pivot_fit_metrics_for_both_sides_survive_candidate_update(tmp_path) -> None:
    output = tmp_path / "candidate.yaml"
    left_csv = tmp_path / "left.csv"
    right_csv = tmp_path / "right.csv"
    _pose_csv(left_csv, side="left", offset=np.array([0.04, 0.02, 0.13]))
    _pose_csv(right_csv, side="right", offset=np.array([0.04, -0.02, 0.13]))

    for side, csv_path in (("left", left_csv), ("right", right_csv)):
        args = calibrate_tcp_offset.build_parser().parse_args(
            [
                "pivot",
                "--side",
                side,
                "--csv",
                str(csv_path),
                "--output",
                str(output),
                "--device",
                "meta",
            ]
        )
        args.func(args)

    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    fits = data["calibration"]["pivot_fits"]
    assert fits["left"]["accepted"] is True
    assert fits["right"]["accepted"] is True


def test_capture_rows_stores_only_selected_controller(monkeypatch) -> None:
    ticks = iter([0.0, 0.0, 0.0, 0.02, 0.04, 0.06])
    monkeypatch.setattr(calibrate_tcp_offset.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(calibrate_tcp_offset.time, "sleep", lambda _seconds: None)
    pose = np.array([1, 2, 3, 0, 0, 0, 1], dtype=np.float32)
    sample = SimpleNamespace(
        streaming=True,
        left_device_tracked=True,
        left_device_controller_pose=pose,
        device_time_ns=10,
        pc_monotonic_ns=20,
        sequence=30,
    )
    tracker = SimpleNamespace(latest=lambda: sample)

    rows = calibrate_tcp_offset._capture_rows(
        tracker, side="left", duration_s=0.05, rate_hz=1000
    )

    assert len(rows) == 2
    assert rows[0]["side"] == "left"
    assert [rows[0][name] for name in ("x", "y", "z")] == [1.0, 2.0, 3.0]
    assert "right" not in " ".join(rows[0])


def test_symmetry_report_recommends_weaker_fit_first(capsys) -> None:
    calibration = SimpleNamespace(
        left=np.array([0.11383, -0.17316, -0.06112, 0, 0, 0, 1]),
        right=np.array([-0.10973, -0.18705, -0.04337, 0, 0, 0, 1]),
    )
    fits = {
        "left": {
            "tracking_device": "pico",
            "rms_error_m": 0.0024,
            "max_error_m": 0.0067,
            "condition": 16.0,
        },
        "right": {
            "tracking_device": "pico",
            "rms_error_m": 0.0016,
            "max_error_m": 0.0032,
            "condition": 15.4,
        },
    }

    assert calibrate_tcp_offset._print_symmetry_report(calibration, fits) is False
    output = capsys.readouterr().out
    assert "y: FAIL mismatch=13.89mm" in output
    assert "z: FAIL mismatch=17.75mm" in output
    assert "Recommended first recapture: left" in output


def test_promote_overrides_only_positions_and_preserves_quaternions(
    tmp_path, monkeypatch
) -> None:
    candidate = tmp_path / "candidate.yaml"
    left_csv = tmp_path / "left.csv"
    right_csv = tmp_path / "right.csv"
    _pose_csv(left_csv, side="left", offset=np.array([0.04, 0.02, 0.13]))
    _pose_csv(right_csv, side="right", offset=np.array([0.04, -0.02, 0.13]))
    for side, csv_path in (("left", left_csv), ("right", right_csv)):
        args = calibrate_tcp_offset.build_parser().parse_args(
            [
                "pivot",
                "--side",
                side,
                "--csv",
                str(csv_path),
                "--output",
                str(candidate),
                "--device",
                "meta",
            ]
        )
        args.func(args)

    calibration_dir = tmp_path / "controller_tcp"
    calibration_dir.mkdir()
    target = calibration_dir / "meta_test.yaml"
    target.write_text(
        """# keep this comment
calibration:
  controller_to_gripper_tcp:
    left:
      position:
      - 9
      - 9
      - 9
      quaternion:
      - 0.1
      - 0.2
      - 0.3
      - 0.4
    right:
      position:
      - 8
      - 8
      - 8
      quaternion:
      - 0.5
      - 0.6
      - 0.7
      - 0.8
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(calibrate_tcp_offset, "DEFAULT_CALIBRATION_DIR", calibration_dir)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    args = calibrate_tcp_offset.build_parser().parse_args(
        [
            "promote",
            "--target",
            "meta_test.yaml",
            "--candidate",
            str(candidate),
        ]
    )

    args.func(args)

    text = target.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    transforms = data["calibration"]["controller_to_gripper_tcp"]
    assert "# keep this comment" in text
    np.testing.assert_allclose(transforms["left"]["position"], [0.04, 0.02, 0.13])
    np.testing.assert_allclose(transforms["right"]["position"], [0.04, -0.02, 0.13])
    assert transforms["left"]["quaternion"] == [0.1, 0.2, 0.3, 0.4]
    assert transforms["right"]["quaternion"] == [0.5, 0.6, 0.7, 0.8]


def test_promote_force_allows_failed_symmetry_and_symmetrizes(
    tmp_path, monkeypatch, capsys
) -> None:
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(
        """calibration:
  controller_to_gripper_tcp:
    left:
      position: [0.113, -0.175, -0.068]
      quaternion: [0, 0, 0, 1]
    right:
      position: [-0.110, -0.187, -0.043]
      quaternion: [0, 0, 0, 1]
  pivot_fits:
    left:
      tracking_device: pico
      rms_error_m: 0.002
      max_error_m: 0.004
      condition: 15
    right:
      tracking_device: pico
      rms_error_m: 0.002
      max_error_m: 0.004
      condition: 15
""",
        encoding="utf-8",
    )
    calibration_dir = tmp_path / "controller_tcp"
    calibration_dir.mkdir()
    target = calibration_dir / "pico_test.yaml"
    target.write_text(
        """calibration:
  controller_to_gripper_tcp:
    left:
      position:
      - 0
      - 0
      - 0
      quaternion: [0.1, 0.2, 0.3, 0.4]
    right:
      position:
      - 0
      - 0
      - 0
      quaternion: [0.5, 0.6, 0.7, 0.8]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(calibrate_tcp_offset, "DEFAULT_CALIBRATION_DIR", calibration_dir)
    args = calibrate_tcp_offset.build_parser().parse_args(
        [
            "promote",
            "--target",
            "pico_test.yaml",
            "--candidate",
            str(candidate),
            "--force",
            "--yes",
        ]
    )

    args.func(args)

    output = capsys.readouterr().out
    assert "WARNING: forcing promotion despite failed bilateral symmetry" in output
    transforms = yaml.safe_load(target.read_text(encoding="utf-8"))["calibration"][
        "controller_to_gripper_tcp"
    ]
    np.testing.assert_allclose(
        transforms["left"]["position"], [0.1115, -0.181, -0.0555]
    )
    np.testing.assert_allclose(
        transforms["right"]["position"], [-0.1115, -0.181, -0.0555]
    )
    assert transforms["left"]["quaternion"] == [0.1, 0.2, 0.3, 0.4]
    assert transforms["right"]["quaternion"] == [0.5, 0.6, 0.7, 0.8]
