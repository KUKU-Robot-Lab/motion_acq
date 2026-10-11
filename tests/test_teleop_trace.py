"""Teleop trace (10.11 user): one JSON line per frame, readable offline."""

import json

import numpy as np

from motion_acq.teleop.trace import TeleopTrace


def test_trace_writes_setup_anchor_and_frames(tmp_path):
    path = tmp_path / "t" / "trace.jsonl"
    tr = TeleopTrace(path, {"left": [0, 1, 2]})
    tr.setup(translation_scale=0.5)
    tr.anchor("left", heading_rad=np.pi / 2, source_pose7=np.arange(7), robot_pose7=np.zeros(7))
    q = np.arange(10, dtype=np.float32)
    tr.frame(sample_time_ns=5, active=("left",), tracked={"left": True},
             source={"left": np.ones(7)}, hmd=None, target={}, ik_q=q, filtered_q=q,
             feedback_q=None, openings={"left": 0.25})
    tr.event("park", sides=["left"])
    tr.close()
    tr.close()  # idempotent
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["setup", "anchor", "frame", "park"]
    assert rows[1]["heading_deg"] == 90.0
    frame = rows[2]
    assert frame["ik"] == {"left": [0.0, 1.0, 2.0]}
    assert frame["fb"] == {"left": None} and frame["tgt"] == {"left": None}
    assert frame["hmd"] is None and frame["open"] == {"left": 0.25}
