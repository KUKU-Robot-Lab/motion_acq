"""Controller direction check: how far each controller moved, in robot axes, without moving anything.

The arms follow the controllers' world-frame displacement (teleop: the
controller's own frame cancels out), mapped into the robot frame: x forward,
y left, z up, re-centred on the HMD heading like a teleop start. Before the
arms are driven from the headset view (WebXR), the operator moves each
controller forward / left / up and the window must show the same axis.
Reads the pose stream on TCP 65432 as one more client (quest-view or the mock
serve any number); never connects to the HandUMI app's single stream.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

SIDES = ("left", "right")
AXES_KO = (("앞", "뒤"), ("왼", "오른"), ("위", "아래"))


def describe(delta_m: np.ndarray) -> dict:
    """{"x_cm", "y_cm", "z_cm", "main": "앞 12 cm"} for one displacement in robot axes."""
    d = np.asarray(delta_m, dtype=float) * 100.0
    axis = int(np.argmax(np.abs(d)))
    words = AXES_KO[axis][0 if d[axis] >= 0 else 1]
    main = f"{words} {abs(d[axis]):.0f} cm" if abs(d[axis]) >= 2.0 else "거의 그대로"
    return {"x_cm": round(d[0], 1), "y_cm": round(d[1], 1), "z_cm": round(d[2], 1), "main": main}


class DirectionCheck:
    def __init__(self, rig_config: Path) -> None:
        self.rig_config = rig_config
        self.provider = None
        self.base: dict[str, np.ndarray] | None = None
        self.started_at = 0.0
        self.lock = threading.Lock()
        self.error = ""

    @property
    def active(self) -> bool:
        return self.provider is not None

    def start(self) -> None:
        from motion_acq.calibration.control_tcp import ControllerTcpCalibration
        from motion_acq.robots.utils import IDENTITY_POSE7
        from motion_acq.tracking.meta_quest import MetaQuestConfig, MetaQuestTrackingProvider

        with self.lock:
            if self.provider is not None:
                return
            identity = IDENTITY_POSE7.astype(np.float32)
            calibration = ControllerTcpCalibration(left=identity.copy(), right=identity.copy(), source=None)
            provider = MetaQuestTrackingProvider(config=MetaQuestConfig.from_yaml(self.rig_config),
                                                 calibration=calibration, reset_workspace_on_x=False)
            provider.start()
            self.provider, self.base, self.started_at, self.error = provider, None, time.time(), ""

    def stop(self) -> None:
        with self.lock:
            provider, self.provider, self.base = self.provider, None, None
        if provider is not None:
            provider.stop()

    def rebase(self) -> None:
        """Re-centre on the HMD now and take the current controller positions as zero."""
        with self.lock:
            if self.provider is not None:
                self.provider.reset_workspace()
                self.base = None

    def snapshot(self) -> dict:
        with self.lock:
            provider = self.provider
        if provider is None:
            return {"active": False}
        try:
            sample = provider.latest()
        except Exception as exc:  # noqa: BLE001 - shown in the window
            return {"active": True, "error": str(exc)}
        tracked = {side: bool(getattr(sample, f"{side}_tracked")) for side in SIDES}
        poses = {side: np.asarray(getattr(sample, f"{side}_controller_pose"))[:3] for side in SIDES}
        with self.lock:
            if self.base is None and all(tracked.values()) and getattr(sample, "hmd_tracked", True):
                self.base = {side: poses[side].copy() for side in SIDES}
            base = self.base
        out: dict = {"active": True, "tracked": tracked, "hmd_tracked": bool(getattr(sample, "hmd_tracked", False))}
        if base is None:
            out["waiting"] = "두 컨트롤러와 HMD 가 보이면 그 위치를 0 으로 잡는다"
            return out
        out["sides"] = {side: describe(poses[side] - base[side]) for side in SIDES if tracked[side]}
        return out
