"""Joystick tuning window (10.11 user): each robot joint from one glove angle, set by eye.

    ros2 run motion_acq_hand tune --side right --user yoon       # needs a desktop (DISPLAY)

Per robot joint (thumb_1 the thumb swing across the palm, thumb_2 the thumb tip bend, the index,
middle and ring+pinky bend) the window shows, live:
* the robot joint's whole range (limits_rad, what the RH56F1 can move) as a dial, the points of the map
  on it and where the map puts the joint now (degrees and 0..1);
* the glove reading that drives it, with the points' readings marked.
Each point has a robot angle (editable, inside the joint's range) and a glove reading: hold the glove
finger in that pose and press [지금 자세로 기록] (median of the last 0.5 s). The glove source can be
switched (the thumb candidates are listed live with their min/max at the bottom, to see which one
follows the thumb swing and not the tip bend). [저장] writes the map into the calibration file
(joystick, motion_acq.hand.joystick); the hand node picks it up when it is not following the glove
(press [홈] then [따라가기] again).
"""

# ruff: noqa: I001  -- motion_acq_hand.common must be imported first (it puts motion_acq on sys.path)
from __future__ import annotations

import argparse
import collections
import math
import statistics
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from motion_acq_hand.common import REPO_ROOT, check_side, glove_qos, glove_topic, load_gloves

import rclpy
from rclpy.node import Node
from senseglove_msgs.msg import SenseGloveState

from motion_acq.hand.calibration import HandCalibration
from motion_acq.hand.joystick import (
    JOINT_KO, JOINTS, POINTS, SOURCE_CHOICES, JointMap, JoystickError, JoystickMap, JoystickRetargeter,
    default_joint, raw_reading)
from motion_acq.hand.kinematic import glove_points
from motion_acq.hand.nova2 import GloveDataError, angles_from_state, tip_signals
from motion_acq.hand.retarget import DEFAULT_RETARGET, load_hand_retarget_config

CAPTURE_S = 0.5
THUMB_SIGNALS = ("thumb_brake", "thumb_mcp", "thumb_pip", "thumb_tip_x", "thumb_tip_y", "thumb_tip_z")
POINT_COLORS = ("#1f77b4", "#2ca02c", "#d62728")
DIAL_W, DIAL_H, BAR_W, BAR_H = 230, 130, 420, 46


class GloveListener(Node):
    def __init__(self, topic: str, side: str) -> None:
        super().__init__("motion_acq_hand_tune")
        self.side = side
        self.history: collections.deque[tuple[float, dict[str, float]]] = collections.deque()
        self.create_subscription(SenseGloveState, topic, self._on_glove, glove_qos())

    def _on_glove(self, msg: SenseGloveState) -> None:
        try:
            s = angles_from_state(list(msg.joint_names), list(msg.position), self.side)
            s.update(tip_signals([(p.x, p.y, p.z) for p in msg.finger_tip_position]))
            s.update(glove_points([(p.x, p.y, p.z) for p in msg.hand_position],
                                  [(p.x, p.y, p.z) for p in msg.finger_tip_position]))
        except GloveDataError:
            return
        t = time.monotonic()
        self.history.append((t, s))
        while self.history and t - self.history[0][0] > 2.0:
            self.history.popleft()

    def latest(self) -> dict[str, float] | None:
        if not self.history or time.monotonic() - self.history[-1][0] > 0.5:
            return None
        return self.history[-1][1]

    def recent(self, seconds: float) -> list[dict[str, float]]:
        t = time.monotonic()
        return [s for ts, s in self.history if t - ts <= seconds]


def _deg(q: float) -> float:
    return math.degrees(q)


class JointRow:
    """One robot joint: dial, glove bar, source, and its points."""

    def __init__(self, app: TuneApp, parent: tk.Widget, joint: str, row: int) -> None:
        self.app, self.joint = app, joint
        self.lo, self.hi = app.config.limits_rad[joint]
        m = app.start_map(joint)
        self.glove: list[float | None] = list(m.glove) if m else [None] * len(POINTS[joint])
        frame = ttk.LabelFrame(parent, text=JOINT_KO[joint], padding=4)
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        self.dial = tk.Canvas(frame, width=DIAL_W, height=DIAL_H, bg="white", highlightthickness=0)
        self.dial.grid(row=0, column=0, rowspan=3, padx=(0, 8))
        top = ttk.Frame(frame)
        top.grid(row=0, column=1, sticky="w")
        ttk.Label(top, text="장갑 값:").pack(side="left")
        self.source = tk.StringVar(value=m.source if m else SOURCE_CHOICES[joint][0])
        box = ttk.Combobox(top, textvariable=self.source, values=SOURCE_CHOICES[joint], width=22, state="readonly")
        box.pack(side="left", padx=4)
        box.bind("<<ComboboxSelected>>", lambda _e: self._source_changed())
        self.now_label = ttk.Label(top, text="", width=34)
        self.now_label.pack(side="left", padx=6)
        self.bar = tk.Canvas(frame, width=BAR_W, height=BAR_H, bg="white", highlightthickness=0)
        self.bar.grid(row=1, column=1, sticky="w")
        pts = ttk.Frame(frame)
        pts.grid(row=2, column=1, sticky="w")
        self.robot_deg: list[tk.DoubleVar] = []
        self.glove_labels: list[ttk.Label] = []
        targets = [m.robot[i] for i in range(len(POINTS[joint]))] if m else app.default_robot(joint)
        for i, (_, _, ko) in enumerate(POINTS[joint]):
            ttk.Label(pts, text=f"● {ko}", foreground=POINT_COLORS[i]).grid(row=i, column=0, sticky="w")
            ttk.Label(pts, text="로봇").grid(row=i, column=1)
            var = tk.DoubleVar(value=round(_deg(targets[i]), 1))
            spin = ttk.Spinbox(pts, from_=round(_deg(self.lo), 1), to=round(_deg(self.hi), 1), increment=1.0,
                               textvariable=var, width=7)
            spin.grid(row=i, column=2)
            ttk.Label(pts, text=f"° ({_deg(self.lo):.0f}~{_deg(self.hi):.0f})").grid(row=i, column=3, sticky="w")
            self.robot_deg.append(var)
            lab = ttk.Label(pts, text="", width=16)
            lab.grid(row=i, column=4, padx=6)
            self.glove_labels.append(lab)
            ttk.Button(pts, text="지금 자세로 기록", command=lambda k=i: self.capture(k)).grid(row=i, column=5)
        self.error = ttk.Label(frame, text="", foreground="#c00000")
        self.error.grid(row=3, column=0, columnspan=2, sticky="w")

    # -- state ---------------------------------------------------------------------------------------
    def robot(self) -> list[float]:
        out = []
        for var in self.robot_deg:
            try:
                q = math.radians(float(var.get()))
            except (tk.TclError, ValueError):
                q = self.lo
            out.append(min(max(q, self.lo), self.hi))
        return out

    def joint_map(self) -> JointMap | None:
        """The map as set now (None: a point not recorded yet); raises JoystickError when not monotonic."""
        if any(g is None for g in self.glove):
            return None
        m = JointMap(self.source.get(), tuple(float(g) for g in self.glove if g is not None), tuple(self.robot()))
        m.check(self.joint)
        return m

    def reading(self, signals) -> float | None:
        return self.app.reader.reading(self.joint, signals, self.source.get()) if signals else None

    def _source_changed(self) -> None:
        cal = self.app.calibration
        d = default_joint(self.joint, cal.medians, self.app.targets, self.source.get())
        self.glove = list(d.glove) if d else [None] * len(POINTS[self.joint])
        self.app.dirty = True

    def capture(self, k: int) -> None:
        samples = self.app.node.recent(CAPTURE_S)
        values = [v for s in samples if (v := self.reading(s)) is not None]
        if len(values) < 5:
            self.app.status(f"장갑 값이 없습니다 ({len(values)} 샘플): 장갑 드라이버를 확인하세요", error=True)
            return
        self.glove[k] = statistics.median(values)
        self.app.dirty = True
        self.app.status(f"{JOINT_KO[self.joint]} · {POINTS[self.joint][k][2]}: 장갑 {self.glove[k]:.3f} 기록 "
                        f"({len(values)} 샘플). 저장해야 적용됩니다")

    # -- drawing -------------------------------------------------------------------------------------
    def draw(self, signals) -> None:
        x = self.reading(signals)
        robot = self.robot()
        try:
            m = self.joint_map()
            self.error.config(text="")
        except JoystickError as exc:
            m = None
            self.error.config(text=str(exc))
        q = m.angle(x) if (m is not None and x is not None) else None
        for k, lab in enumerate(self.glove_labels):
            g = self.glove[k]
            lab.config(text="장갑 —  (미기록)" if g is None else f"장갑 {g:+.3f}")
        if q is None:
            self.now_label.config(text="장갑 값 없음" if x is None else f"장갑 {x:+.3f} → (점을 다 기록하세요)")
        else:
            frac = (q - robot[0]) / (robot[-1] - robot[0]) if abs(robot[-1] - robot[0]) > 1e-9 else 0.0
            self.now_label.config(text=f"장갑 {x:+.3f} → 로봇 {_deg(q):6.1f}°  ({frac:4.2f})")
        self._draw_dial(robot, q)
        self._draw_bar(x)

    def _draw_dial(self, robot: list[float], q: float | None) -> None:
        c = self.dial
        c.delete("all")
        cx, cy, r = DIAL_W / 2, DIAL_H - 14, DIAL_H - 30
        span = self.hi - self.lo

        def ang(v: float) -> float:  # the joint's range drawn over 180 deg, low end on the left
            return math.pi * (1.0 - (v - self.lo) / span)

        def at(v: float, rr: float) -> tuple[float, float]:
            a = ang(v)
            return cx + rr * math.cos(a), cy - rr * math.sin(a)

        c.create_arc(cx - r, cy - r, cx + r, cy + r, start=0, extent=180, style="arc", width=6, outline="#ddd")
        if q is not None:
            lo_q, hi_q = sorted((robot[0], robot[-1]))
            a0, a1 = math.degrees(ang(hi_q)), math.degrees(ang(lo_q))
            c.create_arc(cx - r, cy - r, cx + r, cy + r, start=a0, extent=a1 - a0, style="arc", width=6,
                         outline="#9ecae1")
        for v, label in ((self.lo, f"{_deg(self.lo):.0f}°"), (self.hi, f"{_deg(self.hi):.0f}°")):
            x, y = at(v, r + 10)
            c.create_text(x, y, text=label, font=("TkDefaultFont", 8))
        for k, v in enumerate(robot):
            x0, y0 = at(v, r - 8)
            x1, y1 = at(v, r + 6)
            c.create_line(x0, y0, x1, y1, fill=POINT_COLORS[k], width=3)
        if q is not None:
            x, y = at(q, r - 4)
            c.create_line(cx, cy, x, y, fill="#222", width=4, capstyle="round")
            c.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="#222")
            c.create_text(cx, cy - r / 2.6, text=f"{_deg(q):.1f}°", font=("TkDefaultFont", 13, "bold"))
        else:
            c.create_text(cx, cy - r / 2.6, text="—", font=("TkDefaultFont", 13, "bold"))

    def _draw_bar(self, x: float | None) -> None:
        c = self.bar
        c.delete("all")
        values = [g for g in self.glove if g is not None] + ([x] if x is not None else [])
        if not values:
            c.create_text(BAR_W / 2, BAR_H / 2, text="장갑 값 없음")
            return
        lo, hi = min(values), max(values)
        pad = max(0.15 * (hi - lo), 0.05 if "tip" not in self.source.get() else 5.0)
        lo, hi = lo - pad, hi + pad

        def px(v: float) -> float:
            return 10 + (v - lo) / (hi - lo) * (BAR_W - 20)

        c.create_line(10, BAR_H / 2, BAR_W - 10, BAR_H / 2, fill="#aaa", width=2)
        for k, g in enumerate(self.glove):
            if g is not None:
                c.create_line(px(g), 8, px(g), BAR_H - 8, fill=POINT_COLORS[k], width=3)
                c.create_text(px(g), 6, text=POINTS[self.joint][k][2], fill=POINT_COLORS[k],
                              font=("TkDefaultFont", 8), anchor="n")
        if x is not None:
            c.create_oval(px(x) - 7, BAR_H / 2 - 7, px(x) + 7, BAR_H / 2 + 7, fill="#222")
        c.create_text(12, BAR_H - 2, text=f"{lo:.2f}", anchor="sw", font=("TkDefaultFont", 7))
        c.create_text(BAR_W - 12, BAR_H - 2, text=f"{hi:.2f}", anchor="se", font=("TkDefaultFont", 7))


class TuneApp:
    def __init__(self, root: tk.Tk, node: GloveListener, path: Path, side: str, retarget_config: Path) -> None:
        self.root, self.node, self.path, self.side = root, node, path, side
        self.config = load_hand_retarget_config(retarget_config)
        self.targets = {p: e.target for p, e in self.config.examples.items()}
        self.calibration = HandCalibration.load(path, side=side)
        self.reader = JoystickRetargeter(JoystickMap({}), self.config.home_rad, self.calibration.sweeps)
        self.dirty = False
        side_ko = "오른손" if side == "right" else "왼손"
        root.title(f"손가락 튜닝 (조이스틱) — {side_ko} — {path.name}")
        outer = ttk.Frame(root, padding=6)
        outer.pack(fill="both", expand=True)
        sweeps = "있음 (다른 손가락 간섭을 뺀 값)" if self.calibration.sweeps is not None else "없음 (장갑 값 그대로)"
        ttk.Label(outer, text=f"보정 파일 {path}  ·  손가락별 맞춤 {sweeps}").grid(row=0, column=0, sticky="w")
        ttk.Label(outer, text="점마다: 로봇 각도를 정하고, 장갑 손가락을 그 자세로 둔 채 [지금 자세로 기록]. "
                              "바늘이 지금 로봇이 갈 각도입니다.", foreground="#555").grid(row=1, column=0, sticky="w")
        self.rows = [JointRow(self, outer, j, 2 + i) for i, j in enumerate(JOINTS)]
        thumb = ttk.LabelFrame(outer, text="엄지 장갑 값 (어느 값이 엄지 회전만 따라가는지 보기: 최소~최대는 [초기화] 부터)",
                               padding=4)
        thumb.grid(row=2 + len(JOINTS), column=0, sticky="ew", pady=2)
        self.thumb_labels = {}
        self.thumb_range: dict[str, list[float]] = {}
        for i, n in enumerate(THUMB_SIGNALS):
            lab = ttk.Label(thumb, text=n, width=40, font=("TkFixedFont", 9))
            lab.grid(row=i // 2, column=i % 2, sticky="w", padx=4)
            self.thumb_labels[n] = lab
        ttk.Button(thumb, text="초기화", command=self.thumb_range.clear).grid(row=0, column=2, rowspan=3, padx=6)
        bottom = ttk.Frame(outer)
        bottom.grid(row=3 + len(JOINTS), column=0, sticky="ew", pady=(6, 0))
        ttk.Button(bottom, text="저장", command=self.save).pack(side="left")
        ttk.Button(bottom, text="파일 다시 읽기", command=self.reload).pack(side="left", padx=4)
        self.status_label = ttk.Label(bottom, text="", width=110)
        self.status_label.pack(side="left", padx=8)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.status("저장한 맵은 손 노드가 따라가기 중이 아닐 때 읽습니다 ([홈] 뒤 [따라가기]).")

    def start_map(self, joint: str) -> JointMap | None:
        saved = self.calibration.joystick
        if saved is not None and joint in saved.joints:
            return saved.joints[joint]
        return default_joint(joint, self.calibration.medians, self.targets)

    def default_robot(self, joint: str) -> list[float]:
        lo, hi = self.config.limits_rad[joint]
        return [float(self.targets.get(pose, {}).get(joint, lo if i == 0 else hi))
                for i, (_, pose, _) in enumerate(POINTS[joint])]

    def status(self, text: str, error: bool = False) -> None:
        self.status_label.config(text=text, foreground="#c00000" if error else "#000")

    def tick(self) -> None:
        for _ in range(20):
            rclpy.spin_once(self.node, timeout_sec=0.0)
        signals = self.node.latest()
        for row in self.rows:
            row.draw(signals)
        for n, lab in self.thumb_labels.items():
            v = raw_reading(n, signals) if signals else None
            if v is None:
                lab.config(text=f"{n:12s}  —")
                continue
            rng = self.thumb_range.setdefault(n, [v, v])
            rng[0], rng[1] = min(rng[0], v), max(rng[1], v)
            lab.config(text=f"{n:12s} {v:+8.2f}   [{rng[0]:+.2f} ~ {rng[1]:+.2f}]")
        self.root.after(40, self.tick)

    def save(self) -> None:
        joints = {}
        try:
            for row in self.rows:
                m = row.joint_map()
                if m is None:
                    raise JoystickError(f"{JOINT_KO[row.joint]}: 기록하지 않은 점이 있습니다")
                joints[row.joint] = m
            joystick = JoystickMap(joints, time.strftime("%Y-%m-%dT%H:%M:%S"))
        except JoystickError as exc:
            self.status(f"저장 안 함: {exc}", error=True)
            return
        try:  # keep what other tools wrote meanwhile (e.g. [편 손 맞춤]); only the joystick part is ours
            current = HandCalibration.load(self.path, side=self.side)
        except Exception:  # noqa: BLE001
            current = self.calibration
        self.calibration = current.with_joystick(joystick)
        self.calibration.save(self.path)
        self.reader = JoystickRetargeter(JoystickMap({}), self.config.home_rad, self.calibration.sweeps)
        self.dirty = False
        self.status(f"저장됨 {time.strftime('%H:%M:%S')}: 손 노드는 [홈] 뒤 [따라가기] 에서 새 맵을 씁니다")

    def reload(self) -> None:
        self.calibration = HandCalibration.load(self.path, side=self.side)
        self.reader = JoystickRetargeter(JoystickMap({}), self.config.home_rad, self.calibration.sweeps)
        for row in self.rows:
            m = self.start_map(row.joint)
            row.source.set(m.source if m else SOURCE_CHOICES[row.joint][0])
            row.glove = list(m.glove) if m else [None] * len(POINTS[row.joint])
            for var, q in zip(row.robot_deg, m.robot if m else self.default_robot(row.joint)):
                var.set(round(_deg(q), 1))
        self.dirty = False
        self.status("파일에서 다시 읽었습니다")

    def close(self) -> None:
        if self.dirty and not messagebox.askyesno("손가락 튜닝", "저장하지 않은 변경이 있습니다. 닫을까요?"):
            return
        self.root.destroy()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--calibration", type=Path, default=None,
                    help="default: configs/hands/calibration/<user>_<side>.yaml")
    ap.add_argument("--glove-serial", default=None, help="default: the side's glove in nova2_gloves.yaml")
    ap.add_argument("--glove-topic", default="")
    ap.add_argument("--retarget-config", type=Path, default=DEFAULT_RETARGET)
    args = ap.parse_args(argv)
    side = check_side(args.side)
    path = args.calibration or REPO_ROOT / "configs" / "hands" / "calibration" / f"{args.user}_{side}.yaml"
    if not path.exists():
        raise SystemExit(f"{path} 이 없습니다: 먼저 [보정] 을 하세요")
    topic = args.glove_topic or glove_topic(args.glove_serial or load_gloves()[side].serial, side)
    rclpy.init()
    node = GloveListener(topic, side)
    try:
        root = tk.Tk()
        app = TuneApp(root, node, path, side, args.retarget_config)
        root.after(40, app.tick)
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
