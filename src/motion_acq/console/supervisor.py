"""Child processes of the operator console: own process group, log file, optional pty.

Units that read keys (macq head: Space; teleop-real / teleop-record: Space, R, Q;
calibrate and the OpenArm power-off prompt: Enter) run on a pseudo-terminal so
the console can send those keys and the programs keep their terminal behaviour.

Stopping sends SIGINT to the whole group once and waits: the motion_acq
programs take the robot to a safe pose on SIGINT (head home, arms home then
rest, hands open) before they exit, and a second SIGINT would cut that short.
Nothing is killed harder unless the operator asks (force_stop: SIGTERM, then
SIGKILL).

Every running child is listed in <log_dir>/running.json (pid, start time,
argv), so a console started after a crash finds what the last one left behind.
"""

from __future__ import annotations

import collections
import fcntl
import json
import os
import re
import signal
import struct
import subprocess
import termios
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[()][0-9A-Za-z]|\r")
TAIL_LINES = 600
BOX_DRAWING = ("│", "╭", "╰", "┃", "┏", "┗", "━", "─", "═")
PTY_ROWS, PTY_COLS = 50, 200


@dataclass
class Child:
    key: str
    argv: list[str]
    log_path: Path
    proc: subprocess.Popen
    started_at: float
    pty_master: int | None = None
    stopping_since: float | None = None
    forced: bool = False
    partial: str = ""  # the line being written (input() prompts end without a newline)
    io_lock: threading.Lock = field(default_factory=threading.Lock)  # pty master write vs close
    tail: collections.deque = field(default_factory=lambda: collections.deque(maxlen=TAIL_LINES))
    reader: threading.Thread | None = None

    @property
    def pid(self) -> int:
        return self.proc.pid

    def alive(self) -> bool:
        return self.proc.poll() is None


def process_start_epoch(pid: int) -> float | None:
    """Wall-clock start of a process from /proc (stat field 22 + boot time)."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        boot = next(float(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                    if line.startswith("btime "))
        return boot + int(fields[19]) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError, StopIteration):
        return None


class Supervisor:
    def __init__(self, log_dir: Path, registry: Path | None = None) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.children: dict[str, Child] = {}
        self.lock = threading.RLock()
        self.foreign: list[dict] = []  # left behind by an earlier console, kept in the registry while alive

    # -- start ------------------------------------------------------------------
    def start(self, key: str, argv: list[str], *, cwd: Path, env: dict[str, str], pty: bool) -> Child:
        with self.lock:
            old = self.children.get(key)
            if old is not None and old.alive():
                raise RuntimeError(f"{key} is already running (pid {old.pid})")
            log_path = self.log_dir / f"{key}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            log_file = open(log_path, "wb", buffering=0)  # noqa: SIM115 - owned by the reader thread
            log_file.write(f"$ {' '.join(argv)}\n".encode())
            env = {**env, "PYTHONUNBUFFERED": "1"}
            master = None
            try:
                if pty:
                    master, slave = os.openpty()
                    try:
                        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", PTY_ROWS, PTY_COLS, 0, 0))
                        env.setdefault("TERM", "xterm-256color")
                        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=slave, stdout=slave, stderr=slave,
                                                start_new_session=True, close_fds=True)
                    finally:
                        os.close(slave)
                else:
                    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            start_new_session=True, close_fds=True)
            except Exception:
                log_file.close()
                if master is not None:
                    os.close(master)
                raise
            child = Child(key=key, argv=list(argv), log_path=log_path, proc=proc, started_at=time.time(),
                          pty_master=master)
            child.reader = threading.Thread(target=self._pump, args=(child, log_file), daemon=True,
                                            name=f"console-{key}")
            child.reader.start()
            self.children[key] = child
            self._write_registry()
            return child

    def _read(self, child: Child) -> bytes:
        if child.pty_master is not None:
            return os.read(child.pty_master, 4096)
        stream = child.proc.stdout
        return stream.read1(4096) if stream is not None else b""  # type: ignore[attr-defined]

    def _pump(self, child: Child, log_file) -> None:
        """Copy the child's output to its log and keep a clean tail for the UI."""
        try:
            while True:
                try:
                    data = self._read(child)
                except OSError:  # pty closed when the child exits (EIO)
                    break
                if not data:
                    break
                log_file.write(data)
                lines = (child.partial + data.decode("utf-8", "replace")).split("\n")
                child.partial = ANSI.sub("", lines.pop())
                for line in lines:
                    clean = ANSI.sub("", line).strip()
                    if clean and not clean.startswith(BOX_DRAWING):  # rich panels stay in the log file only
                        child.tail.append(clean)
        finally:
            if child.partial.strip():
                child.tail.append(child.partial.rstrip())
            child.partial = ""
            child.proc.wait()
            log_file.write(f"\n[exit {child.proc.returncode}]\n".encode())
            log_file.close()
            with child.io_lock:
                if child.pty_master is not None:
                    try:
                        os.close(child.pty_master)
                    except OSError:
                        pass
                    child.pty_master = None
            with self.lock:
                self._write_registry()

    # -- control ----------------------------------------------------------------
    def get(self, key: str) -> Child | None:
        return self.children.get(key)

    def keys(self) -> list[str]:
        with self.lock:
            return list(self.children)

    def is_running(self, key: str) -> bool:
        child = self.children.get(key)
        return child is not None and child.alive()

    def send_key(self, key: str, text: str) -> None:
        child = self.children.get(key)
        if child is None or not child.alive():
            raise RuntimeError(f"{key} is not running")
        with child.io_lock:
            if child.pty_master is None:
                raise RuntimeError(f"{key} takes no keys")
            try:
                os.write(child.pty_master, text.encode("utf-8"))
            except OSError as exc:
                raise RuntimeError(f"{key}: the program no longer reads keys ({exc})") from exc

    def stop(self, key: str) -> bool:
        """SIGINT to the group, once: the program returns the robot to its safe pose and exits.

        Returns False if the child is not running or already stopping (a second
        SIGINT would skip the safe-pose return). Check and mark are atomic."""
        with self.lock:
            child = self.children.get(key)
            if child is None or not child.alive() or child.stopping_since is not None:
                return False
            child.stopping_since = time.time()
        try:
            os.killpg(child.pid, signal.SIGINT)
        except ProcessLookupError:
            return False
        return True

    def force_stop(self, key: str, grace_s: float = 5.0) -> None:
        """Operator-confirmed: SIGTERM, then SIGKILL after grace_s (no safe pose). Blocks up to grace_s."""
        with self.lock:
            child = self.children.get(key)
            if child is None or not child.alive():
                return
            child.forced = True
            child.stopping_since = child.stopping_since or time.time()
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.time() + grace_s
        while time.time() < deadline and child.alive():
            time.sleep(0.1)
        if child.alive():
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    # -- state ------------------------------------------------------------------
    def snapshot(self, key: str, lines: int = 14) -> dict:
        child = self.children.get(key)
        if child is None:
            return {"running": False, "rc": None, "pid": None, "tail": [], "partial": ""}
        rc = child.proc.poll()
        return {
            "running": rc is None,
            "rc": rc,
            "pid": child.pid,
            "argv": child.argv,
            "started_at": child.started_at,
            "stopping_s": None if child.stopping_since is None or rc is not None
            else round(time.time() - child.stopping_since, 1),
            "forced": child.forced,
            "log": str(child.log_path),
            "tail": list(child.tail)[-lines:],
            "partial": child.partial.strip() if rc is None else "",
        }

    def log_tail(self, key: str, lines: int = 300) -> list[str]:
        child = self.children.get(key)
        return [] if child is None else list(child.tail)[-lines:]

    def running(self) -> list[str]:
        with self.lock:
            items = list(self.children.items())
        return [key for key, child in items if child.alive()]

    def _write_registry(self) -> None:
        if self.registry is None:
            return
        with self.lock:
            children = list(self.children.values())
        entries = [{"key": c.key, "pid": c.pid, "start": process_start_epoch(c.pid), "argv": c.argv}
                   for c in children if c.alive()]
        entries += [e for e in self.foreign if _same_process(e)]
        tmp = self.registry.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=1))
        tmp.replace(self.registry)


def _same_process(entry: dict) -> bool:
    start = process_start_epoch(int(entry["pid"]))
    return start is not None and entry.get("start") is not None and abs(start - float(entry["start"])) < 1.0


def left_behind(registry: Path) -> list[dict]:
    """Children of an earlier console that still run (same pid and start time)."""
    try:
        entries = json.loads(registry.read_text())
    except (OSError, ValueError):
        return []
    alive = []
    for entry in entries if isinstance(entries, list) else []:
        try:
            pid, start = int(entry["pid"]), entry.get("start")
        except (KeyError, TypeError, ValueError):
            continue
        now_start = process_start_epoch(pid)
        if now_start is not None and start is not None and abs(now_start - float(start)) < 1.0:
            alive.append({"key": str(entry.get("key")), "pid": pid, "start": now_start,
                          "argv": entry.get("argv") or []})
    return alive


def stop_left_behind(entry: dict) -> None:
    """SIGINT to a left-behind group after re-checking it is the same process."""
    if not _same_process(entry):
        return  # gone, or the pid now belongs to another process
    try:
        os.killpg(int(entry["pid"]), signal.SIGINT)
    except (ProcessLookupError, PermissionError):
        return
