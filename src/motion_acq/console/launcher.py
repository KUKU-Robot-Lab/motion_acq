"""macq console: one window for the assistant (Quest, head, arms, gloves, hands, recording).

    MACQ_STATION=arm4090 macq console            # fake mode: nothing touches hardware
    MACQ_STATION=arm4090 macq console --real     # real robot: units that move ask first
    MACQ_STATION=arm4090 macq console --install-desktop   # app menu entry (real mode)

A local server (127.0.0.1:8790) runs the programs as before (macq head,
teleop-real, teleop-record, hand_node, quest_usb.sh, nova2.sh) and a Chrome app
window shows them, separate from the sim2real console. Closing the window
leaves the server and the robot as they are; open it again with the same
command. The server stops only from the window ([콘솔 종료]) or Ctrl+C here,
and then first takes every unit to its safe pose (SIGINT, as Ctrl+C would).
The RH56F1 EtherCAT driver is never started here: the s2r console runs it and
the hand nodes join its ROS domain (hands.ros_domain_id of the station rig).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path

from motion_acq.config import STATION_ENV
from motion_acq.console.units import ROOT, Station, UnitError

DEFAULT_PORT = 8790
CHROMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
log = logging.getLogger("motion_acq.console")


def running_console(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=1.5) as reply:
            data = json.load(reply)
    except (OSError, ValueError):
        return None
    return data if data.get("app") == "macq-console" else None


def desktop_env() -> dict[str, str]:
    """DISPLAY / XAUTHORITY of the operator's desktop, also when started over ssh."""
    env = dict(os.environ)
    if not env.get("DISPLAY"):
        who = subprocess.run(["who"], capture_output=True, text=True).stdout
        user = os.environ.get("USER", "")
        for line in who.splitlines():
            parts = line.split()
            if parts and parts[0] == user and parts[-1].startswith("(:") and parts[-1].endswith(")"):
                env["DISPLAY"] = parts[-1][1:-1]
                break
        else:
            env["DISPLAY"] = ":0"
    if not env.get("XAUTHORITY"):
        gdm = Path(f"/run/user/{os.getuid()}/gdm/Xauthority")
        if gdm.exists():
            env["XAUTHORITY"] = str(gdm)
    return env


def open_window(url: str, log_dir: Path) -> bool:
    chrome = next((shutil.which(name) for name in CHROMES if shutil.which(name)), None)
    if chrome is None:
        print(f"no Chrome/Chromium found; open {url} in a browser", file=sys.stderr)
        return False
    profile = Path.home() / ".config" / "macq-console" / "chrome"
    profile.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "chrome.log").open("ab") as out:
        subprocess.Popen([chrome, f"--app={url}", "--class=macq-console", f"--user-data-dir={profile}",
                          "--window-size=1560,1000", "--no-first-run", "--no-default-browser-check"],
                         env=desktop_env(), stdout=out, stderr=out, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    return True


def install_desktop(station: str, port: int) -> Path:
    path = Path.home() / ".local" / "share" / "applications" / "macq-console.desktop"
    path.parent.mkdir(parents=True, exist_ok=True)
    command = (f"cd {ROOT} && mkdir -p logs/console && {STATION_ENV}={station} exec {sys.executable} -m "
               f"motion_acq.console.launcher --real --port {port} >> logs/console/launcher.log 2>&1")
    path.write_text("\n".join([
        "[Desktop Entry]", "Type=Application", f"Name=macq 콘솔 ({station})",
        "Comment=motion_acq operator console (Quest, head, arms, gloves, hands, recording)",
        f"Exec=bash -c '{command}'", "Icon=applications-engineering", "Terminal=false",
        "StartupWMClass=macq-console", "Categories=Utility;", ""]))
    path.chmod(0o755)
    return path


async def serve(console, port: int, *, window: bool) -> None:
    from aiohttp import web

    from motion_acq.console.server import build_app

    loop = asyncio.get_running_loop()
    done = asyncio.Event()
    abandon = asyncio.Event()

    def on_signal(name: str) -> None:
        if done.is_set():
            log.error("%s again: leaving the remaining units unsupervised (their terminals close; an arm "
                      "waiting for Enter switches its motors off)", name)
            abandon.set()
        done.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, on_signal, sig.name)
    # an ssh logout must not take the console (and the units' terminals) down
    loop.add_signal_handler(signal.SIGHUP, lambda: log.warning("SIGHUP ignored: the console keeps running"))
    runner = web.AppRunner(build_app(console, port=port, on_quit=done.set), shutdown_timeout=1.0, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    url = f"http://127.0.0.1:{port}/"
    log.info("macq console %s (%s) on %s", console.station.name, console.mode, url)
    if window:
        open_window(url, console.log_root)
    await done.wait()
    log.info("stopping: every unit to its safe pose first (no new starts)")
    await asyncio.to_thread(console.begin_shutdown)
    # Keep serving (the window can still answer an Enter prompt) until every unit has exited.
    while not abandon.is_set():
        left = await asyncio.to_thread(console.wait_all, 15.0)
        if not left:
            break
        log.warning("still stopping: %s (the window stays usable; Ctrl+C again leaves them)", ", ".join(
            f"{key} (PID {console.sup.get(key).pid})" for key in left))
    if console.probes:
        console.probes.close()
    await runner.cleanup()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", help="real robot (units that move ask first)")
    mode.add_argument("--fake", action="store_true", help="fake devices only (default)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-window", action="store_true", help="server only (open http://127.0.0.1:PORT yourself)")
    ap.add_argument("--install-desktop", action="store_true", help="write the app menu entry and exit")
    args = ap.parse_args(argv)
    station_name = os.environ.get(STATION_ENV, "")
    try:
        station = Station.load(station_name)
    except UnitError as exc:
        raise SystemExit(str(exc)) from exc
    if args.install_desktop:
        print(f"installed {install_desktop(station.name, args.port)}")
        return 0
    url = f"http://127.0.0.1:{args.port}/"
    other = running_console(args.port)
    if other is not None:
        print(f"macq console already runs ({other.get('station')}, {other.get('mode')}); opening its window")
        if not args.no_window:
            open_window(url, ROOT / "logs" / "console")
        return 0

    os.chdir(ROOT)
    from motion_acq.cpu import keep_off_rt
    from motion_acq.console.core import Console

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    log.info(keep_off_rt())  # children inherit: off the RH56F1 EtherCAT cores
    console = Console(station, "real" if args.real else "fake")
    asyncio.run(serve(console, args.port, window=not args.no_window))
    return 0


if __name__ == "__main__":
    sys.exit(main())
