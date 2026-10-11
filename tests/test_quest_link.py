"""The Quest over the cable or Wi-Fi (10.12 user): which adb device, and the Wi-Fi switch."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from motion_acq.quest_view import link
from motion_acq.quest_view.link import Device, parse_devices, pick

ROOT = Path(__file__).resolve().parents[1]
USB = Device("2G0YC5ZH3F01LM", "device")
WIFI = Device("192.168.1.77:5555", "device")


def test_adb_devices_output_is_parsed():
    text = ("* daemon started successfully\nList of devices attached\n"
            "2G0YC5ZH3F01LM\tdevice\n192.168.1.77:5555\tdevice\nX\tunauthorized\n\n")
    assert parse_devices(text) == [USB, WIFI, Device("X", "unauthorized")]
    assert WIFI.wifi and not USB.wifi


def test_the_remembered_wifi_device_wins_while_it_is_up():
    assert pick([USB, WIFI], WIFI.serial) == WIFI  # cable in to charge: still Wi-Fi
    assert pick([USB, WIFI]) == USB  # not switched: the cable
    assert pick([USB], WIFI.serial) == USB  # Wi-Fi down: the cable
    assert pick([WIFI]) == WIFI
    assert pick([Device("X", "unauthorized")]) == Device("X", "unauthorized")  # reported as is
    assert pick([]) is None


# ---- quest_usb.sh against a fake adb ----------------------------------------------------------


@pytest.fixture()
def fake_adb(tmp_path):
    """adb that knows one Quest: on USB while tmp/usb exists, on Wi-Fi once 'tcpip' + 'connect' ran."""
    state = tmp_path / "state"
    state.mkdir()
    (state / "usb").touch()
    log = tmp_path / "calls"
    adb = tmp_path / "adb"
    adb.write_text(f"""#!/usr/bin/env bash
S={state}
echo "${{ANDROID_SERIAL:-}}|$*" >> {log}
args=("$@"); [[ "$1" == "-s" ]] && args=("${{@:3}}")
case "${{args[0]}}" in
  devices) echo "List of devices attached"
           [[ -e $S/usb ]] && printf 'USB1\\tdevice\\n'
           [[ -e $S/wifi ]] && printf '10.0.0.5:5555\\tdevice\\n'; true ;;
  tcpip) touch $S/tcp ;;
  connect) if [[ -e $S/tcp ]]; then touch $S/wifi; echo "connected to ${{args[1]}}"; else echo "failed to connect"; fi ;;
  disconnect) rm -f $S/wifi ;;
  usb) rm -f $S/tcp $S/wifi ;;
  shell) if [[ "${{args[*]}}" == *wlan0* ]]; then echo "    inet 10.0.0.5/24 brd 10.0.0.255 scope global wlan0"; fi ;;
  reverse) [[ "${{args[1]:-}}" == "--list" ]] && [[ -e $S/reverse ]] && echo "UsbFfs tcp:8787 tcp:8787"; true ;;
  forward) true ;;
esac
""")
    adb.chmod(0o755)
    env = {**os.environ, "ADB": str(adb), "MACQ_QUEST_WIFI_FILE": str(tmp_path / "quest_wifi")}
    env.pop("ANDROID_SERIAL", None)

    def run(*args):
        return subprocess.run(["bash", str(ROOT / "scripts" / "quest_usb.sh"), *args], env=env,
                              capture_output=True, text=True, timeout=30)

    return run, state, log, tmp_path / "quest_wifi"


def test_the_switch_to_wifi_moves_the_page_link_and_is_remembered(fake_adb):
    run, state, log, saved = fake_adb
    (state / "reverse").touch()  # the headset page is open over the cable
    done = run("wifi")
    assert done.returncode == 0, done.stderr
    assert saved.read_text().strip() == "10.0.0.5:5555"
    calls = log.read_text().splitlines()
    assert "|-s USB1 tcpip 5555" in calls
    assert "10.0.0.5:5555|reverse tcp:8787 tcp:8787" in calls  # set again on the Wi-Fi device
    (state / "usb").unlink()  # the cable comes out
    log.write_text("")
    assert run("status").returncode == 0
    assert any(line.startswith("10.0.0.5:5555|") for line in log.read_text().splitlines())


def test_after_the_switch_commands_use_wifi_even_with_the_cable_in(fake_adb):
    run, state, log, _ = fake_adb
    assert run("wifi").returncode == 0
    log.write_text("")
    assert run("worn").returncode == 0
    assert all(line.startswith("10.0.0.5:5555|") for line in log.read_text().splitlines()
               if "shell" in line)


def test_a_lost_wifi_device_is_dialled_again(fake_adb):
    run, state, log, _ = fake_adb
    assert run("wifi").returncode == 0
    (state / "wifi").unlink()  # adb server restarted: the TCP device is gone, the Quest still listens
    (state / "usb").unlink()
    assert run("worn").returncode == 0
    assert "|connect 10.0.0.5:5555" in log.read_text().splitlines()


def test_the_switch_needs_the_cable(fake_adb):
    run, state, _, saved = fake_adb
    (state / "usb").unlink()
    done = run("wifi")
    assert done.returncode != 0 and "cable" in done.stderr and not saved.exists()


def test_wifi_off_goes_back_to_the_cable(fake_adb):
    run, state, log, saved = fake_adb
    assert run("wifi").returncode == 0
    done = run("wifi-off")
    assert done.returncode == 0 and not saved.exists()
    assert "|-s USB1 usb" in log.read_text().splitlines()


def test_python_picks_the_same_device(fake_adb, monkeypatch):
    run, state, _, saved = fake_adb
    assert run("wifi").returncode == 0
    adb = str(Path(saved).parent / "adb")
    monkeypatch.setattr(link, "WIFI_FILE", saved)
    monkeypatch.setattr("motion_acq.quest_view.devtools.adb_path", lambda: adb)
    assert link.current() == Device("10.0.0.5:5555", "device")
    assert link.adb_env()["ANDROID_SERIAL"] == "10.0.0.5:5555"
    saved.unlink()
    assert link.current() == Device("USB1", "device")
