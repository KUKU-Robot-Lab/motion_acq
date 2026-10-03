#!/usr/bin/env bash
# SenseGlove Nova 2 link on this host: SenseCom + Bluetooth LE.
#
#   scripts/nova2.sh up [both|right|left]      # start SenseCom, wait for the gloves
#   scripts/nova2.sh status                    # adapter, gloves, SenseCom, glove topics
#   scripts/nova2.sh connect [both|right|left] # fallback: BlueZ connect (bumsu's connect_senseglove.py)
#   scripts/nova2.sh disconnect [both|right|left]
#
# Then: ros2 launch motion_acq_hand nova2.launch.py (the glove driver refuses
# to start without SenseCom), then calibrate (every SenseCom start: the raw
# ranges move, macq station --real checks the calibration is newer).
#
# Firmware v2 gloves are BLE: SenseCom finds them by name and connects itself,
# BlueZ only needs to trust them (no pairing). The gloves are the shared lab
# pair (configs/hands/nova2_gloves.yaml): a glove holds one connection and a
# running SenseCom keeps reconnecting it, so close SenseCom on the other
# station first. If SenseCom dies, the driver keeps publishing the last
# values; the hand node holds on such frozen data, restart SenseCom and
# recalibrate. SenseCom also probes serial ports (/dev/ttyUSB*).
# SenseCom comes from this repo's senseglove_ros build (scripts/ros_ws_setup.sh --full).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
SENSECOM="$ROOT/ros_ws/install/senseglove_com/share/senseglove_com/Linux/SenseCom_Linux_Latest/SenseCom.x86_64"
LOG_DIR="$ROOT/logs/nova2"
CONNECT_TIMEOUT_S="${NOVA2_CONNECT_TIMEOUT_S:-10}"
UP_TIMEOUT_S="${NOVA2_UP_TIMEOUT_S:-60}"
PLAYER_LOG="$HOME/.config/unity3d/SenseGlove/SenseCom 1.9.0/Player.log"
SCAN_TIMEOUT_S="${NOVA2_SCAN_TIMEOUT_S:-15}"

die() { echo "error: $*" >&2; exit 1; }
command -v bluetoothctl >/dev/null || die "bluetoothctl not found (bluez)"
[[ -x "$PY" ]] || die "$PY missing: run uv sync first"

# "side serial mac name" per glove, from the single inventory file.
gloves() {
  "$PY" - "$1" <<'EOF'
import sys
from motion_acq.hand.nova2 import load_gloves
gloves = load_gloves()
which = sys.argv[1]
sides = ["right", "left"] if which == "both" else [which]
for side in sides:
    if side not in gloves:
        sys.exit(f"unknown side {side!r}: use both, right or left")
    g = gloves[side]
    print(side, g.serial, g.mac, g.name.replace(" ", "_"))
EOF
}

prop() { bluetoothctl info "$1" 2>/dev/null | awk -v k="$2:" '$1 == k {print $2; exit}'; }
is_connected() { [[ "$(prop "$1" Connected)" == "yes" ]]; }
sensecom_pids() { pgrep -f "SenseCom.x86_64" || true; }

cmd_status() {
  echo "adapter: $(bluetoothctl show 2>/dev/null | awk '$1 == "Controller" {print $2} $1 == "Powered:" {print "powered " $2}' | paste -sd' ')"
  while read -r side serial mac name; do
    if bluetoothctl info "$mac" >/dev/null 2>&1; then
      echo "$side ${name//_/ } $mac: trusted $(prop "$mac" Trusted), connected $(prop "$mac" Connected)"
    else
      echo "$side ${name//_/ } $mac: not seen yet (scripts/nova2.sh up)"
    fi
  done < <(gloves both)
  local pids; pids="$(sensecom_pids)"
  echo "SenseCom: ${pids:+running (PID ${pids//$'\n'/ })}${pids:-not running} (log: $PLAYER_LOG)"
  if [[ -n "${ROS_DISTRO:-}" ]] && command -v ros2 >/dev/null; then
    echo "glove topics (ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}):"
    timeout 15 ros2 topic list 2>/dev/null | grep '^/senseglove/.*/senseglove_states$' | sed 's/^/  /' || echo "  none"
  fi
}

connect_one() {
  local side="$1" mac="$2" name="${3//_/ }"
  if is_connected "$mac"; then echo "$side $name: already connected"; return 0; fi
  bluetoothctl power on >/dev/null
  if ! bluetoothctl info "$mac" >/dev/null 2>&1; then
    echo "$side $name: scanning ${SCAN_TIMEOUT_S} s (glove on, LED lit)..."
    bluetoothctl --timeout "$SCAN_TIMEOUT_S" scan on >/dev/null 2>&1 || true
    bluetoothctl info "$mac" >/dev/null 2>&1 || {
      echo "$side $name: not found. Is it on, and disconnected from the other station?" >&2; return 1; }
  fi
  bluetoothctl trust "$mac" >/dev/null
  bluetoothctl --timeout "$CONNECT_TIMEOUT_S" connect "$mac" >/dev/null 2>&1 || true
  for _ in $(seq 1 "$CONNECT_TIMEOUT_S"); do
    if is_connected "$mac"; then echo "$side $name: connected"; return 0; fi
    sleep 1
  done
  echo "$side $name: connect failed (power-cycle the glove; disconnect it on the other station)" >&2
  return 1
}

cmd_connect() {
  local failed=0
  while read -r side serial mac name; do connect_one "$side" "$mac" "$name" || failed=1; done < <(gloves "$1")
  return "$failed"
}

cmd_disconnect() {
  while read -r side serial mac name; do
    bluetoothctl disconnect "$mac" >/dev/null 2>&1 || true
    echo "$side ${name//_/ }: connected $(prop "$mac" Connected)"
  done < <(gloves "$1")
}

desktop_display() {
  # SenseCom is a Unity app: it needs the operator's desktop session.
  [[ -n "${DISPLAY:-}" ]] && { echo "$DISPLAY"; return; }
  who | awk -v u="$USER" '$1 == u && $NF ~ /^\(:[0-9]+\)$/ {gsub(/[()]/, "", $NF); print $NF; exit}'
}

cmd_sensecom() {
  local pids; pids="$(sensecom_pids)"
  if [[ -n "$pids" ]]; then echo "SenseCom already running (PID ${pids//$'\n'/ })"; return 0; fi
  [[ -x "$SENSECOM" ]] || die "SenseCom missing: $SENSECOM (scripts/ros_ws_setup.sh --full)"
  local display; display="$(desktop_display)"
  [[ -n "$display" ]] || die "no desktop display for $USER: start SenseCom from the desktop terminal"
  mkdir -p "$LOG_DIR"
  local log; log="$LOG_DIR/sensecom_$(date +%Y%m%d_%H%M%S).log"
  (cd "$(dirname "$SENSECOM")" && DISPLAY="$display" setsid "$SENSECOM" >"$log" 2>&1 < /dev/null &)
  sleep 3
  pids="$(sensecom_pids)"
  [[ -n "$pids" ]] || die "SenseCom exited at start; see $log"
  echo "SenseCom running (PID ${pids//$'\n'/ }, DISPLAY $display, log $log)"
}

cmd_up() {
  bluetoothctl power on >/dev/null
  while read -r side serial mac name; do bluetoothctl trust "$mac" >/dev/null 2>&1 || true; done < <(gloves "$1")
  cmd_sensecom
  echo "waiting up to ${UP_TIMEOUT_S} s for SenseCom to connect the gloves (power them on, LED lit)..."
  local deadline=$((SECONDS + UP_TIMEOUT_S)) pending
  while :; do
    pending=""
    while read -r side serial mac name; do is_connected "$mac" || pending+=" $side"; done < <(gloves "$1")
    [[ -z "$pending" ]] && { echo "gloves connected; next: ros2 launch motion_acq_hand nova2.launch.py"; return 0; }
    (( SECONDS >= deadline )) && break
    sleep 2
  done
  echo "not connected:$pending. Glove on? SenseCom closed on the other station? Fallback: $0 connect" >&2
  return 1
}

which="${2:-both}"
case "$which" in both|right|left) ;; *) die "side must be both, right or left, not $which" ;; esac
case "${1:-status}" in
  status) cmd_status ;;
  connect) cmd_connect "$which" ;;
  disconnect) cmd_disconnect "$which" ;;
  sensecom) cmd_sensecom ;;
  up) cmd_up "$which" ;;
  *) die "usage: $0 {up|status|sensecom|connect|disconnect} [both|right|left]" ;;
esac
