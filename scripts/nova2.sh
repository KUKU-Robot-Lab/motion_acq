#!/usr/bin/env bash
# SenseGlove Nova 2 link on this host: SenseCom + Bluetooth LE.
#
#   scripts/nova2.sh up [both|right|left]      # start SenseCom, wait for the gloves
#   scripts/nova2.sh status                    # adapter, gloves, SenseCom, glove topics
#   scripts/nova2.sh driver                    # glove driver (nova2.launch.py) in the background, logged
#   scripts/nova2.sh stop                      # stop that driver (its whole process group)
#   scripts/nova2.sh connect [both|right|left] # fallback: BlueZ connect (bumsu's connect_senseglove.py)
#   scripts/nova2.sh disconnect [both|right|left]
#
# Then: scripts/nova2.sh driver (the glove driver refuses to start without
# SenseCom), then calibrate (every SenseCom start: the raw ranges move,
# macq station --real checks the calibration is newer). SenseCom and the
# driver run on the general cores of the sim2real CPU plan (motion_acq.cpu),
# never on the RH56F1 EtherCAT master cores.
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
DRIVER_PID="$LOG_DIR/driver.pid"
CONNECT_TIMEOUT_S="${NOVA2_CONNECT_TIMEOUT_S:-10}"
UP_TIMEOUT_S="${NOVA2_UP_TIMEOUT_S:-60}"
PLAYER_LOG="$HOME/.config/unity3d/SenseGlove/SenseCom 1.9.0/Player.log"
SCAN_TIMEOUT_S="${NOVA2_SCAN_TIMEOUT_S:-15}"

die() { echo "error: $*" >&2; exit 1; }

# taskset prefix for the general cores; empty when the PC is not pinned.
general_cores() {
  local list; list="$("$PY" -m motion_acq.cpu --general 2>/dev/null || true)"
  if [[ -n "$list" ]] && command -v taskset >/dev/null; then echo "taskset -c $list"; fi
}
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
  if [[ -n "$pids" ]]; then
    echo "SenseCom: running (PID ${pids//$'\n'/ }; log: $PLAYER_LOG)"
  else
    echo "SenseCom: not running"
  fi
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
  # From ssh there is no XAUTHORITY; the GDM X11 session keeps its cookie here.
  local xauth="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
  local pin; pin="$(general_cores)"
  # shellcheck disable=SC2086
  (cd "$(dirname "$SENSECOM")" && DISPLAY="$display" XAUTHORITY="$xauth" setsid $pin "$SENSECOM" >"$log" 2>&1 < /dev/null &)
  sleep 3
  pids="$(sensecom_pids)"
  [[ -n "$pids" ]] || die "SenseCom exited at start; see $log"
  echo "SenseCom running (PID ${pids//$'\n'/ }, DISPLAY $display, log $log)"
}

cmd_driver() {
  if pgrep -f "motion_acq_hand nova2.launch.py" >/dev/null; then echo "glove driver already running"; return 0; fi
  mkdir -p "$LOG_DIR"
  [[ -n "$(sensecom_pids)" ]] || die "SenseCom is not running (scripts/nova2.sh up first)"
  local distro="${ROS_DISTRO:-}"
  [[ -n "$distro" ]] || for d in humble jazzy; do [[ -f "/opt/ros/$d/setup.bash" ]] && distro="$d" && break; done
  [[ -f "$ROOT/ros_ws/install/setup.bash" ]] || die "ros_ws not built (scripts/ros_ws_setup.sh --full)"
  mkdir -p "$LOG_DIR"
  local log; log="$LOG_DIR/driver_$(date +%Y%m%d_%H%M%S).log"
  local pin; pin="$(general_cores)"
  # setsid -f, not "&": a background job of a script has SIGINT ignored, so the launch could
  # not pass a stop on to its nodes (10.04: SIGTERM left orphaned ros2_control_node).
  # The launch writes its PID (= its process group) for "stop".
  setsid -f bash -c "echo \$\$ > '$DRIVER_PID'; source /opt/ros/$distro/setup.bash && source '$ROOT/ros_ws/install/setup.bash' && exec $pin ros2 launch motion_acq_hand nova2.launch.py" \
    >"$log" 2>&1 < /dev/null
  local deadline=$((SECONDS + 15))
  until grep -q "Configured and activated senseglove_state_broadcaster" "$log" 2>/dev/null; do
    (( SECONDS >= deadline )) && { echo "glove driver not confirmed yet; see $log" >&2; return 1; }
    sleep 1
  done
  echo "glove driver up (PID $(cat "$DRIVER_PID" 2>/dev/null), log $log); stop: $0 stop"
}

cmd_stop() {
  # The driver's whole process group (launch + ros2_control_node + robot_state_publisher), by its PID file.
  [[ -f "$DRIVER_PID" ]] || { echo "no glove driver PID file ($DRIVER_PID)"; return 0; }
  local pid; pid="$(cat "$DRIVER_PID")"
  if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "nova2.launch.py"; then
    echo "PID $pid is not the glove driver (already gone?); removing the PID file"; rm -f "$DRIVER_PID"; return 0
  fi
  kill -INT -- "-$pid" 2>/dev/null || true
  local deadline=$((SECONDS + 10))
  while kill -0 -- "-$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 0.5; done
  if kill -0 -- "-$pid" 2>/dev/null; then kill -TERM -- "-$pid" 2>/dev/null || true; sleep 2; fi
  kill -0 -- "-$pid" 2>/dev/null && die "glove driver group $pid still alive"
  rm -f "$DRIVER_PID"
  echo "glove driver stopped (group $pid)"
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
    [[ -z "$pending" ]] && { echo "gloves connected; next: $0 driver"; return 0; }
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
  driver) cmd_driver ;;
  stop) cmd_stop ;;
  up) cmd_up "$which" ;;
  *) die "usage: $0 {up|driver|stop|status|sensecom|connect|disconnect} [both|right|left]" ;;
esac
