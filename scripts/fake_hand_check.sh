#!/usr/bin/env bash
# STEP 3 fake end-to-end check (no glove, no hand): fake Nova 2 -> calibrate ->
# hand_node -> fake RH56F1, isolated on ROS_DOMAIN_ID=177 localhost-only.
#
#   scripts/fake_hand_check.sh [right|left] [seconds] [object register]
#
# The fake hand has an object in the index finger's way (register 1300 by
# default, -1 = none): the log must show the glove index brake while it presses.
set -euo pipefail

SIDE="${1:-right}"
SECONDS_RUN="${2:-20}"
OBJECT_REG="${3:-1300}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISTRO="${ROS_DISTRO:-}"
if [[ -z "$DISTRO" ]]; then
  for d in humble jazzy; do [[ -f "/opt/ros/$d/setup.bash" ]] && DISTRO="$d" && break; done
fi
RC_WS="${ROBOT_CONTROL_WS:-$HOME/rl_ws/robot_control/ros_ws/install}"

export ROS_DOMAIN_ID=177 ROS_LOCALHOST_ONLY=1  # the fakes refuse anything else
set +u
# shellcheck disable=SC1090
source "/opt/ros/$DISTRO/setup.bash"
# shellcheck disable=SC1091
source "$RC_WS/setup.bash"
# shellcheck disable=SC1091
source "$ROOT/ros_ws/install/setup.bash"
set -u
BIN="$ROOT/ros_ws/install/motion_acq_hand/lib/motion_acq_hand"
CAL="$ROOT/logs/hand/fake_calibration_${SIDE}.yaml"
mkdir -p "$ROOT/logs/hand"

echo "== 1. calibrate against the fake glove ($SIDE)"
"$BIN/fake_glove" --ros-args -p side:="$SIDE" -p mode:=pose > "$ROOT/logs/hand/fake_glove_cal.log" 2>&1 &
GLOVE_PID=$!
trap 'kill "$GLOVE_PID" 2>/dev/null || true' EXIT
sleep 2
"$BIN/calibrate" --side "$SIDE" --user fake --fake --yes --out "$CAL"
kill "$GLOVE_PID"
wait "$GLOVE_PID" 2>/dev/null || true
trap - EXIT

echo "== 2. fake chain for ${SECONDS_RUN}s (glove drops 1 s every 8 s)"
START_MARK="$(date +%s)"
timeout --foreground --signal=INT "$SECONDS_RUN" ros2 launch motion_acq_hand fake_hand.launch.py \
  side:="$SIDE" calibration:="$CAL" dropout_every_s:=8.0 dropout_s:=1.0 object_index_reg:="$OBJECT_REG" \
  > "$ROOT/logs/hand/fake_chain.log" 2>&1 || true
sleep 1

echo "== 3. analyse the hand log"
LOG="$(ls -t "$ROOT"/logs/hand/hand_"$SIDE"_*.jsonl | head -1)"
[[ "$(stat -c %Y "$LOG")" -ge "$START_MARK" ]] || { echo "no new hand log" >&2; exit 1; }
python3 "$ROOT/scripts/analyse_hand_log.py" "$LOG"
