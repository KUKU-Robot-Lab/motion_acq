#!/usr/bin/env bash
# Build the motion_acq ROS 2 workspace (STEP 3 hand nodes).
#
#   scripts/ros_ws_setup.sh            # senseglove_msgs only (fake glove, hand node)
#   scripts/ros_ws_setup.sh --full     # whole senseglove_ros (real Nova 2: SenseCom, ros2_control)
#
# senseglove_ros (MIT, Adjuvo) is cloned into ros_ws/external/ on the branch
# matching $ROS_DISTRO (humble | jazzy) and is not part of this repo.
# rh56f1_interfaces comes from robot_control (ROBOT_CONTROL_WS, default
# ~/rl_ws/robot_control/ros_ws/install), the same build the s2r stack uses.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$ROOT/ros_ws"
FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1
DISTRO="${ROS_DISTRO:-}"
if [[ -z "$DISTRO" ]]; then
  for d in humble jazzy; do [[ -f "/opt/ros/$d/setup.bash" ]] && DISTRO="$d" && break; done
fi
[[ -n "$DISTRO" ]] || { echo "error: no ROS 2 under /opt/ros" >&2; exit 1; }
RC_WS="${ROBOT_CONTROL_WS:-$HOME/rl_ws/robot_control/ros_ws/install}"

EXT="$WS/external/senseglove_ros"
if [[ ! -d "$EXT/.git" ]]; then
  mkdir -p "$WS/external"
  git clone --quiet --filter=blob:none --sparse -b "$DISTRO" \
    https://github.com/Adjuvo/senseglove_ros.git "$EXT"
fi
git -C "$EXT" fetch --quiet origin "$DISTRO"
git -C "$EXT" checkout --quiet "$DISTRO"
git -C "$EXT" pull --quiet --ff-only origin "$DISTRO"
if [[ "$FULL" -eq 1 ]]; then
  git -C "$EXT" sparse-checkout disable
else
  git -C "$EXT" sparse-checkout set senseglove/senseglove_msgs
fi
echo "senseglove_ros $DISTRO @ $(git -C "$EXT" rev-parse --short HEAD) ($([[ $FULL -eq 1 ]] && echo full || echo msgs only))"

set +u
# shellcheck disable=SC1090
source "/opt/ros/$DISTRO/setup.bash"
if [[ -f "$RC_WS/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "$RC_WS/setup.bash"
else
  echo "warning: $RC_WS/setup.bash missing; rh56f1_interfaces must come from elsewhere" >&2
fi
set -u
cd "$WS"
colcon build --symlink-install --base-paths src external
echo "done: source $WS/install/setup.bash"
