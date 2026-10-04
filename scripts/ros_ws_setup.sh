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

# Pinned third-party revisions (bump deliberately after reviewing upstream changes).
# humble: the humble branch head (80e2ff9) ships libsgcore built on Ubuntu 24.04
# (GLIBC 2.38, GLIBCXX 3.4.32) and a state broadcaster using get_optional (not in
# ros2_control 2.54): it does not link or build on 22.04 (arm4090, 10.04). humble-dev
# a14a468 ("hardware interface: humble port fixes", SGBackend v2.306) needs at most
# GLIBC 2.34 / GLIBCXX 3.4.29 and uses get_value; same launch arguments and install paths.
declare -A SENSEGLOVE_PIN=(
  [humble]=a14a4683cf60b7296c863ec69c6e3f8e5c998e99
  [jazzy]=30c962452f7fbea4dd3c91e897f1a28571b2e15a
)
declare -A SENSEGLOVE_BRANCH=([humble]=humble-dev [jazzy]=jazzy)
PIN="${SENSEGLOVE_PIN[$DISTRO]:-}"
BRANCH="${SENSEGLOVE_BRANCH[$DISTRO]:-$DISTRO}"
[[ -n "$PIN" ]] || { echo "error: no senseglove_ros pin for $DISTRO" >&2; exit 1; }
EXT="$WS/external/senseglove_ros"
if [[ ! -d "$EXT/.git" ]]; then
  mkdir -p "$WS/external"
  git clone --quiet --filter=blob:none --sparse --no-checkout \
    https://github.com/Adjuvo/senseglove_ros.git "$EXT"
fi
git -C "$EXT" fetch --quiet origin "$BRANCH"
OLD_PIN="$(git -C "$EXT" rev-parse HEAD 2>/dev/null || true)"
git -C "$EXT" checkout --quiet --detach "$PIN"
if [[ -n "$OLD_PIN" && "$OLD_PIN" != "$(git -C "$EXT" rev-parse HEAD)" ]]; then
  # a new pin: drop the old senseglove build so nothing links against the previous libraries
  echo "senseglove_ros pin changed ($(git -C "$EXT" rev-parse --short "$OLD_PIN") -> $(git -C "$EXT" rev-parse --short HEAD)): clean rebuild"
  rm -rf "$WS"/build/senseglove_* "$WS"/install/senseglove_*
fi
if [[ "$FULL" -eq 1 ]]; then
  git -C "$EXT" sparse-checkout disable
else
  git -C "$EXT" sparse-checkout set senseglove/senseglove_msgs
fi
echo "senseglove_ros $DISTRO @ $(git -C "$EXT" rev-parse --short HEAD) (pinned) ($([[ $FULL -eq 1 ]] && echo full || echo msgs only))"

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
if [[ "$FULL" -eq 1 ]] && command -v rosdep >/dev/null; then
  # senseglove_ros needs ros2_control, ros2_controllers, xacro (upstream README);
  # report what is missing instead of installing it (apt needs sudo).
  # ament_python is a build type some package.xml files list as a key; rosdep has no rule for it.
  rosdep check --from-paths external/senseglove_ros --ignore-src --rosdistro "$DISTRO" --skip-keys ament_python \
    || echo "warning: missing system deps above (sudo apt install ...); the build may fail" >&2
fi
# A package built as a copy keeps a site-packages directory that shadows the
# symlinked sources: nodes would run stale code after every git pull
# (arm4090 10.03). Drop such a copy before building.
for pkg_dir in "$WS"/src/*/; do
  pkg="$(basename "$pkg_dir")"
  for copied in "$WS"/install/"$pkg"/lib/python3*/site-packages/"$pkg"; do
    if [[ -d "$copied" && ! -L "$copied" ]]; then
      echo "removing the copied (non-symlink) install of $pkg: $copied"
      rm -rf "$WS/build/$pkg" "$WS/install/$pkg"
    fi
  done
done
# setuptools >= 80 (a pip --user install on arm4090: 84) dropped `develop`, so
# colcon silently copies ament_python packages even with --symlink-install.
# Build against the system setuptools (no user site) to keep real symlinks.
PYTHONNOUSERSITE=1 colcon build --symlink-install --base-paths src external
echo "done: source $WS/install/setup.bash"
