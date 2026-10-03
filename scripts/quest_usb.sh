#!/usr/bin/env bash
# Wired Meta Quest link: tunnel the HandUMI Quest App pose stream over USB.
#
#   scripts/quest_usb.sh status    # adb device, app installed, forward active
#   scripts/quest_usb.sh install   # install the pinned APK (first time per Quest)
#   scripts/quest_usb.sh forward   # adb forward tcp:65432 -> Quest 65432
#
# With the forward active, the station rig uses quest_ip 127.0.0.1. Only TCP is
# tunnelled; the UDP time-sync (42000) is not, so frames are stamped with the PC
# receive time (clock_synced=false). USB latency keeps that error small.
# Do not run the mock Quest sender at the same time: both use local port 65432.
set -euo pipefail

ADB="${ADB:-$(command -v adb || echo "$HOME/opt/platform-tools/adb")}"
APK="${QUEST_APK:-$HOME/opt/handumi-quest-app/handumi-quest-app-v0.2.1.apk}"
PACKAGE="com.handumi.questapp"
TCP_PORT="${QUEST_TCP_PORT:-65432}"

die() { echo "error: $*" >&2; exit 1; }

[[ -x "$ADB" ]] || die "adb not found (expected $HOME/opt/platform-tools/adb)."

one_device() {
  local states
  states="$("$ADB" devices | awk 'NR > 1 && NF >= 2 {print $2}')"
  [[ -n "$states" ]] || die "no Quest on USB. Plug it in and accept 'Allow USB debugging' in the headset."
  [[ "$(wc -l <<< "$states")" -eq 1 ]] || die "more than one adb device; unplug the others."
  case "$states" in
    device) ;;
    unauthorized) die "Quest is unauthorized: accept 'Allow USB debugging' inside the headset." ;;
    no*) die "no USB permission for the Quest: a udev rule for vendor 2833 (plugdev) is needed." ;;
    *) die "unexpected adb state: $states" ;;
  esac
}

cmd_status() {
  "$ADB" devices -l
  one_device
  if "$ADB" shell pm list packages "$PACKAGE" | grep -q "$PACKAGE"; then
    echo "app: $PACKAGE installed"
  else
    echo "app: $PACKAGE NOT installed (run: $0 install)"
  fi
  echo "forwards:"; "$ADB" forward --list
}

cmd_install() {
  one_device
  [[ -f "$APK" ]] || die "APK missing: $APK"
  "$ADB" install -r "$APK"
}

cmd_forward() {
  one_device
  "$ADB" forward "tcp:$TCP_PORT" "tcp:$TCP_PORT"
  echo "forward active: 127.0.0.1:$TCP_PORT -> Quest:$TCP_PORT"
  echo "Launch the app in the headset (Library -> Unknown Sources) and keep it in front."
}

case "${1:-status}" in
  status) cmd_status ;;
  install) cmd_install ;;
  forward) cmd_forward ;;
  *) die "usage: $0 {status|install|forward}" ;;
esac
