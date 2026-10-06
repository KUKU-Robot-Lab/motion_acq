#!/usr/bin/env bash
# Wired Meta Quest link: tunnel the HandUMI Quest App pose stream over USB.
#
#   scripts/quest_usb.sh status    # adb device, app installed, forward active
#   scripts/quest_usb.sh install   # install the pinned APK (first time per Quest)
#   scripts/quest_usb.sh forward   # adb forward tcp:65432 -> Quest 65432
#   scripts/quest_usb.sh launch    # (re)start the app in the headset over adb
#   scripts/quest_usb.sh view      # head camera page instead of the app (macq quest-view runs), enters VR
#   scripts/quest_usb.sh page      # the same page, without entering VR (the console's [연결]; vr = [시작])
#   scripts/quest_usb.sh vr        # (re)enter VR on the open page from the PC (an asleep headset is woken
#                                  # with the proximity sensor off first); vr --status = page state
#   scripts/quest_usb.sh app       # back to the HandUMI app (forward + launch)
#   scripts/quest_usb.sh unworn    # keep the headset awake off the head (proximity sensor off), wake it
#   scripts/quest_usb.sh worn      # proximity sensor back to normal (also after a Quest reboot)
#
# Not worn (10.05 user: arms only, controllers on the hands): the controllers are still
# tracked by the headset cameras, so the headset stays on, awake (unworn) and placed so
# it sees the hands, its front toward the robot's front (Space takes the robot +x from
# the headset heading).
#
# With the forward active, the station rig uses quest_ip 127.0.0.1. Only TCP is
# tunnelled; the UDP time-sync (42000) is not, so frames are stamped with the PC
# receive time (clock_synced=false). USB latency keeps that error small.
# Do not run the mock Quest sender at the same time: both use local port 65432.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ADB="${ADB:-$(command -v adb || echo "$HOME/opt/platform-tools/adb")}"
APK="${QUEST_APK:-$HOME/opt/handumi-quest-app/handumi-quest-app-v0.2.1.apk}"
PACKAGE="com.handumi.questapp"
ACTIVITY="com.unity3d.player.UnityPlayerActivity"
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

cmd_launch() {
  # Closing a Quest system panel can close the app too; this brings it back.
  one_device
  "$ADB" shell am start -n "$PACKAGE/$ACTIVITY" >/dev/null
  sleep 3
  if "$ADB" shell pidof "$PACKAGE" >/dev/null; then
    echo "app running; put the headset on (it pauses while asleep)"
  else
    die "app did not start; open it in the headset (Library -> Unknown Sources)"
  fi
}

VIEW_PORT="${QUEST_VIEW_PORT:-8787}"

cmd_page() {
  # The page reaches macq quest-view as http://localhost (adb reverse: a secure
  # context for WebXR, no certificate); quest-view itself serves port 65432.
  one_device
  "$ADB" forward --remove "tcp:$TCP_PORT" 2>/dev/null || true
  "$ADB" reverse "tcp:$VIEW_PORT" "tcp:$VIEW_PORT"
  "$ADB" shell am force-stop "$PACKAGE" || true
  "$ADB" shell am start -a android.intent.action.VIEW -d "http://localhost:$VIEW_PORT" com.oculus.browser >/dev/null
  echo "Quest Browser opened http://localhost:$VIEW_PORT"
}

cmd_view() {
  cmd_page
  echo "entering VR from here ..."
  cmd_vr || echo "VR did not start from the PC; press 'VR 시작' in the headset (first time: allow VR)."
}

cmd_vr() {
  # WebXR needs a user gesture in the page; the browser devtools give it to the PC.
  one_device
  if [[ "${1:-}" != "--status" ]] && "$ADB" shell dumpsys power | grep -q "mWakefulness=Asleep"; then
    # 10.06 user: the headset is used off the head as well: wake it with the proximity sensor off
    echo "the headset is asleep (not worn): proximity sensor off and wake it ..."
    cmd_unworn
  fi
  "$ROOT/.venv/bin/python" -m motion_acq.quest_view.devtools --view-port "$VIEW_PORT" "$@"
}

cmd_unworn() {
  one_device
  "$ADB" shell am broadcast -a com.oculus.vrpowermanager.prox_close >/dev/null
  "$ADB" shell input keyevent KEYCODE_WAKEUP
  sleep 1
  if "$ADB" shell dumpsys power | grep -q "mWakefulness=Awake"; then
    echo "proximity sensor off: the headset stays awake off the head (undo: $0 worn, or a Quest reboot)"
  else
    die "the headset did not wake: press its power button once, then run $0 unworn again"
  fi
}

cmd_worn() {
  one_device
  "$ADB" shell am broadcast -a com.oculus.vrpowermanager.automation_disable >/dev/null
  echo "proximity sensor back to normal: the headset sleeps when taken off"
}

cmd_app() {
  one_device
  "$ADB" reverse --remove "tcp:$VIEW_PORT" 2>/dev/null || true
  cmd_forward
  cmd_launch
}

case "${1:-status}" in
  status) cmd_status ;;
  install) cmd_install ;;
  forward) cmd_forward ;;
  launch) cmd_launch ;;
  view) cmd_view ;;
  page) cmd_page ;;
  vr) shift; cmd_vr "$@" ;;
  app) cmd_app ;;
  unworn) cmd_unworn ;;
  worn) cmd_worn ;;
  *) die "usage: $0 {status|install|forward|launch|view|page|vr|app|unworn|worn}" ;;
esac
