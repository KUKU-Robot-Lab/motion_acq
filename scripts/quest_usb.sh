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
#   scripts/quest_usb.sh wifi      # (cable in, once per Quest boot) the same link over Wi-Fi; then the
#                                  # cable may come out. wifi-off: back to the cable.
#
# Wi-Fi (10.12 user): adb over TCP 5555 to the headset's Wi-Fi address, remembered in
# ~/.config/macq/quest_wifi; every command then talks to that device (cable in or not).
# The headset and this PC must be on the same network. More latency / jitter than USB, a
# dropped link stops the arms (tracking lost), and the headset runs on its battery.
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

WIFI_FILE="${MACQ_QUEST_WIFI_FILE:-$HOME/.config/macq/quest_wifi}"
WIFI_PORT=5555

devices() { "$ADB" devices | awk 'NR > 1 && NF >= 2 {print $1, $2}'; }

pick_serial() {
  # the remembered Wi-Fi device (dialled again when adb lost it), else USB (as motion_acq.quest_view.link)
  local list saved
  list="$(devices)"
  saved="$(cat "$WIFI_FILE" 2>/dev/null || true)"
  if [[ -n "$saved" ]] && ! grep -qx "$saved device" <<< "$list"; then
    "$ADB" connect "$saved" >/dev/null 2>&1 || true
    list="$(devices)"
  fi
  if [[ -n "$saved" ]] && grep -qx "$saved device" <<< "$list"; then echo "$saved"; return; fi
  awk '$2 == "device" && $1 !~ /:/ {print $1; exit}' <<< "$list" | grep . && return
  awk '$2 == "device" {print $1; exit}' <<< "$list" | grep . && return
  awk 'NR == 1 {print $1}' <<< "$list"
}

one_device() {
  local serial state saved
  serial="$(pick_serial)"
  saved="$(cat "$WIFI_FILE" 2>/dev/null || true)"
  [[ -n "$serial" ]] || die "no Quest on USB${saved:+ or Wi-Fi ($saved)}. Plug it in and accept 'Allow USB debugging' in the headset${saved:+ (after a Quest reboot: cable in, then $0 wifi again)}."
  state="$(devices | awk -v s="$serial" '$1 == s {print $2}')"
  case "$state" in
    device) ;;
    unauthorized) die "Quest is unauthorized: accept 'Allow USB debugging' inside the headset." ;;
    no*) die "no USB permission for the Quest: a udev rule for vendor 2833 (plugdev) is needed." ;;
    *) die "unexpected adb state: $state ($serial)" ;;
  esac
  export ANDROID_SERIAL="$serial"
}

cmd_wifi() {
  local usb ip target had_reverse=0 had_forward=0 i
  usb="$(devices | awk '$2 == "device" && $1 !~ /:/ {print $1; exit}')"
  [[ -n "$usb" ]] || die "the switch to Wi-Fi needs the cable once: plug the Quest in (USB debugging allowed), then $0 wifi"
  ip="$("$ADB" -s "$usb" shell ip -f inet addr show wlan0 | awk '/inet /{sub(/\/.*/, "", $2); print $2; exit}')"
  [[ -n "$ip" ]] || die "the Quest has no Wi-Fi address: join the same Wi-Fi as this PC in the headset"
  target="$ip:$WIFI_PORT"
  # 10.12: the Quest on another router with the same 192.168.1.x range looks reachable but is not.
  # Stop before 'tcpip', which restarts adbd and drops the cable's page link.
  local quest_ssid pc_ssids
  quest_ssid="$("$ADB" -s "$usb" shell cmd wifi status 2>/dev/null | sed -n 's/^Wifi is connected to "\(.*\)"\r\{0,1\}$/\1/p' | head -1)"
  pc_ssids="$("${NMCLI:-nmcli}" -t -f ACTIVE,SSID dev wifi 2>/dev/null | awk -F: '$1 == "yes" {print $2}' | sort -u)"
  if [[ -n "$quest_ssid" && -n "$pc_ssids" ]] && ! grep -qxF "$quest_ssid" <<< "$pc_ssids" \
     && ! "${PING:-ping}" -c 1 -W 1 "$ip" >/dev/null 2>&1; then
    die "the Quest is on Wi-Fi '$quest_ssid' but this PC is on '$(tr '\n' ' ' <<< "$pc_ssids" | sed 's/ $//')' and cannot reach $ip: join the PC's Wi-Fi in the headset, then $0 wifi again (the cable link is unchanged)"
  fi
  # adbd restarts in TCP mode and drops the links of the cable: set them again on the Wi-Fi device
  "$ADB" -s "$usb" reverse --list | grep -q "tcp:$VIEW_PORT" && had_reverse=1
  "$ADB" forward --list | grep -q "^$usb tcp:$TCP_PORT" && had_forward=1
  "$ADB" -s "$usb" tcpip "$WIFI_PORT" >/dev/null
  for i in 1 2 3 4 5 6 7 8; do
    sleep 1
    "$ADB" connect "$target" 2>&1 | grep -q "connected to" && break
  done
  devices | grep -qx "$target device" || die "could not reach $target over Wi-Fi (same network as this PC? $(ip -4 -o addr show scope global | awk '{print $4}' | tr '\n' ' '))"
  mkdir -p "$(dirname "$WIFI_FILE")"
  echo "$target" > "$WIFI_FILE"
  export ANDROID_SERIAL="$target"
  if (( had_reverse )); then "$ADB" reverse "tcp:$VIEW_PORT" "tcp:$VIEW_PORT" >/dev/null; echo "page link (reverse $VIEW_PORT) moved to Wi-Fi"; fi
  if (( had_forward )); then "$ADB" forward "tcp:$TCP_PORT" "tcp:$TCP_PORT" >/dev/null; echo "app link (forward $TCP_PORT) moved to Wi-Fi"; fi
  echo "Wi-Fi link: $target. The cable may come out now; the headset page reconnects by itself."
  echo "After a Quest reboot: cable in, then $0 wifi again."
}

cmd_wifi_off() {
  local saved usb
  saved="$(cat "$WIFI_FILE" 2>/dev/null || true)"
  rm -f "$WIFI_FILE"
  [[ -n "$saved" ]] && "$ADB" disconnect "$saved" >/dev/null 2>&1 || true
  usb="$(devices | awk '$2 == "device" && $1 !~ /:/ {print $1; exit}')"
  if [[ -n "$usb" ]]; then
    "$ADB" -s "$usb" usb >/dev/null || true  # adbd back to USB only (restarts: reopen the page)
    echo "back to the cable ($usb): reopen the page ($0 page) or the app ($0 app)"
  else
    echo "Wi-Fi link forgotten; plug the cable in"
  fi
}

cmd_status() {
  "$ADB" devices -l
  one_device
  echo "link: $ANDROID_SERIAL ($([[ "$ANDROID_SERIAL" == *:* ]] && echo Wi-Fi || echo USB))"
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
  wifi) cmd_wifi ;;
  wifi-off) cmd_wifi_off ;;
  *) die "usage: $0 {status|install|forward|launch|view|page|vr|app|unworn|worn|wifi|wifi-off}" ;;
esac
