# motion_acq

Meta Quest -> OpenArm teleoperation and demonstration capture, run as two independent stations
(arm4090, arm5080). Derived from HandUMI (Apache-2.0); see UPSTREAM.md.

## Stations

Two independent stations, each with its own Meta Quest and OpenArm pair:

| station | PC | robot YAML | hands | rig file |
|---|---|---|---|---|
| arm4090 | RTX 4090 robot PC | `openarm_rh56f1` | RH56F1 on both arms | `configs/stations/arm4090.yaml` |
| arm5080 | RTX 5080 PC | `openarmv1` (HandUMI default) | OpenArm gripper now, LEAP hand planned (Quest-only tests) | `configs/stations/arm5080.yaml` |

`openarm_rh56f1` (arm4090) is built from the hdgp asset
`openarm_rh56f1_bi_rl` by `scripts/build_openarm_rh56f1_urdf.py`: arm joints
renamed for the OpenArm driver, hands fixed open (thumb_1 1.57 rad), head
fixed, TCP = palm sensor frame (right (0.08, -0.30, 0.45) m at home), home and
IK posture = sim2real `rh56f1_aglt`. Every link matches the hdgp model within
1e-9 m; `--check` (also a test) fails if the hdgp asset changes. Meshes are
referenced from `~/rl_ws/urdf` like the hdgp asset; IK does not load them.
`--robot` defaults to the station's `recording.robot`.

Select the station per shell: `export MACQ_STATION=arm4090`. Run commands from
the repository root. STEP 1 drives only J1..J7; the J8 gripper motor is disabled
by `robots.openarmv1.gripper.enabled: false` in each station file.

## Install

```bash
echo 3.14 > .python-version   # any Python >= 3.12
CMAKE_PREFIX_PATH=$HOME/opt/cli11 CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$HOME/opt/cli11" \
  uv sync --extra openarm --extra sim
uv run --group dev pytest -q tests
```

`openarm_can` needs CLI11 at build time (`libcli11-dev` or a local install).

## Check without a Quest (mock sender)

The pipeline also runs against the bundled mock Quest (TCP 65432, UDP 42000 on
localhost). Stop any `adb forward` first; both use local port 65432:

```bash
export MACQ_STATION=arm4090
.venv/bin/python -m motion_acq.tracking.mock_quest_sender &      # terminal 1
macq tracking pose --device meta --quest-ip 127.0.0.1             # terminal 2
macq teleop --device meta --robot openarmv1 --quest-ip 127.0.0.1 \
  --skip-feetech --skip-cameras --auto-start --no-browser         # sim only
```

`--skip-feetech` is needed: the HandUMI handheld Feetech grippers are not used.
Without them the double-squeeze start gesture is unavailable, so start with
`--auto-start` (sim) or `--space-start` (real).

## Code flow

The main development checkout is the local 5090 PC. Commit and push there to
`kuku` (https://github.com/KUKU-Robot-Lab/motion_acq, the canonical repo);
`origin` is a personal mirror. The stations only pull:

```bash
cd ~/rl_ws/motion_acq && git pull   # main tracks kuku/main
```

## CAN per station

| station | right arm | left arm | auto_repair |
|---|---|---|---|
| arm4090 | can0 | can1 | false: s2r owns the links; motion_acq only validates them |
| arm5080 | can0 | can1 | true (one PCAN-USB Pro FD; VERIFY wiring with show_param) |

## Meta Quest over USB (default)

Each station's Quest is wired to its PC. The HandUMI Quest App listens on
TCP 65432 on all interfaces, so `adb forward` tunnels the pose stream and the
rig uses `quest_ip: 127.0.0.1`. adb (Google platform-tools) and the pinned APK
(v0.2.1, sha256 checked) live under `~/opt` on each station.

```bash
scripts/quest_usb.sh status    # device authorized? app installed? forwards
scripts/quest_usb.sh install   # first time per Quest
scripts/quest_usb.sh forward   # after every replug or adb restart
macq tracking pose --device meta
```

The UDP time-sync is not tunnelled, so frames carry the PC receive time
(`clock_synced=false`). Enable Developer Mode on the Quest first; do not start
Quest Link.

## STEP 2: head (Meta Quest HMD -> Dynamixel pan/tilt)

`macq head` runs as its own process, independent of the arms. It anchors after
the HMD has been tracked for 2 s, then maps yaw -> pan and pitch -> tilt
relative to that moment: deadband -> One Euro -> sign/scale -> window around
home (pan ±20°, tilt ±15° at first) -> 60°/s, 300°/s² limits. Yaw is unwrapped
continuously and held while |pitch| > 75°. HMD loss holds the head; a HOLD
longer than 0.5 s or a > 30° HMD jump (Quest recenter) re-anchors at the
current head pose instead of snapping. Startup refuses another process on the
port (flock + /proc check), a latched hardware error, a wrong motor model or a
head outside its window, and enables torque in place (goal = present); a
failure while torque is off re-enables it in place. A hardware-alert flag
stops at once; other faults stop after 10 faulty cycles in a row. SIGTERM runs
the same cleanup as Ctrl+C. Each run writes
`logs/head/head_<station>_<time>.jsonl`.

```bash
uv sync --extra head ...                 # dynamixel-sdk
macq head                                # fake bus (default): no serial port opened
macq head --axes pan                     # staged: tilt stays at its anchor
macq head --backend real                 # opens head.port; s2r must not hold it
```

Check without hardware: run the mock with a moving HMD and periodic loss,
`python -m motion_acq.tracking.mock_quest_sender --hmd-yaw-amp-deg 30
--hmd-pitch-amp-deg 10 --hmd-loss-every-s 10 --hmd-loss-s 1.5`, then
`macq head --duration-s 25`.

Only arm4090 has a `head:` section (values from sim2real
`config/head_home_rh56f1.yaml`). The pan/tilt `sign` entries are unverified
until the staged hardware test.

## STEP 3: hand (SenseGlove Nova 2 -> RH56F1), arm4090

Pure retargeting lives in `motion_acq.hand` (Python 3.10 safe); the ROS 2
package `ros_ws/src/motion_acq_hand` wraps it for the system interpreter:

| executable | role |
|---|---|
| `hand_node` | glove `/senseglove/glove<serial>/<rh\|lh>/senseglove_states` (best_effort) -> `/hand_<side>/angle_set` at 30 Hz. Rules in `motion_acq.hand.controller`: disabled until `/motion_acq/hand_<side>/enable` (Bool); enable needs a fresh, plausible `angle_actual` (no 0/-1/65535, within each axis range ±100) and subscribed driver topics; starts from the measured pose; `hand_id` taken from `angle_actual`; speed/force re-sent every 1 s; glove older than 0.2 s -> HOLD (nothing published); `angle_actual` lost > 0.5 s -> latched FAULT (disable + enable to clear); disable freezes the hand at its measured pose. Params `amplitude` (0-1 of the open->closed travel), `driver_speed`, `driver_force`. `enable_on_start` only on the fake domain. JSONL log + optional UDP sidecar |
| `calibrate` | records the open / fist / thumb_opposed poses -> `configs/hands/calibration/<user>_<side>.yaml` |
| `fake_glove`, `fake_rh56f1` | fake Nova 2 and fake RH56F1 (honours `hand_id` like the vendor driver); refuse to run unless `ROS_DOMAIN_ID=177` and `ROS_LOCALHOST_ONLY=1`, fixed (they use the real driver topic names) |

```bash
scripts/ros_ws_setup.sh             # senseglove_msgs (pinned humble/jazzy commit) + motion_acq_hand
scripts/ros_ws_setup.sh --full      # whole senseglove_ros for the real glove (SenseCom)
scripts/fake_hand_check.sh right 20 # fake calibration + fake chain with glove dropouts -> PASS/FAIL
```

Mapping: `configs/hands/nova2_to_rh56f1.yaml` (features, poses, RH56F1 joint
ends, filter, 2 rad/s limit, driver speed 2000 / force 600). RH56F1 rad ->
register: `configs/hands/rh56f1_hand_map.yaml`, a port of sim2real
`rh56f1_map.py` (numerically identical; parity test runs when sim2real is
checked out next to this repo; touch and mimic not ported). For the first real
runs use a small `amplitude` and lower `driver_speed`/`driver_force`. Unverified until the glove is worn: which
glove angle drives thumb opposition, and the opposed end of `thumb_1`.

## Recording (LeRobot): arms + head + hands in one dataset

`macq teleop-record` writes one LeRobot dataset. Head and hand processes stream
each control cycle over UDP to localhost; the recorder aligns them to every
frame by CLOCK_MONOTONIC (`motion_acq.sidecar`) and treats a stale stream like
a lost camera (sensor health gate). Streams per station come from
`recording.sidecars` (arm4090: head 47101, hand_right 47111, hand_left 47112);
`--no-sidecars` records the arms only (arm5080).

| feature | shape | content |
|---|---|---|
| `observation.state`, `action` | arm joints | HandUMI canonical arm layout |
| `observation.head.state`, `action.head` | 2 | pan, tilt (rad) measured / commanded |
| `observation.head.hmd_rel` | 2 | HMD yaw, pitch relative to the anchor (rad) |
| `observation.hand.<side>.state`, `action.hand.<side>` | 6 | RH56F1 `<r\|l>_hj_*` (rad) measured / commanded |
| `observation.glove.<side>.angles` | 20 | raw Nova 2 joints (rad) |
| `observation.{head,hand.<side>}.status` | 1 | -1 missing, 0 idle/disabled, 1 running, 2 hold, 3 fault |

Assistant keyboard (with `--space-start`, the operator's hands are busy):
Space start, Space save (>= 1 s after start), R reset the attempt and home,
Q finish the session, Esc stop. One reader owns the terminal (the upstream
Esc listener used to swallow Space presses).

Fake end to end (no hardware): `scripts/fake_hand_check.sh right` and `left`
once (fake glove calibrations), then `.venv/bin/python scripts/fake_record_check.py`
(mock Quest, fake head bus, fake hands on domain 177, `teleop-record
--fake-robot` with a simulated OpenArm SDK) -> PASS/FAIL on the dataset.

## Bringing a station up: `macq station`

```bash
MACQ_STATION=arm4090 macq station --fake -- --num-episodes 3          # all fake
MACQ_STATION=arm4090 macq station --real --user op1 --preflight-only  # read-only checks
MACQ_STATION=arm4090 macq station --real --user op1 -- --num-episodes 10
MACQ_STATION=arm5080 macq station --real -- --num-episodes 10          # arms only
```

Head and hands run in their own process groups (logs under `logs/station/`);
`teleop-record` runs in the foreground for the assistant's keys. `--real`
first checks, read-only: Quest USB forward, CAN up with FD 1M/5M and not held
by an s2r bringup, head port free, RH56F1 driver and glove topics on the
current ROS_DOMAIN_ID, hand calibration files. Any failure stops before
anything starts. Real hands start disabled and are enabled after Enter.
`scripts/fake_record_check.py` drives `macq station --fake` end to end.
