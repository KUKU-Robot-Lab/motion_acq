# motion_acq

Meta Quest -> OpenArm teleoperation and demonstration capture, run as two independent stations
(arm4090, arm5080). Derived from HandUMI (Apache-2.0); see UPSTREAM.md.

## Stations

Two independent stations, each with its own Meta Quest and OpenArm pair:

| station | PC | hands | rig file |
|---|---|---|---|
| arm4090 | RTX 4090 robot PC | RH56F1 on both arms | `configs/stations/arm4090.yaml` |
| arm5080 | RTX 5080 PC | to be connected | `configs/stations/arm5080.yaml` |

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

`quest_ip` stays empty until each station's Quest IP is known. Meanwhile the
pipeline runs against the bundled mock Quest (TCP 65432, UDP 42000 on localhost):

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
| arm5080 | can2 | can3 | true (VERIFY left/right once the OpenArm is connected) |
