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
