# Upstream

- Source: https://github.com/murobotics-ai/handumi-sw
- Commit: 8f7264b418f84813264ff3cab14d267c7e821616 (2026-09-17)
- License: Apache-2.0 (LICENSE copied unchanged)

## Changes from upstream

- Package renamed `handumi` -> `motion_acq`; CLI `handumi`/`hu` -> `macq`.
- Kept only the modules reachable from: doctor, tracking pose, calibrate tcp, teleop (sim),
  teleop-real, teleop-record. Piper backend, inpainting, dataset curation/conversion/replay removed.
- Assets: openarm, scenes. Robot configs: openarmv1.
