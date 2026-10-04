# Stored arm paths (from sim2real)

`home_rh56f1_right.npz`, `home_rh56f1_left.npz`: rest (all joints 0, "차렷") to
the `rh56f1_aglt` home of `configs/robots/openarm_rh56f1.yaml`, copied unchanged
from sim2real `deploy/policy_control/paths/` at commit 3bc3deb (2026-10-01,
"RH56F1 차렷 손 = 주먹 · 홈 경로 0.3 rad/s 재계획").

| file | sha256 | steps | step | clearance | max speed |
|---|---|---|---|---|---|
| home_rh56f1_right.npz | 6a224a0ba158ac4c3ac31e18ac5a7857d7cd9aed97bf4307bd21267c7de93acf | 711 | 0.02 s | 20.2 mm | 0.30 rad/s |
| home_rh56f1_left.npz | deecf9b4a3df037944a227e9d6e932306d88f73f4d0b7e215680a5f43401f105 | 730 | 0.02 s | 20.1 mm | 0.30 rad/s |

Planned with RRT against the arm4090 table, the back box and the side walls,
with the RH56F1 closed as a fist (`meta_hand_q`). sim2real allows only these
paths between rest and home, never an unverified straight joint line.
motion_acq plays them forward at start (from rest) and backwards at the end
(to rest, then motors off). If sim2real replans them (home or table change),
copy the new files here; `tests/test_home_path.py` compares them with the
sim2real checkout when it is present.
