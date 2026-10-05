# Patches to third-party checkouts

`scripts/ros_ws_setup.sh --full` applies these to `ros_ws/external/` after
checking out the pinned revision (and skips a patch that is already in).

| patch | onto | why |
|---|---|---|
| `senseglove_ros_haptics_levels.patch` | senseglove_ros humble-dev a14a468 | Haptic commands are percent (0..100, the joints' `maxPosition`) but upstream passed them unscaled to SGCore, which takes 0..1, and dropped any command whose sum was under 10. A brake was either full or nothing, and the 0..1 levels a caller would send were never sent at all. The patch scales percent to 0..1, lowers the thresholds to 0.5 %, and gives the Nova 2 brakes only the brake channels (the strap goes to the squeeze). Together with `motion_acq_hand/config/nova2_<side>_controllers.yaml` (forward_command_controller instead of a JointTrajectoryController PID) the hand node can grade the RH56F1 contact feedback (10.05). Review fixes: a channel that drops to 0 while the other stays on is flushed once with zeros (a vibration pulse could otherwise keep buzzing under a held brake); a command unchanged for 1 s while any level is on is released (`forward_command_controller` holds the last command forever: a dead hand node would leave the brakes and strap on; the hand node re-sends every 0.25 s with an alternating last digit); the strap squeeze is capped at 0.5 in the driver too. |
