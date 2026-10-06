# RH56F1 hand tuning (left hand, 2026-10-06, arm4090, EtherCAT 500 Hz)

Measured with `scripts/hand_contact_probe.py` (logs: arm4090 `logs/hand/contact_probe_left_*_20261006_*.jsonl`).
One joint moves, the others hold. Object: a cup (index, thumb_2 held / taped; middle, ring, pinky with
the cup deep in the palm). Registers: fingers 1740 open .. 900 closed (~550 reg/rad), thumb_2 1350 .. 1100
(~527 reg/rad). Firmware current limit 0x2000:07~0C = 800 mA (write + read back at driver start).

Why: 10.06 cup grasp in teleop froze the left hand (force 1715 g, 6.5 A in total).

## Firmware finger modes (SDO 0x2000:1B~20)

| mode | free motion | in contact | force_set honoured | current in contact | use |
|---|---|---|---|---|---|
| 0 speed/force protect (default) | follows angle exactly | stiff position servo, pushes to 1.1-1.85 kg (saturation) | **no** (100: stops in free space; 300-1000: ~1.7 kg) | 1.1-1.5 A | position following |
| 1 force closed loop | **ignores angle**: closes until force = force_set, opens when above (force_set 0 opens past 1740, to 1813) | **holds force_set**: fingers 600 -> 594-602 g, 300 -> 299-315 g; thumb_2 ~0.65 x (600 -> ~395 g) | **yes** | 0-250 mA | holding a grasp |
| 2 impedance | slow (2 s and still ~50 reg short), speed ignored, did not move at all when started beyond the range (1815) | softer at first (thumb_2 10 reg -> 100 g) but still 1.2-1.4 kg at full squeeze; tremor seen | no | 0.5-1.3 A | not used |

## Mode 0: steady force vs how far the target is past the contact

Contact = first reading 120 g over rest at speed 300 (dynamic: the true hard contact can be deeper).

| joint | rest (g) | +0 | +10 | +20 | +40 | +80 / full | note |
|---|---|---|---|---|---|---|---|
| index | 91 | 205 | 402 | 711 | 1194 | 1611-1820 | ~30 g/reg, cup on the fingertip |
| thumb_2 | 2 | 93 | 556 | 1111 | 1397 | 1400-1700 | ~55 g/reg (5 reg -> 272 g) |
| middle | -12 | 78 | 122 | 185 | 533 | 1362-1520 | cup deep in the palm: soft first, then steep |
| ring | -17 | 75 | 110 | 137 | 200 | 1046-1530 | idem |
| pinky | -4 | 84 | 81 | 108 | 153 | 1102-1124 | idem |

So the force from a position offset depends on the object and where it touches: a fixed margin past
the contact cannot set a grip force. Speed (150-2000) changes neither the peak nor the held force in
mode 0, unlike the RH56DFX paper (arXiv 2603.08988). After a mode 0 squeeze the current drops to 0
within ~0.6 s and the force stays (the lead screw holds it) until the finger is told to open.

## Right hand (10.06 14:12, after force zero calibration)

Force sensor zero calibration (`rh56f1_driver.py --force-calibrate`, SDO 0x2000:06, empty hand): rest
readings -17, 13, 23, 17, 27, -70 g -> 0, 0, 0, 0, -5, -1 g (pinky .. thumb_1). Mode 1, cup deep in the palm:

| force_set | index | middle | ring | pinky | thumb_2 |
|---|---|---|---|---|---|
| 600 | 581-616 | 601-603 | 593-617 | 587-613 | 375-390 |
| 300 | 297-302 | 294-298 | 289-299 | 307-312 | 197-203 |
| 100 | 105 (free) | 102 (free) | 101 (free) | 105 (free) | 74 (free) |

Same as the left hand: fingers hold force_set within a few %, thumb_2 reads ~0.64 x force_set.

## thumb_1 (rotation)
Force reads negative when the thumb is loaded towards the outside (-475..-750 g at the outer end
against a held cup); a contact towards the palm at ~733 raised it only ~90 g and the current to 94 mA
while a hand-held cup gave way. Stiffness not measured (needs a rigid fixture); user 10.06: leave thumb_1
for now.

## Runtime mode switch: refused (10.06 15:00, right index)
Writing FINGER_MODE over SDO while the hand is in OP fails ("mailbox protocol not supported", 3.5-7.7 s
per request, the mode stays 0); at PREOP (driver start) it works. So no firmware mode switching while
teleoperating. The process data loop was not disturbed (separate SDO thread, no WKC warnings).

## Controller chosen: position-based admittance in mode 0 (motion_acq hand/admittance.py)
User 10.06: position and force together, not switched; linkage with a non-backdrivable lead screw, so
admittance, not impedance. Per finger, every tick: q_cmd = q_operator - y, y low-passes
f / 2000 g/rad (+ (f - 800) / 200 above 800 g), tau 0.5 s in contact / 0.15 s once free; in contact the
command leads the finger by <= 0.02 x (1 - f/800) rad. Simulated (25-50 ms delay, 1.6 rad/s finger,
contact 1.6-29 kg/rad): grip settles at stiffness x penetration (284 / 512 / 599 g for 0.3 rad), impact
peak ~1 kg at 25 ms delay; at 50 ms delay the stiffest contact (29 kg/rad) limit-cycles: thumb_2 to be
checked on the real hand.

## 10.06 evening: the lead cap capped the grip; replaced by a closing rate
Right index, rigid cup, driver admittance at 500 Hz (robot_control components/rh56f1.yaml): 0.4 rad past the
contact held 353 / 362 g instead of ~800 g, finger still at the cup, 245-272 mA. The lead cap (command <= 11
x (1 - f/800) registers past the measured finger) with ~55 g per register of command past a rigid object
gives f = 605 (1 - f/800) -> ~345 g, whatever the penetration. The same cap made fast free-space closing
(4 rad/s target) move at ~0.45 rad/s: the motion alone reads 50-60 g over rest, above the 40 g deadband.
Now: once the force is 60 g over the deadband the command restarts from the measured angle and closes at
<= 0.3 rad/s x (1 - f/800), opening at once, until the force and the offset are gone (else the offset
releasing in 0.15 s re-closes the command into the object: relaxation cycle in simulation). Simulated with
that plant (55 and 10 g/register, 24-50 ms delay): no cycle, 0.4 rad -> 750-840 g, 0.5 rad/s approach peak
< 900 g. A fast approach (4 rad/s) into a rigid object is not limited before the force is read: real peak to
be measured (the guard's 800 mA current limit stays).

## For the controller and the simulator
- Position following: mode 0. Grip force: not from a position margin (object dependent); either a force
  loop in software around mode 0, or firmware mode 1 while holding (accurate, low current, but it ignores
  the angle, so a switch back to mode 0 on release is needed).
- Simulator: position PD for free motion; in contact a force-limited hold (mode 1 like) at the grip force,
  not a stiff spring to 1.7 kg.
