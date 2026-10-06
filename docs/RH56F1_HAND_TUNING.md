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

## thumb_1 (rotation)
Force reads negative when the thumb is loaded towards the outside (-475..-750 g at the outer end
against a held cup); a contact towards the palm at ~733 raised it only ~90 g and the current to 94 mA
while a hand-held cup gave way. Stiffness not measured (needs a rigid fixture); user 10.06: leave thumb_1
for now.

## For the controller and the simulator
- Position following: mode 0. Grip force: not from a position margin (object dependent); either a force
  loop in software around mode 0, or firmware mode 1 while holding (accurate, low current, but it ignores
  the angle, so a switch back to mode 0 on release is needed).
- Simulator: position PD for free motion; in contact a force-limited hold (mode 1 like) at the grip force,
  not a stiff spring to 1.7 kg.
