"""CPU placement shared with the sim2real stack on the same PC.

Port of sim2real deploy/policy_control/policy_control/cpu_plan.py (10.03):
the RH56F1 EtherCAT masters (right, left) each get one physical core (an
isolated one first, else the highest numbered, never the core holding cpu0)
and run SCHED_FIFO 80; everything else stays on the remaining "general"
cores. motion_acq computes the same plan from sysfs, so its processes never
land on the master cores or their SMT siblings, whether or not the masters
run. tests/test_cpu.py checks parity with sim2real when it is checked out.

PCs with fewer than (roles + 4) physical cores are left unpinned, as in
sim2real. MACQ_CPU_PIN=0 (or sim2real's S2R_CPU_PIN=0) turns pinning off.

The arm streamer thread asks for SCHED_FIFO 50 like the s2r OpenArm
controller_manager; that needs the RT limit opened by sim2real
scripts/setup/rt_setup.sh (or the identical scripts/rt_setup.sh here); without it the thread runs normally.

    python -m motion_acq.cpu              # this PC's plan, one line
    python -m motion_acq.cpu --general    # general cpulist ("0-13,16-29"), empty if unpinned
"""

from __future__ import annotations

import logging
import os
import resource
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

SYSFS = Path("/sys/devices/system/cpu")
RT_ROLES = ("ecat_right", "ecat_left")  # sim2real RH56F1 EtherCAT masters
MIN_GENERAL_CORES = 4
PIN_ENVS = ("MACQ_CPU_PIN", "S2R_CPU_PIN")
ARM_STREAMER_FIFO = 50  # same as the s2r OpenArm controller_manager RT thread


def parse_cpulist(text: str) -> tuple[int, ...]:
    """Kernel cpulist "0-3,8,10-11" -> (0, 1, 2, 3, 8, 10, 11)."""
    out: list[int] = []
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi or lo) + 1))
    return tuple(sorted(set(out)))


def format_cpulist(cpus: Iterable[int]) -> str:
    """(0, 1, 2, 3, 8, 10, 11) -> "0-3,8,10-11" (taskset / docker format)."""
    out: list[str] = []
    run: list[int] = []
    for c in sorted(set(int(c) for c in cpus)):
        if run and c == run[-1] + 1:
            run.append(c)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [c]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


@dataclass(frozen=True)
class Topology:
    cores: tuple[tuple[int, ...], ...]  # online logical CPUs per physical core, by first CPU
    isolated: frozenset[int] = frozenset()


def read_topology(sysfs: Path = SYSFS) -> Topology:
    online = parse_cpulist(_read(sysfs / "online") or "0")
    groups: dict[tuple[int, ...], list[int]] = {}
    for cpu in online:
        siblings = parse_cpulist(_read(sysfs / f"cpu{cpu}" / "topology" / "thread_siblings_list") or str(cpu))
        groups.setdefault(tuple(c for c in siblings if c in online), []).append(cpu)
    cores = tuple(sorted(tuple(sorted(c)) for c in groups.values()))
    return Topology(cores=cores, isolated=frozenset(parse_cpulist(_read(sysfs / "isolated"))))


@dataclass(frozen=True)
class CpuPlan:
    general: tuple[int, ...]
    rt: dict[str, int] = field(default_factory=dict)  # role -> logical CPU (empty: unpinned)
    reserved: tuple[int, ...] = ()
    note: str = ""


def make_plan(topo: Topology, roles: Iterable[str] = RT_ROLES, env: Mapping[str, str] | None = None) -> CpuPlan:
    env = os.environ if env is None else env
    roles = tuple(roles)
    cores = topo.cores
    usable = tuple(c for c in cores if not set(c) <= topo.isolated)
    head = f"{len(cores)} physical / {sum(len(c) for c in cores)} logical"

    def unpinned(why: str) -> CpuPlan:
        return CpuPlan(general=tuple(sorted(c for core in usable for c in core)), note=f"{head}; {why}")

    if any(env.get(name, "1") == "0" for name in PIN_ENVS):
        return unpinned("pinning off by env")
    iso = [c for c in cores if set(c) <= topo.isolated and 0 not in c]
    rest = [c for c in cores if c not in iso and 0 not in c]
    candidates = iso[::-1] + rest[::-1]
    if len(cores) < len(roles) + MIN_GENERAL_CORES or len(candidates) < len(roles):
        return unpinned(f"fewer than {len(roles) + MIN_GENERAL_CORES} physical cores, unpinned")
    picked = candidates[: len(roles)]
    general_cores = [c for c in usable if c not in picked]
    return CpuPlan(
        general=tuple(sorted(c for core in general_cores for c in core)),
        rt={role: core[0] for role, core in zip(roles, picked, strict=True)},
        reserved=tuple(sorted(c for core in picked for c in core)),
        note=head,
    )


def current_plan(sysfs: Path = SYSFS) -> CpuPlan:
    """Never raises: an unreadable sysfs means "unpinned" (pinning is an optimisation)."""
    try:
        return make_plan(read_topology(sysfs))
    except Exception as exc:  # noqa: BLE001 - parse or permission problems alike
        return CpuPlan(general=tuple(sorted(os.sched_getaffinity(0))), note=f"cannot read cores ({exc}); unpinned")


def keep_off_rt(plan: CpuPlan | None = None) -> str:
    """Pin this process (all its threads; children inherit) to the general cores.

    Call first thing in a process main. Returns one log line; never raises.
    """
    plan = plan or current_plan()
    if not plan.rt:
        return f"CPU: {plan.note}"
    mask = set(plan.general)
    try:
        tids = [int(t) for t in os.listdir("/proc/self/task")]
    except OSError:
        tids = [0]
    try:
        for tid in tids:
            os.sched_setaffinity(tid, mask)
    except OSError as exc:
        return f"CPU: could not pin to the general cores ({exc.strerror or exc})"
    return (f"CPU: general cores {format_cpulist(plan.general)} "
            f"(RH56F1 EtherCAT cores {format_cpulist(plan.reserved)} kept free)")


def rt_limit() -> int:
    """Highest SCHED_FIFO priority this process may take (RLIMIT_RTPRIO soft); 0 = none."""
    soft = resource.getrlimit(resource.RLIMIT_RTPRIO)[0]
    return 99 if soft == resource.RLIM_INFINITY else int(soft)


def request_fifo(priority: int) -> str:
    """Give the calling thread SCHED_FIFO priority. Returns one log line; never raises."""
    if priority <= 0:
        return "RT: off"
    limit = rt_limit()
    if limit < priority:
        return (f"RT: SCHED_FIFO {priority} not allowed (RLIMIT_RTPRIO {limit}); running normally "
                "(once: sudo bash scripts/rt_setup.sh, then reboot)")
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
    except OSError as exc:
        return f"RT: SCHED_FIFO {priority} failed ({exc.strerror or exc}); running normally"
    return f"RT: SCHED_FIFO {priority}"


def governors(sysfs: Path = SYSFS) -> tuple[str, ...]:
    return tuple(sorted({g.read_text().strip() for g in sysfs.glob("cpu[0-9]*/cpufreq/scaling_governor")
                         if g.is_file()}))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    plan = current_plan()
    if "--general" in args:
        print(format_cpulist(plan.general) if plan.rt else "")
        return 0
    print(f"{plan.note}; general {format_cpulist(plan.general)}; reserved {format_cpulist(plan.reserved) or '-'}; "
          f"rt {plan.rt or '-'}; RLIMIT_RTPRIO {rt_limit()}; governor {','.join(governors()) or '?'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
