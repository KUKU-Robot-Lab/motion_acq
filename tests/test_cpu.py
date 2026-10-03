"""CPU plan shared with sim2real: same RT cores, motion_acq stays off them."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from motion_acq.cpu import (
    Topology,
    format_cpulist,
    keep_off_rt,
    make_plan,
    parse_cpulist,
    read_topology,
    request_fifo,
)

S2R_CPU_PLAN = Path.home() / "rl_ws/sim2real/deploy/policy_control/policy_control/cpu_plan.py"


def smt(physical: int) -> Topology:
    """Intel-style numbering: core i has CPUs i and i + physical (arm4090: 16 / 32)."""
    return Topology(cores=tuple((i, i + physical) for i in range(physical)))


def test_arm4090_plan_matches_the_measured_s2r_placement():
    plan = make_plan(smt(16), env={})
    assert plan.rt == {"ecat_right": 15, "ecat_left": 14}
    assert format_cpulist(plan.reserved) == "14-15,30-31"
    assert format_cpulist(plan.general) == "0-13,16-29"  # sim2real docs 10.03: FP++ on 0-13,16-29


def test_small_pcs_and_env_switch_stay_unpinned():
    assert make_plan(smt(5), env={}).rt == {}
    for name in ("MACQ_CPU_PIN", "S2R_CPU_PIN"):
        plan = make_plan(smt(16), env={name: "0"})
        assert plan.rt == {} and len(plan.general) == 32


def test_isolated_cores_are_taken_first_and_never_general():
    topo = Topology(cores=tuple((i,) for i in range(12)), isolated=frozenset({3, 4}))
    plan = make_plan(topo, env={})
    assert set(plan.rt.values()) == {3, 4}
    assert not {3, 4} & set(plan.general)


def test_cpulist_roundtrip_and_sysfs_reader(tmp_path):
    assert parse_cpulist("0-3,8,10-11") == (0, 1, 2, 3, 8, 10, 11)
    assert format_cpulist((0, 1, 2, 3, 8, 10, 11)) == "0-3,8,10-11"
    (tmp_path / "online").write_text("0-3\n")
    for cpu, sib in ((0, "0,2"), (1, "1,3"), (2, "0,2"), (3, "1,3")):
        d = tmp_path / f"cpu{cpu}" / "topology"
        d.mkdir(parents=True)
        (d / "thread_siblings_list").write_text(sib + "\n")
    assert read_topology(tmp_path).cores == ((0, 2), (1, 3))


def test_unpinned_plan_and_rt_requests_never_raise():
    assert keep_off_rt(make_plan(smt(5), env={})).startswith("CPU:")
    assert request_fifo(0) == "RT: off"
    assert request_fifo(50).startswith("RT:")  # granted or explained, never an exception


@pytest.mark.skipif(not S2R_CPU_PLAN.exists(), reason="sim2real not checked out next to this repo")
def test_parity_with_sim2real_cpu_plan():
    spec = importlib.util.spec_from_file_location("s2r_cpu_plan", S2R_CPU_PLAN)
    assert spec is not None and spec.loader is not None
    s2r = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = s2r  # dataclasses resolve their module through sys.modules
    try:
        spec.loader.exec_module(s2r)
    finally:
        sys.modules.pop(spec.name, None)
    cases = [smt(16), smt(12), smt(8), smt(6), smt(5),
             Topology(cores=tuple((i,) for i in range(24))),
             Topology(cores=tuple((i,) for i in range(12)), isolated=frozenset({3, 4}))]
    for topo in cases:
        ours = make_plan(topo, env={})
        theirs = s2r.make_plan(s2r.Topology(cores=topo.cores, isolated=topo.isolated), env={})
        assert (ours.rt, ours.reserved, ours.general) == (dict(theirs.rt), tuple(theirs.reserved),
                                                         tuple(theirs.general)), topo
