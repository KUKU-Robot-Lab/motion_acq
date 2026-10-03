from motion_acq.teleop.standby import GRIPPER_PARK_HOLD_S, GripperHomeStandby


def test_default_requires_two_seconds_of_continuous_physical_closure() -> None:
    standby = GripperHomeStandby()

    assert GRIPPER_PARK_HOLD_S == 2.0
    assert standby.update({"right": 0.0}, 10.0, ("right",)) == ((), ())
    assert standby.update({"right": 1.0}, 11.5, ("right",)) == ((), ())
    assert standby.update({"right": 0.0}, 12.0, ("right",)) == ((), ())
    assert standby.update({"right": 0.0}, 13.99, ("right",)) == ((), ())
    assert standby.update({"right": 0.0}, 14.0, ("right",)) == (("right",), ())


def test_missing_physical_feedback_cancels_close_timer() -> None:
    standby = GripperHomeStandby(hold_s=1.0)

    assert standby.update({"left": 0.0}, 0.0, ("left",)) == ((), ())
    assert standby.update({}, 2.0, ("left",)) == ((), ())
    assert standby.update({"left": 0.0}, 2.1, ("left",)) == ((), ())
    assert standby.update({"left": 0.0}, 3.2, ("left",)) == (("left",), ())


def test_handumi_can_wake_arm_without_robot_feedback() -> None:
    standby = GripperHomeStandby(initial_standby=True)

    assert standby.update(
        {},
        0.0,
        ("left",),
        wake_openings={"left": 1.0},
    ) == ((), ("left",))
