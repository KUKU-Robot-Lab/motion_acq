"""Real Nova 2 glove driver(s): senseglove_ros hardware per glove, no prompt.

    ros2 launch motion_acq_hand nova2.launch.py              # both gloves
    ros2 launch motion_acq_hand nova2.launch.py side:=right

Run SenseCom with the gloves connected first (scripts/nova2.sh up). The
gloves come from configs/hands/nova2_gloves.yaml, so the senseglove_ros
checkout keeps its upstream gloves.yaml and no Enter prompt is needed (its
senseglove.launch.py waits for one). Per glove this starts senseglove_ros
hardware.launch.py (ros2_control_node, state broadcasters, haptics
controller) in /senseglove/glove<serial>/<rh|lh>; the hand node reads
<namespace>/senseglove_states.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from ament_index_python.packages import get_package_share_directory


def _gloves(context):
    from motion_acq_hand.common import load_gloves

    side = LaunchConfiguration("side").perform(context)
    gloves = load_gloves()
    sides = ["right", "left"] if side == "both" else [side]
    if any(s not in gloves for s in sides):
        raise RuntimeError(f"side must be both, right or left (gloves: {sorted(gloves)})")
    hardware = f"{get_package_share_directory('senseglove_hardware_interface')}/launch/hardware.launch.py"
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(hardware),
            launch_arguments={
                "robot": f"nova2_{s}",
                "isRight": "true" if s == "right" else "false",
                "gloveSerial": gloves[s].serial,
            }.items(),
        )
        for s in sides
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("side", default_value="both", choices=["both", "right", "left"]),
        OpaqueFunction(function=_gloves),
    ])
