"""Real Nova 2 glove driver(s): senseglove_ros hardware per glove, no prompt.

    ros2 launch motion_acq_hand nova2.launch.py              # both gloves
    ros2 launch motion_acq_hand nova2.launch.py side:=right

Run SenseCom with the gloves connected first (scripts/nova2.sh up). The
gloves come from configs/hands/nova2_gloves.yaml, so the senseglove_ros
checkout keeps its upstream gloves.yaml and no Enter prompt is needed (its
senseglove.launch.py waits for one). Per glove this starts what senseglove_ros
hardware.launch.py starts (ros2_control_node, robot_state_publisher, state
broadcasters, haptics controller) in /senseglove/glove<serial>/<rh|lh>, with
this package's config/nova2_<side>_controllers.yaml: its haptics_controller is a
forward_command_controller the hand node drives with the RH56F1 feedback
(10.05). The hand node reads <namespace>/senseglove_states.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from ament_index_python.packages import get_package_share_directory


def _glove_nodes(side: str, serial: str) -> list:
    """senseglove_ros hardware.launch.py (humble-dev a14a468) with this package's controllers."""
    from launch.substitutions import Command, FindExecutable
    from launch_ros.actions import Node

    robot = f"nova2_{side}"
    is_right = "true" if side == "right" else "false"
    xacro = f"{get_package_share_directory('senseglove_description')}/urdf/nova2/{robot}.urdf.xacro"
    description = {"robot_description": Command([
        FindExecutable(name="xacro"), " ", xacro, f" selected_robot:={robot} is_right:={is_right}",
        f" glove_serial:={serial} publish_rate:=60",
    ])}
    controllers = f"{get_package_share_directory('motion_acq_hand')}/config/{robot}_controllers.yaml"
    ns = f"/senseglove/glove{serial}/{'rh' if side == 'right' else 'lh'}"

    def spawner(name: str, *extra: str):
        return Node(package="controller_manager", executable="spawner", arguments=[name, *extra],
                    output="screen", namespace=ns)

    return [
        Node(package="controller_manager", executable="ros2_control_node",
             parameters=[description, controllers],
             arguments=["--ros-args", "--log-level", "resource_manager:=WARN"], output="screen", namespace=ns),
        Node(package="robot_state_publisher", executable="robot_state_publisher", parameters=[description],
             output="screen", namespace=ns),
        spawner("joint_state_broadcaster"),
        spawner("senseglove_state_broadcaster", "--param-file", controllers),
        spawner("haptics_controller", "--param-file", controllers),
    ]


def _gloves(context):
    from launch.actions import LogInfo
    from motion_acq_hand.common import load_gloves

    side = LaunchConfiguration("side").perform(context)
    gloves = load_gloves()
    sides = ["right", "left"] if side == "both" else [side]
    if any(s not in gloves for s in sides):
        raise RuntimeError(f"side must be both, right or left (gloves: {sorted(gloves)})")
    actions = []
    for s in sides:
        actions.append(LogInfo(msg=f"[SenseGlove] Launching: robot=nova2_{s} serial={gloves[s].serial}"))
        actions += _glove_nodes(s, gloves[s].serial)
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("side", default_value="both", choices=["both", "right", "left"]),
        OpaqueFunction(function=_gloves),
    ])
