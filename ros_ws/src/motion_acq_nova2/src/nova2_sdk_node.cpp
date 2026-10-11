// Nova 2 glove straight from the SenseGlove SDK (SGCore), the way Nova2Dex reads it (10.11 user:
// "switch to the other project"). Per glove, on /motion_acq/nova2_sdk/<side> (sensor_msgs/JointState,
// values in `position`):
//   nflex_thumb .. nflex_pinky   HandPose::GetNormalizedFlexion(clamped): 0 open .. 1 bent, from the SDK's
//                                own sensor calibration (pinky is the SDK's estimate: no pinky sensor)
//   <finger>_<joint>_<x|y|z>     HandPose::GetHandAngles (rad); thumb_0_x is the thumb's rotation across
//                                the palm that Nova2Dex maps to the Inspire thumb yaw
//   raw_*                        the six physical sensors (Nova2GloveSensorData::GetSensorValue)
// With "index influences others" OFF (Nova2Dex does the same; the SDK default is on, so moving the index
// moved the other fingers' tracking). The SDK's calibration state and instructions go out on
// /motion_acq/nova2_sdk/<side>/calibration (std_msgs/String, 1 Hz and on change); "reset" / "end" on
// .../calibration_cmd restart or lock it. Reads only: no haptics (the senseglove_ros driver keeps those).
// libsgcore has no RTTI typeinfo exported: the glove is cast after checking its device type.

#include <SenseGlove/Core/DeviceTypes.hpp>
#include <SenseGlove/Core/HandLayer.hpp>
#include <SenseGlove/Core/HandPose.hpp>
#include <SenseGlove/Core/HapticGlove.hpp>
#include <SenseGlove/Core/Nova2Glove.hpp>
#include <SenseGlove/Core/Nova2GloveSensorData.hpp>
#include <SenseGlove/Core/Quat.hpp>
#include <SenseGlove/Core/SenseCom.hpp>
#include <SenseGlove/Core/Vect3D.hpp>

#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>

namespace
{
using SensorData = SGCore::Nova::Nova2GloveSensorData;
const char * const FINGERS[] = {"thumb", "index", "middle", "ring", "pinky"};

struct Glove
{
  bool right = true;
  std::string side;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr state_pub;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr calibration_pub;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr command_sub;
  std::string last_calibration;
  rclcpp::Time last_calibration_t{0, 0, RCL_ROS_TIME};
  bool connected = false;
};
}  // namespace

class Nova2SdkNode : public rclcpp::Node
{
public:
  Nova2SdkNode()
  : Node("motion_acq_nova2_sdk")
  {
    const std::string sides = declare_parameter<std::string>("side", "both");
    const double rate_hz = declare_parameter<double>("rate_hz", 100.0);
    index_influences_others_ = declare_parameter<bool>("index_influences_others", false);
    if (sides != "both" && sides != "right" && sides != "left") {
      throw std::invalid_argument("side must be both, right or left, not " + sides);
    }
    for (const std::string side : {"right", "left"}) {
      if (sides != "both" && sides != side) {
        continue;
      }
      auto glove = std::make_shared<Glove>();
      glove->right = side == "right";
      glove->side = side;
      const std::string ns = "/motion_acq/nova2_sdk/" + side;
      glove->state_pub = create_publisher<sensor_msgs::msg::JointState>(ns, rclcpp::SensorDataQoS());
      glove->calibration_pub = create_publisher<std_msgs::msg::String>(ns + "/calibration", 10);
      Glove * raw = glove.get();
      glove->command_sub = create_subscription<std_msgs::msg::String>(
        ns + "/calibration_cmd", 10, [this, raw](const std_msgs::msg::String & msg) {on_command(*raw, msg.data);});
      gloves_.push_back(glove);
    }
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate_hz), [this]() {tick();});
    RCLCPP_INFO(
      get_logger(), "Nova 2 SDK reader: %s at %.0f Hz, index influences others %s, SenseCom %s",
      sides.c_str(), rate_hz, index_influences_others_ ? "on" : "off",
      SGCore::SenseCom::ScanningActive() ? "running" : "NOT running");
  }

private:
  void on_command(const Glove & glove, const std::string & command)
  {
    if (command == "reset") {
      SGCore::HandLayer::ResetCalibration(glove.right);
    } else if (command == "end") {
      SGCore::HandLayer::EndCalibration(glove.right);
    } else {
      RCLCPP_WARN(get_logger(), "%s: unknown calibration command '%s' (reset | end)", glove.side.c_str(), command.c_str());
      return;
    }
    RCLCPP_INFO(get_logger(), "%s: calibration %s", glove.side.c_str(), command.c_str());
  }

  void tick()
  {
    for (auto & glove : gloves_) {
      // 10.11 arm4090: SGCore threw std::invalid_argument (stof) while a glove came back ("no hand pose yet"
      // -> "reading") and the uncaught exception killed the node; a bad frame is skipped instead
      try {
        read(*glove);
      } catch (const std::exception & exc) {
        glove->connected = false;
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000, "%s glove: SDK read failed (%s), retrying", glove->side.c_str(), exc.what());
      }
    }
  }

  void set_connected(Glove & glove, bool connected, const char * why)
  {
    if (connected != glove.connected) {
      if (connected) {
        RCLCPP_INFO(get_logger(), "%s glove: reading", glove.side.c_str());
      } else {
        RCLCPP_WARN(get_logger(), "%s glove: %s", glove.side.c_str(), why);
      }
    }
    glove.connected = connected;
  }

  void read(Glove & glove)
  {
    std::shared_ptr<SGCore::HapticGlove> base;
    if (!SGCore::HandLayer::GetGloveInstance(glove.right, base) || !base) {
      set_connected(glove, false, "not connected (SenseCom running? glove on?)");
      return;
    }
    if (base->GetDeviceType() != SGCore::EDeviceType::Nova2) {
      set_connected(glove, false, "not a Nova 2");
      return;
    }
    auto & nova2 = static_cast<SGCore::Nova::Nova2Glove &>(*base);
    if (nova2.DoesIndexInfluenceOthers() != index_influences_others_) {
      nova2.SetIndexInfluencesOthers(index_influences_others_);
    }
    SGCore::HandPose pose;
    if (!SGCore::HandLayer::GetHandPose(glove.right, pose)) {
      set_connected(glove, false, "no hand pose yet");
      return;
    }
    set_connected(glove, true, "");

    sensor_msgs::msg::JointState msg;
    msg.header.stamp = now();
    msg.header.frame_id = glove.side;
    auto add = [&msg](const std::string & name, double value) {
        msg.name.push_back(name);
        msg.position.push_back(value);
      };
    const std::vector<float> flexion = pose.GetNormalizedFlexion(true);
    for (std::size_t f = 0; f < flexion.size() && f < 5; ++f) {
      add(std::string("nflex_") + FINGERS[f], flexion[f]);
    }
    const auto & angles = pose.GetHandAngles();
    for (std::size_t f = 0; f < angles.size() && f < 5; ++f) {
      for (std::size_t j = 0; j < angles[f].size(); ++j) {
        const std::string prefix = std::string(FINGERS[f]) + "_" + std::to_string(j) + "_";
        add(prefix + "x", angles[f][j].GetX());
        add(prefix + "y", angles[f][j].GetY());
        add(prefix + "z", angles[f][j].GetZ());
      }
    }
    SensorData sensors;
    if (nova2.GetSensorData(sensors)) {
      using L = SensorData::ESensorLocation;
      add("raw_thumb_flex", sensors.GetSensorValue(SGCore::EFinger::Thumb, L::FlexionProximal));
      add("raw_thumb_abd", sensors.GetSensorValue(SGCore::EFinger::Thumb, L::Abduction));
      add("raw_index_prox", sensors.GetSensorValue(SGCore::EFinger::Index, L::FlexionProximal));
      add("raw_index_dist", sensors.GetSensorValue(SGCore::EFinger::Index, L::FlexionDistal));
      add("raw_middle", sensors.GetSensorValue(SGCore::EFinger::Middle, L::FlexionProximal));
      add("raw_ring", sensors.GetSensorValue(SGCore::EFinger::Ring, L::FlexionProximal));
    }
    glove.state_pub->publish(msg);

    const std::string calibration =
      SGCore::HapticGlove::ToString(SGCore::HandLayer::GetCalibrationState(glove.right)) + " | " +
      SGCore::HandLayer::GetCalibrationInstructions(glove.right);
    const rclcpp::Time t = now();
    if (calibration != glove.last_calibration || (t - glove.last_calibration_t).seconds() >= 1.0) {
      if (calibration != glove.last_calibration) {
        RCLCPP_INFO(get_logger(), "%s calibration: %s", glove.side.c_str(), calibration.c_str());
      }
      std_msgs::msg::String out;
      out.data = calibration;
      glove.calibration_pub->publish(out);
      glove.last_calibration = calibration;
      glove.last_calibration_t = t;
    }
  }

  bool index_influences_others_ = false;
  std::vector<std::shared_ptr<Glove>> gloves_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Nova2SdkNode>());
  rclcpp::shutdown();
  return 0;
}
