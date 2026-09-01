#include "mujoco/runtime_config.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace mujoco {
namespace {

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<std::string> split_csv(const std::string& value) {
  if (value.empty()) return {};
  std::vector<std::string> output;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) output.push_back(trim(item));
  return output;
}

double parse_number(const std::string& value) {
  if (value == "nan") return std::numeric_limits<double>::quiet_NaN();
  if (value == "inf") return std::numeric_limits<double>::infinity();
  if (value == "-inf") return -std::numeric_limits<double>::infinity();
  std::size_t parsed = 0;
  const double result = std::stod(value, &parsed);
  if (parsed != value.size()) throw std::runtime_error("Invalid number: " + value);
  return result;
}

template <typename T>
void require_size(const std::vector<T>& values, std::size_t expected,
                  const std::string& name) {
  if (values.size() != expected) {
    throw std::runtime_error(name + " size is " + std::to_string(values.size()) +
                             ", expected " + std::to_string(expected));
  }
}

}  // namespace

FlatConfig::FlatConfig(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("Cannot open runtime config: " + path.string());
  std::string line;
  int line_number = 0;
  while (std::getline(stream, line)) {
    ++line_number;
    line = trim(line);
    if (line.empty() || line.front() == '#') continue;
    const auto equals = line.find('=');
    if (equals == std::string::npos) {
      throw std::runtime_error("Malformed runtime config line " +
                               std::to_string(line_number));
    }
    const auto key = trim(line.substr(0, equals));
    if (!values_.emplace(key, trim(line.substr(equals + 1))).second) {
      throw std::runtime_error("Duplicate runtime config key: " + key);
    }
  }
}

const std::string& FlatConfig::get(const std::string& key) const {
  const auto iterator = values_.find(key);
  if (iterator == values_.end()) throw std::runtime_error("Missing runtime key: " + key);
  return iterator->second;
}

int FlatConfig::integer(const std::string& key) const { return std::stoi(get(key)); }
double FlatConfig::number(const std::string& key) const { return parse_number(get(key)); }
std::vector<std::string> FlatConfig::strings(const std::string& key) const {
  return split_csv(get(key));
}
std::vector<double> FlatConfig::numbers(const std::string& key) const {
  std::vector<double> result;
  for (const auto& value : strings(key)) result.push_back(parse_number(value));
  return result;
}
std::vector<int> FlatConfig::integers(const std::string& key) const {
  std::vector<int> result;
  for (const auto& value : strings(key)) result.push_back(std::stoi(value));
  return result;
}

TaskProfile TaskProfile::Load(const std::filesystem::path& bundle,
                              const std::filesystem::path& repository_root) {
  const FlatConfig cfg(bundle / "runtime.cfg");
  TaskProfile p;
  p.bundle_dir = std::filesystem::absolute(bundle);
  p.task_name = cfg.get("task_name");
  p.policy_path = p.bundle_dir / cfg.get("policy_file");
  p.model_path = std::filesystem::absolute(repository_root / "mujoco" / cfg.get("model"));
  p.checkpoint_number = cfg.integer("checkpoint_number");
  p.policy_input_dim = cfg.integer("policy_input_dim");
  p.policy_output_dim = cfg.integer("policy_output_dim");
  p.frame_dim = cfg.integer("frame_dim");
  p.history_length = cfg.integer("history_length");
  p.physics_dt = cfg.number("physics_dt");
  p.decimation = cfg.integer("decimation");
  p.action_clip = cfg.number("action_clip");
  p.observation_clip = cfg.number("observation_clip");
  p.dof_position_scale = cfg.number("dof_position_scale");
  p.dof_velocity_scale = cfg.number("dof_velocity_scale");
  p.zero_position_dof_indices = cfg.integers("zero_position_dof_indices");
  p.command_names = cfg.strings("command_names");
  p.command_scales = cfg.numbers("command_scales");
  p.command_defaults = cfg.numbers("command_defaults");
  p.command_active_low = cfg.numbers("command_active_low");
  p.command_active_high = cfg.numbers("command_active_high");
  p.command_limit_low = cfg.numbers("command_limit_low");
  p.command_limit_high = cfg.numbers("command_limit_high");
  p.force_frame = cfg.get("force_frame");
  p.base_body = cfg.get("base_body");
  p.end_effector_body = cfg.get("end_effector_body");
  p.initial_base_position = cfg.numbers("initial_base_position");
  p.initial_base_quaternion_wxyz = cfg.numbers("initial_base_quaternion_wxyz");
  p.initial_base_linear_velocity = cfg.numbers("initial_base_linear_velocity");
  p.initial_base_angular_velocity = cfg.numbers("initial_base_angular_velocity");
  p.dof_names = cfg.strings("dof_names");
  p.default_dof_positions = cfg.numbers("default_dof_positions");
  p.joint_lower = cfg.numbers("joint_lower");
  p.joint_upper = cfg.numbers("joint_upper");
  p.joint_velocity = cfg.numbers("joint_velocity");
  p.joint_effort = cfg.numbers("joint_effort");
  p.leg_dof_names = cfg.strings("leg_dof_names");
  p.wheel_dof_names = cfg.strings("wheel_dof_names");
  p.arm_dof_names = cfg.strings("arm_dof_names");
  p.abad_dof_names = cfg.strings("abad_dof_names");
  p.direct_wrist_dof = cfg.get("direct_wrist_dof");
  p.direct_gripper_dof = cfg.get("direct_gripper_dof");
  p.wheel_radius = cfg.number("wheel_radius");
  p.arm_mount_translation = cfg.numbers("arm_mount_translation");
  p.arm_mount_yaw = cfg.number("arm_mount_yaw");
  p.command_base_height = cfg.number("command_base_height");
  p.action_scale_per_dof = cfg.numbers("action_scale_per_dof");
  p.p_gains = cfg.numbers("p_gains");
  p.d_gains = cfg.numbers("d_gains");
  p.control_kind = cfg.strings("control_kind");
  p.arm_target_velocity_limit_scale = cfg.number("arm_target_velocity_limit_scale");
  p.lock_wheels_for_yaw = cfg.get("lock_wheels_for_yaw") == "true";
  p.wheel_lock_command_threshold = cfg.number("wheel_lock_command_threshold");
  p.wheel_lock_kp = cfg.number("wheel_lock_kp");
  p.wheel_lock_kd = cfg.number("wheel_lock_kd");
  p.teleop_force_limit = cfg.number("teleop_force_limit");
  p.teleop_deadzone = cfg.number("teleop_deadzone");
  p.teleop_precision_scale = cfg.number("teleop_precision_scale");
  p.teleop_position_rates = cfg.numbers("teleop_position_rates");
  p.teleop_wrist_rate = cfg.number("teleop_wrist_rate");
  p.teleop_gripper_rate = cfg.number("teleop_gripper_rate");
  p.validate();
  return p;
}

void TaskProfile::validate() const {
  if (!is_zgwsarm() && !is_b1_z1()) throw std::runtime_error("Unsupported task: " + task_name);
  const std::size_t n = static_cast<std::size_t>(policy_output_dim);
  if (n == 0 || frame_dim != 26 + 3 * policy_output_dim + 4 ||
      policy_input_dim != frame_dim * history_length) {
    throw std::runtime_error("Policy/observation dimensions are internally inconsistent");
  }
  require_size(command_names, 23, "command_names");
  require_size(command_scales, 23, "command_scales");
  require_size(command_defaults, 23, "command_defaults");
  require_size(command_active_low, 23, "command_active_low");
  require_size(command_active_high, 23, "command_active_high");
  require_size(command_limit_low, 23, "command_limit_low");
  require_size(command_limit_high, 23, "command_limit_high");
  require_size(dof_names, n, "dof_names");
  require_size(initial_base_position, 3, "initial_base_position");
  require_size(initial_base_quaternion_wxyz, 4, "initial_base_quaternion_wxyz");
  require_size(initial_base_linear_velocity, 3, "initial_base_linear_velocity");
  require_size(initial_base_angular_velocity, 3, "initial_base_angular_velocity");
  require_size(default_dof_positions, n, "default_dof_positions");
  require_size(joint_lower, n, "joint_lower");
  require_size(joint_upper, n, "joint_upper");
  require_size(joint_velocity, n, "joint_velocity");
  require_size(joint_effort, n, "joint_effort");
  require_size(action_scale_per_dof, n, "action_scale_per_dof");
  require_size(p_gains, n, "p_gains");
  require_size(d_gains, n, "d_gains");
  require_size(control_kind, n, "control_kind");
  if (is_zgwsarm() && lock_wheels_for_yaw &&
      (wheel_lock_command_threshold <= 0.0 || wheel_lock_kp <= 0.0 ||
       wheel_lock_kd <= 0.0)) {
    throw std::runtime_error("Invalid locked-wheel yaw control contract");
  }
  require_size(arm_mount_translation, 3, "arm_mount_translation");
  require_size(teleop_position_rates, 3, "teleop_position_rates");
  if (physics_dt <= 0 || decimation <= 0 || action_clip <= 0 || observation_clip <= 0)
    throw std::runtime_error("Invalid timestep, decimation, or clipping contract");
  if (teleop_deadzone < 0.0 || teleop_deadzone >= 1.0 ||
      teleop_precision_scale <= 0.0 || teleop_precision_scale > 1.0 ||
      teleop_force_limit < 0.0 || teleop_wrist_rate < 0.0 ||
      teleop_gripper_rate < 0.0 ||
      std::any_of(teleop_position_rates.begin(), teleop_position_rates.end(),
                  [](double value) { return value < 0.0; })) {
    throw std::runtime_error("Invalid teleoperation override");
  }
  if (startup_fold_duration <= 0.0 || startup_stand_duration <= 0.0) {
    throw std::runtime_error("Startup interpolation durations must be positive");
  }
  std::unordered_set<std::string> unique(dof_names.begin(), dof_names.end());
  if (unique.size() != n) throw std::runtime_error("DOF names must be unique");
  if (startup_dog_dof_names.empty() &&
      (!startup_fold_positions.empty() || !startup_stand_positions.empty() ||
       !startup_p_gains.empty() || !startup_d_gains.empty())) {
    throw std::runtime_error(
        "startup dog arrays require startup_dog_dof_names");
  }
  if (!startup_dog_dof_names.empty()) {
    const std::size_t dog_count = startup_dog_dof_names.size();
    require_size(startup_fold_positions, dog_count, "startup_fold_positions");
    require_size(startup_stand_positions, dog_count, "startup_stand_positions");
    require_size(startup_p_gains, dog_count, "startup_p_gains");
    require_size(startup_d_gains, dog_count, "startup_d_gains");
    std::unordered_set<std::string> startup_unique;
    for (const auto& name : startup_dog_dof_names) {
      if (!unique.count(name))
        throw std::runtime_error("Startup DOF is not in the policy contract: " + name);
      if (!startup_unique.insert(name).second)
        throw std::runtime_error("Startup DOF names must be unique");
    }
  }
  for (int index : zero_position_dof_indices) {
    if (index < 0 || index >= policy_output_dim)
      throw std::runtime_error("zero_position_dof_indices contains an invalid index");
  }
  if (is_zgwsarm() && (policy_output_dim != 22 || frame_dim != 96 || history_length != 10))
    throw std::runtime_error("ZGWSARM deployment dimensions differ from the trained contract");
  if (is_b1_z1() && (policy_output_dim != 19 || frame_dim != 87 || history_length != 10))
    throw std::runtime_error("B1+Z1 deployment dimensions differ from the trained contract");
}

std::size_t TaskProfile::dof_index(const std::string& name) const {
  const auto iterator = std::find(dof_names.begin(), dof_names.end(), name);
  if (iterator == dof_names.end()) throw std::runtime_error("DOF not in policy contract: " + name);
  return static_cast<std::size_t>(std::distance(dof_names.begin(), iterator));
}

}  // namespace mujoco
