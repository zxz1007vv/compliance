#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace mujoco {

class FlatConfig {
 public:
  explicit FlatConfig(const std::filesystem::path& path);
  const std::string& get(const std::string& key) const;
  int integer(const std::string& key) const;
  double number(const std::string& key) const;
  std::vector<std::string> strings(const std::string& key) const;
  std::vector<double> numbers(const std::string& key) const;
  std::vector<int> integers(const std::string& key) const;

 private:
  std::unordered_map<std::string, std::string> values_;
};

struct TaskProfile {
  std::filesystem::path bundle_dir;
  std::string task_name;
  std::filesystem::path policy_path;
  std::filesystem::path model_path;
  int checkpoint_number = 0;
  int policy_input_dim = 0;
  int policy_output_dim = 0;
  int frame_dim = 0;
  int history_length = 0;
  double physics_dt = 0.0;
  int decimation = 0;
  double action_clip = 0.0;
  double observation_clip = 0.0;
  double dof_position_scale = 1.0;
  double dof_velocity_scale = 1.0;
  std::vector<int> zero_position_dof_indices;
  std::vector<std::string> command_names;
  std::vector<double> command_scales;
  std::vector<double> command_defaults;
  std::vector<double> command_active_low;
  std::vector<double> command_active_high;
  std::vector<double> command_limit_low;
  std::vector<double> command_limit_high;
  std::string force_frame;
  std::string base_body;
  std::string end_effector_body;
  std::vector<double> initial_base_position;
  std::vector<double> initial_base_quaternion_wxyz;
  std::vector<double> initial_base_linear_velocity;
  std::vector<double> initial_base_angular_velocity;
  std::vector<std::string> dof_names;
  std::vector<double> default_dof_positions;
  std::vector<double> joint_lower;
  std::vector<double> joint_upper;
  std::vector<double> joint_velocity;
  std::vector<double> joint_effort;
  std::vector<std::string> leg_dof_names;
  std::vector<std::string> wheel_dof_names;
  std::vector<std::string> arm_dof_names;
  std::vector<std::string> abad_dof_names;
  std::string direct_wrist_dof;
  std::string direct_gripper_dof;
  double wheel_radius = 0.0;
  std::vector<double> arm_mount_translation;
  double arm_mount_yaw = 0.0;
  double command_base_height = 0.0;
  std::vector<double> action_scale_per_dof;
  std::vector<double> p_gains;
  std::vector<double> d_gains;
  std::vector<std::string> control_kind;
  double arm_target_velocity_limit_scale = 0.0;
  bool lock_wheels_for_yaw = false;
  double wheel_lock_command_threshold = 0.05;
  double wheel_lock_kp = 0.0;
  double wheel_lock_kd = 0.0;
  double teleop_force_limit = 0.0;
  double teleop_deadzone = 0.1;
  double teleop_precision_scale = 0.25;
  std::vector<double> teleop_position_rates;
  double teleop_wrist_rate = 0.75;
  double teleop_gripper_rate = 0.75;
  double startup_fold_duration = 2.0;
  double startup_stand_duration = 2.0;
  std::vector<std::string> startup_dog_dof_names;
  std::vector<double> startup_fold_positions;
  std::vector<double> startup_stand_positions;
  std::vector<double> startup_p_gains;
  std::vector<double> startup_d_gains;

  static TaskProfile Load(const std::filesystem::path& bundle_dir,
                          const std::filesystem::path& repository_root);
  void validate() const;
  std::size_t dof_index(const std::string& name) const;
  bool is_zgwsarm() const { return task_name == "zgwsarm_compliance"; }
  bool is_b1_z1() const { return task_name == "b1_z1_ik"; }
  double control_dt() const { return physics_dt * decimation; }
};

}  // namespace mujoco
