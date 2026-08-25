#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "mujoco/runtime_config.hpp"

namespace mujoco {

// User-editable launch settings.  Training-dependent observation/control
// values remain in the exported deployment bundle's runtime.cfg.
struct SimulatorConfig {
  std::filesystem::path source_path;
  std::string task_name;
  std::filesystem::path repository_root;
  std::filesystem::path deployment_bundle;
  std::filesystem::path scene_path;
  std::string policy_backend = "torchscript";
  std::filesystem::path policy_path;
  bool viewer = true;
  bool realtime = true;
  long long steps = 0;
  double status_interval_seconds = 1.0;
  std::optional<double> teleop_deadzone;
  std::optional<double> teleop_force_limit;
  std::optional<std::vector<double>> teleop_position_rates;
  std::optional<double> teleop_wrist_rate;
  std::optional<double> teleop_gripper_rate;
  bool force_field_enabled = true;
  double force_field_stiffness = 200.0;
  double force_field_damping = 6.0;
  double force_field_limit = 70.0;
  std::optional<double> startup_fold_duration;
  std::optional<double> startup_stand_duration;
  std::optional<std::vector<std::string>> startup_dog_dof_names;
  std::optional<std::vector<double>> startup_fold_positions;
  std::optional<std::vector<double>> startup_stand_positions;
  std::optional<std::vector<double>> startup_p_gains;
  std::optional<std::vector<double>> startup_d_gains;

  static SimulatorConfig Load(const std::filesystem::path& path);
  void apply_profile_overrides(TaskProfile& profile) const;
};

}  // namespace mujoco
