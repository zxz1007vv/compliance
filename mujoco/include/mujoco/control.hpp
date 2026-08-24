#pragma once

#include <vector>

#include "mujoco/command.hpp"
#include "mujoco/observation.hpp"
#include "mujoco/runtime_config.hpp"

namespace mujoco {

struct ControlOutput {
  std::vector<double> torque;
  std::vector<double> target_position;
};

enum class RobotControlMode {
  kZeroTorque,
  kFolding,
  kStanding,
  kStandby,
  kRl,
};

const char* RobotControlModeName(RobotControlMode mode);

class TaskController {
 public:
  explicit TaskController(const TaskProfile& profile);
  void reset();
  void start_standup(const RobotState& state);
  bool start_rl(const RobotState& state);
  RobotControlMode mode() const { return mode_; }
  bool policy_active() const { return mode_ == RobotControlMode::kRl; }
  std::vector<float> prepare_action(const std::vector<float>& policy_action) const;
  ControlOutput compute(const std::vector<float>& action, const RobotState& state,
                        const CommandState& teleop);

 private:
  const TaskProfile& profile_;
  std::vector<double> previous_arm_target_;
  std::vector<std::size_t> dog_indices_;
  std::vector<std::size_t> arm_indices_;
  std::vector<double> fold_positions_;
  std::vector<double> stand_positions_;
  std::vector<double> startup_p_gains_;
  std::vector<double> startup_d_gains_;
  std::vector<double> phase_start_positions_;
  RobotControlMode mode_ = RobotControlMode::kZeroTorque;
  double phase_elapsed_ = 0.0;

  ControlOutput compute_rl(const std::vector<float>& action, const RobotState& state,
                           const CommandState& teleop);
  void apply_arm_default_pd(const RobotState& state, ControlOutput& output);
  void clamp_torque(std::size_t index, double& torque) const;
};

struct SafetyResult {
  bool safe = true;
  std::string reason;
};

SafetyResult CheckSafety(const TaskProfile& profile, const RobotState& state,
                         const std::vector<double>& torque);

}  // namespace mujoco
