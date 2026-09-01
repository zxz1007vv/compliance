#include "mujoco/control.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mujoco {
namespace {

double smoothstep(double value) {
  const double x = std::clamp(value, 0.0, 1.0);
  return x * x * (3.0 - 2.0 * x);
}

}  // namespace

const char* RobotControlModeName(RobotControlMode mode) {
  switch (mode) {
    case RobotControlMode::kZeroTorque: return "dog_zero_arm_hold";
    case RobotControlMode::kFolding: return "folding";
    case RobotControlMode::kStanding: return "standing";
    case RobotControlMode::kStandby: return "standby";
    case RobotControlMode::kRl: return "rl";
  }
  return "unknown";
}

TaskController::TaskController(const TaskProfile& profile) : profile_(profile) {
  const bool explicit_startup = !profile_.startup_dog_dof_names.empty();
  std::vector<std::string> dog_names = profile_.startup_dog_dof_names;
  if (!explicit_startup) {
    dog_names = profile_.leg_dof_names;
    dog_names.insert(dog_names.end(), profile_.wheel_dof_names.begin(),
                     profile_.wheel_dof_names.end());
  }
  for (std::size_t startup_index = 0; startup_index < dog_names.size(); ++startup_index) {
    const std::size_t index = profile_.dof_index(dog_names[startup_index]);
    dog_indices_.push_back(index);
    fold_positions_.push_back(explicit_startup
                                  ? profile_.startup_fold_positions[startup_index]
                                  : profile_.default_dof_positions[index]);
    stand_positions_.push_back(explicit_startup
                                   ? profile_.startup_stand_positions[startup_index]
                                   : profile_.default_dof_positions[index]);
    startup_p_gains_.push_back(explicit_startup
                                   ? profile_.startup_p_gains[startup_index]
                                   : profile_.p_gains[index]);
    startup_d_gains_.push_back(explicit_startup
                                   ? profile_.startup_d_gains[startup_index]
                                   : profile_.d_gains[index]);
  }
  for (const auto& name : profile_.arm_dof_names)
    arm_indices_.push_back(profile_.dof_index(name));
  for (const auto& name : profile_.wheel_dof_names)
    wheel_indices_.push_back(profile_.dof_index(name));
  reset();
}

void TaskController::reset() {
  previous_arm_target_ = profile_.default_dof_positions;
  phase_start_positions_ = profile_.default_dof_positions;
  phase_elapsed_ = 0.0;
  wheel_lock_reference_ = profile_.default_dof_positions;
  wheel_lock_active_ = false;
  mode_ = RobotControlMode::kZeroTorque;
}

void TaskController::start_standup(const RobotState& state) {
  if (state.joint_position.size() != static_cast<std::size_t>(profile_.policy_output_dim))
    throw std::runtime_error("Cannot start stand-up from an invalid robot state");
  phase_start_positions_ = state.joint_position;
  phase_elapsed_ = 0.0;
  mode_ = RobotControlMode::kFolding;
}

bool TaskController::start_rl(const RobotState& state) {
  if (mode_ != RobotControlMode::kStandby) return false;
  if (state.joint_position.size() != static_cast<std::size_t>(profile_.policy_output_dim))
    throw std::runtime_error("Cannot start RL from an invalid robot state");
  phase_start_positions_ = state.joint_position;
  previous_arm_target_ = state.joint_position;
  phase_elapsed_ = 0.0;
  wheel_lock_active_ = false;
  mode_ = RobotControlMode::kRl;
  return true;
}

std::vector<float> TaskController::prepare_action(const std::vector<float>& policy_action) const {
  if (policy_action.size() != static_cast<std::size_t>(profile_.policy_output_dim))
    throw std::runtime_error("Policy produced the wrong action dimension");
  std::vector<float> action = policy_action;
  const float clip = static_cast<float>(profile_.action_clip);
  for (float& value : action) {
    if (!std::isfinite(value)) throw std::runtime_error("Policy produced a non-finite action");
    value = std::clamp(value, -clip, clip);
  }
  // This matches the B1+Z1 training/evaluation task's closed gripper baseline.
  if (profile_.is_b1_z1() && !profile_.direct_gripper_dof.empty())
    action[profile_.dof_index(profile_.direct_gripper_dof)] = -0.1f;
  return action;
}

void TaskController::clamp_torque(std::size_t index, double& torque) const {
  if (std::isfinite(profile_.joint_effort[index]))
    torque = std::clamp(torque, -profile_.joint_effort[index], profile_.joint_effort[index]);
}

void TaskController::apply_arm_default_pd(const RobotState& state,
                                          ControlOutput& output) {
  for (const std::size_t index : arm_indices_) {
    const double target = profile_.default_dof_positions[index];
    output.target_position[index] = target;
    output.torque[index] = profile_.p_gains[index] *
                               (target - state.joint_position[index]) -
                           profile_.d_gains[index] * state.joint_velocity[index];
    clamp_torque(index, output.torque[index]);
    previous_arm_target_[index] = target;
  }
}

ControlOutput TaskController::compute_rl(const std::vector<float>& action,
                                         const RobotState& state,
                                         const CommandState& teleop) {
  const auto n = static_cast<std::size_t>(profile_.policy_output_dim);
  if (action.size() != n || state.joint_position.size() != n || state.joint_velocity.size() != n)
    throw std::runtime_error("Controller dimensions do not match task profile");
  ControlOutput output{std::vector<double>(n, 0.0), std::vector<double>(n, 0.0)};
  for (std::size_t index = 0; index < n; ++index) {
    double target = profile_.default_dof_positions[index] +
                    static_cast<double>(action[index]) * profile_.action_scale_per_dof[index];
    if (std::isfinite(profile_.joint_lower[index])) target = std::max(target, profile_.joint_lower[index]);
    if (std::isfinite(profile_.joint_upper[index])) target = std::min(target, profile_.joint_upper[index]);

    if (profile_.control_kind[index] == "arm_position_pd" &&
        profile_.arm_target_velocity_limit_scale > 0.0) {
      const double max_step = profile_.joint_velocity[index] * profile_.physics_dt *
                              profile_.arm_target_velocity_limit_scale;
      target = std::clamp(target, previous_arm_target_[index] - max_step,
                          previous_arm_target_[index] + max_step);
    }
    output.target_position[index] = target;
  }

  if (teleop.input_active() && teleop.has_wrist_target())
    output.target_position[profile_.dof_index(profile_.direct_wrist_dof)] = teleop.wrist_target();
  if (teleop.input_active() && teleop.has_gripper_target())
    output.target_position[profile_.dof_index(profile_.direct_gripper_dof)] = teleop.gripper_target();

  const bool lock_wheels = profile_.is_zgwsarm() &&
      profile_.lock_wheels_for_yaw &&
      std::abs(teleop.values()[2]) > profile_.wheel_lock_command_threshold;
  if (lock_wheels && !wheel_lock_active_) {
    for (const std::size_t index : wheel_indices_)
      wheel_lock_reference_[index] = state.joint_position[index];
  }
  wheel_lock_active_ = lock_wheels;

  for (std::size_t index = 0; index < n; ++index) {
    if (profile_.control_kind[index] == "wheel_torque" && lock_wheels) {
      output.target_position[index] = wheel_lock_reference_[index];
      output.torque[index] = profile_.wheel_lock_kp *
                                 (wheel_lock_reference_[index] -
                                  state.joint_position[index]) -
                             profile_.wheel_lock_kd *
                                 state.joint_velocity[index];
    } else if (profile_.control_kind[index] == "wheel_torque") {
      output.torque[index] = static_cast<double>(action[index]) *
                                 profile_.action_scale_per_dof[index] * profile_.p_gains[index] -
                             profile_.d_gains[index] * state.joint_velocity[index];
    } else {
      output.torque[index] = profile_.p_gains[index] *
                                 (output.target_position[index] - state.joint_position[index]) -
                             profile_.d_gains[index] * state.joint_velocity[index];
    }
    clamp_torque(index, output.torque[index]);
  }
  previous_arm_target_ = output.target_position;
  return output;
}

ControlOutput TaskController::compute(const std::vector<float>& action,
                                      const RobotState& state,
                                      const CommandState& teleop) {
  const auto n = static_cast<std::size_t>(profile_.policy_output_dim);
  if (action.size() != n || state.joint_position.size() != n ||
      state.joint_velocity.size() != n) {
    throw std::runtime_error("Controller dimensions do not match task profile");
  }

  if (mode_ == RobotControlMode::kZeroTorque) {
    ControlOutput output{std::vector<double>(n, 0.0), state.joint_position};
    apply_arm_default_pd(state, output);
    return output;
  }

  if (mode_ == RobotControlMode::kFolding ||
      mode_ == RobotControlMode::kStanding ||
      mode_ == RobotControlMode::kStandby) {
    ControlOutput output{std::vector<double>(n, 0.0), state.joint_position};
    apply_arm_default_pd(state, output);
    double alpha = 1.0;
    if (mode_ == RobotControlMode::kFolding)
      alpha = smoothstep(phase_elapsed_ / profile_.startup_fold_duration);
    else if (mode_ == RobotControlMode::kStanding)
      alpha = smoothstep(phase_elapsed_ / profile_.startup_stand_duration);

    for (std::size_t startup_index = 0; startup_index < dog_indices_.size();
         ++startup_index) {
      const std::size_t index = dog_indices_[startup_index];
      double target = stand_positions_[startup_index];
      if (mode_ == RobotControlMode::kFolding) {
        target = phase_start_positions_[index] +
                 alpha * (fold_positions_[startup_index] - phase_start_positions_[index]);
      } else if (mode_ == RobotControlMode::kStanding) {
        target = fold_positions_[startup_index] +
                 alpha * (stand_positions_[startup_index] - fold_positions_[startup_index]);
      }
      output.target_position[index] = target;
      output.torque[index] = startup_p_gains_[startup_index] *
                                 (target - state.joint_position[index]) -
                             startup_d_gains_[startup_index] * state.joint_velocity[index];
      clamp_torque(index, output.torque[index]);
    }

    phase_elapsed_ += profile_.physics_dt;
    if (mode_ == RobotControlMode::kFolding &&
        phase_elapsed_ >= profile_.startup_fold_duration) {
      mode_ = RobotControlMode::kStanding;
      phase_elapsed_ = 0.0;
    } else if (mode_ == RobotControlMode::kStanding &&
               phase_elapsed_ >= profile_.startup_stand_duration) {
      mode_ = RobotControlMode::kStandby;
      phase_elapsed_ = 0.0;
    }
    return output;
  }

  return compute_rl(action, state, teleop);
}

SafetyResult CheckSafety(const TaskProfile& profile, const RobotState& state,
                         const std::vector<double>& torque) {
  const auto n = static_cast<std::size_t>(profile.policy_output_dim);
  if (state.joint_position.size() != n || state.joint_velocity.size() != n || torque.size() != n)
    return {false, "safety input dimension mismatch"};
  for (std::size_t index = 0; index < n; ++index) {
    if (!std::isfinite(state.joint_position[index]) || !std::isfinite(state.joint_velocity[index]) ||
        !std::isfinite(torque[index]))
      return {false, "non-finite state/control at " + profile.dof_names[index]};
    if (std::isfinite(profile.joint_lower[index]) &&
        state.joint_position[index] < profile.joint_lower[index] - 0.05)
      return {false, "lower joint limit exceeded at " + profile.dof_names[index] +
                         " (value=" + std::to_string(state.joint_position[index]) +
                         ", limit=" + std::to_string(profile.joint_lower[index]) + ")"};
    if (std::isfinite(profile.joint_upper[index]) &&
        state.joint_position[index] > profile.joint_upper[index] + 0.05)
      return {false, "upper joint limit exceeded at " + profile.dof_names[index] +
                         " (value=" + std::to_string(state.joint_position[index]) +
                         ", limit=" + std::to_string(profile.joint_upper[index]) + ")"};
    if (std::isfinite(profile.joint_velocity[index]) && profile.joint_velocity[index] > 0 &&
        std::abs(state.joint_velocity[index]) > 2.0 * profile.joint_velocity[index])
      return {false, "velocity safety limit exceeded at " + profile.dof_names[index] +
                         " (value=" + std::to_string(state.joint_velocity[index]) +
                         ", limit=" + std::to_string(2.0 * profile.joint_velocity[index]) + ")"};
  }
  return {};
}

}  // namespace mujoco
