#include "mujoco/command.hpp"

#include <algorithm>
#include <cmath>

namespace mujoco {
namespace {
constexpr double kPi = 3.14159265358979323846;
bool rising(bool now, bool previous) { return now && !previous; }
}  // namespace

CommandState::CommandState(const TaskProfile& profile) : profile_(profile) { reset(); }

void CommandState::reset() {
  commands_.assign(profile_.command_defaults.begin(), profile_.command_defaults.end());
  gait_phase_ = 0.0;
  clock_ = {{1, 1, 1, 1}};
  input_active_ = false;
  previous_a_ = previous_b_ = previous_x_ = previous_y_ = false;
  has_wrist_target_ = !profile_.direct_wrist_dof.empty();
  has_gripper_target_ = !profile_.direct_gripper_dof.empty();
  wrist_target_ = has_wrist_target_
                      ? static_cast<float>(profile_.default_dof_positions[profile_.dof_index(profile_.direct_wrist_dof)])
                      : 0.0f;
  gripper_target_ = has_gripper_target_
                        ? static_cast<float>(profile_.default_dof_positions[profile_.dof_index(profile_.direct_gripper_dof)])
                        : 0.0f;
}

float CommandState::clamp_command(int index, float value) const {
  return std::clamp(value, static_cast<float>(profile_.command_limit_low[index]),
                    static_cast<float>(profile_.command_limit_high[index]));
}

float CommandState::normalize_axis(float value) const {
  return ApplyDeadzone(value, static_cast<float>(profile_.teleop_deadzone));
}

TeleopEvent CommandState::update(const GamepadState& pad, double dt) {
  TeleopEvent event;
  if (rising(pad.y, previous_y_)) {
    reset();
    event.request_reset = true;
  } else {
    event.request_standup = rising(pad.a, previous_a_);
    event.request_rl = rising(pad.b, previous_b_);
  }
  if (rising(pad.x, previous_x_) && !event.request_reset) {
    commands_[22] = force_mode() ? 0.0f : 1.0f;
    commands_[12] = commands_[13] = commands_[14] = 0.0f;
    event.mode_changed = true;
  }
  if (event.request_rl) {
    for (int index = 12; index <= 17; ++index)
      commands_[index] = static_cast<float>(profile_.command_defaults[index]);
    if (has_wrist_target_)
      wrist_target_ = static_cast<float>(profile_.default_dof_positions[profile_.dof_index(profile_.direct_wrist_dof)]);
    if (has_gripper_target_)
      gripper_target_ = static_cast<float>(profile_.default_dof_positions[profile_.dof_index(profile_.direct_gripper_dof)]);
  }

  input_active_ = pad.connected;
  const float lx = normalize_axis(pad.left_x);
  const float ly = normalize_axis(pad.left_y);
  const float rx = normalize_axis(pad.right_x);
  const float ry = normalize_axis(pad.right_y);
  const float radial_or_vertical = (pad.right_bumper ? 1.0f : 0.0f) -
                                   (pad.left_bumper ? 1.0f : 0.0f);

  if (input_active_) {
    commands_[0] = clamp_command(0, -ly *
        static_cast<float>(std::max(std::abs(profile_.command_active_low[0]), std::abs(profile_.command_active_high[0]))));
    commands_[1] = 0.0f;
    commands_[2] = clamp_command(2, -lx *
        static_cast<float>(std::max(std::abs(profile_.command_active_low[2]), std::abs(profile_.command_active_high[2]))));
    if (force_mode()) {
      commands_[12] = clamp_command(12, -ry * static_cast<float>(profile_.teleop_force_limit));
      commands_[13] = clamp_command(13, rx * static_cast<float>(profile_.teleop_force_limit));
      commands_[14] = clamp_command(
          14, radial_or_vertical * static_cast<float>(profile_.teleop_force_limit));
    } else {
      commands_[15] = clamp_command(15, commands_[15] + radial_or_vertical *
          static_cast<float>(profile_.teleop_position_rates[0] * dt));
      commands_[16] = clamp_command(16, commands_[16] - ry *
          static_cast<float>(profile_.teleop_position_rates[1] * dt));
      commands_[17] = clamp_command(17, commands_[17] - rx *
          static_cast<float>(profile_.teleop_position_rates[2] * dt));
    }
    if (has_wrist_target_) {
      const float direction = (pad.dpad_right ? 1.0f : 0.0f) - (pad.dpad_left ? 1.0f : 0.0f);
      const auto index = profile_.dof_index(profile_.direct_wrist_dof);
      wrist_target_ += direction * static_cast<float>(profile_.teleop_wrist_rate * dt);
      if (std::isfinite(profile_.joint_lower[index]) && std::isfinite(profile_.joint_upper[index]))
        wrist_target_ = std::clamp(wrist_target_, static_cast<float>(profile_.joint_lower[index]),
                                  static_cast<float>(profile_.joint_upper[index]));
    }
    if (has_gripper_target_) {
      const float direction = (pad.dpad_up ? 1.0f : 0.0f) - (pad.dpad_down ? 1.0f : 0.0f);
      const auto index = profile_.dof_index(profile_.direct_gripper_dof);
      gripper_target_ += direction * static_cast<float>(profile_.teleop_gripper_rate * dt);
      if (std::isfinite(profile_.joint_lower[index]) && std::isfinite(profile_.joint_upper[index]))
        gripper_target_ = std::clamp(gripper_target_, static_cast<float>(profile_.joint_lower[index]),
                                    static_cast<float>(profile_.joint_upper[index]));
    }
  } else {
    commands_[0] = commands_[1] = commands_[2] = 0.0f;
    commands_[12] = commands_[13] = commands_[14] = 0.0f;
  }

  previous_a_ = pad.a;
  previous_b_ = pad.b;
  previous_x_ = pad.x;
  previous_y_ = pad.y;
  return event;
}

void CommandState::advance_clock(double dt) {
  gait_phase_ = std::fmod(gait_phase_ + dt * commands_[4], 1.0);
  if (std::abs(commands_[0]) < 0.2f && std::abs(commands_[1]) < 0.2f &&
      std::abs(commands_[2]) < 0.2f) {
    clock_ = {{1, 1, 1, 1}};
    return;
  }
  const std::array<double, 4> raw{{
      gait_phase_ + commands_[5] + commands_[6] + commands_[7],
      gait_phase_ + commands_[6],
      gait_phase_ + commands_[7],
      gait_phase_ + commands_[5],
  }};
  const double duration = std::clamp(static_cast<double>(commands_[8]), 1e-4, 1.0 - 1e-4);
  for (std::size_t index = 0; index < raw.size(); ++index) {
    double phase = std::fmod(raw[index], 1.0);
    if (phase < 0) phase += 1.0;
    phase = phase < duration ? phase * (0.5 / duration)
                             : 0.5 + (phase - duration) * (0.5 / (1.0 - duration));
    clock_[index] = static_cast<float>(std::sin(2.0 * kPi * phase));
  }
}

}  // namespace mujoco
