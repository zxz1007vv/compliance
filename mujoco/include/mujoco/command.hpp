#pragma once

#include <array>
#include <vector>

#include "mujoco/gamepad.hpp"
#include "mujoco/runtime_config.hpp"

namespace mujoco {

struct TeleopEvent {
  bool request_reset = false;
  bool request_standup = false;
  bool request_rl = false;
  bool mode_changed = false;
};

class CommandState {
 public:
  explicit CommandState(const TaskProfile& profile);
  TeleopEvent update(const GamepadState& pad, double dt);
  void reset();
  void advance_clock(double dt);
  const std::vector<float>& values() const { return commands_; }
  const std::array<float, 4>& clock() const { return clock_; }
  bool force_mode() const { return commands_[22] >= 0.5f; }
  bool input_active() const { return input_active_; }
  bool has_wrist_target() const { return has_wrist_target_; }
  bool has_gripper_target() const { return has_gripper_target_; }
  float wrist_target() const { return wrist_target_; }
  float gripper_target() const { return gripper_target_; }

 private:
  const TaskProfile& profile_;
  std::vector<float> commands_;
  std::array<float, 4> clock_{{1, 1, 1, 1}};
  double gait_phase_ = 0.0;
  bool input_active_ = false;
  bool previous_a_ = false;
  bool previous_b_ = false;
  bool previous_x_ = false;
  bool previous_y_ = false;
  float wrist_target_ = 0.0f;
  float gripper_target_ = 0.0f;
  bool has_wrist_target_ = false;
  bool has_gripper_target_ = false;

  float clamp_command(int index, float value) const;
  float normalize_axis(float value) const;
};

}  // namespace mujoco
