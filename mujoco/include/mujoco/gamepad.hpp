#pragma once

#include <memory>
#include <string>

namespace mujoco {

struct GamepadState {
  bool connected = false;
  float left_x = 0.0f;
  float left_y = 0.0f;
  float right_x = 0.0f;
  float right_y = 0.0f;
  float left_trigger = 0.0f;
  float right_trigger = 0.0f;
  bool a = false;
  bool b = false;
  bool x = false;
  bool y = false;
  bool left_bumper = false;
  bool right_bumper = false;
  bool dpad_left = false;
  bool dpad_right = false;
  bool dpad_up = false;
  bool dpad_down = false;
};

class Gamepad {
 public:
  Gamepad();
  ~Gamepad();
  Gamepad(const Gamepad&) = delete;
  Gamepad& operator=(const Gamepad&) = delete;
  GamepadState poll();
  bool available() const;
  std::string status() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

float ApplyDeadzone(float value, float deadzone);

}  // namespace mujoco
