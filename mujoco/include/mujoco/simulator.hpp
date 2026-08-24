#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <vector>

#include "mujoco/observation.hpp"
#include "mujoco/runtime_config.hpp"

namespace mujoco {

struct EndEffectorDebugState {
  std::array<double, 3> world_position{{0, 0, 0}};
  std::array<double, 3> arm_position{{0, 0, 0}};
  // radius, pitch, yaw in the same arm frame as command indices 15:18.
  std::array<double, 3> arm_spherical{{0, 0, 0}};
};

class MujocoSimulator {
 public:
  explicit MujocoSimulator(const TaskProfile& profile, bool enable_viewer = false);
  ~MujocoSimulator();
  MujocoSimulator(const MujocoSimulator&) = delete;
  MujocoSimulator& operator=(const MujocoSimulator&) = delete;
  void reset();
  RobotState state() const;
  void step(const std::vector<double>& torque);
  double time() const;
  double end_effector_contact_force() const;
  EndEffectorDebugState end_effector_debug_state() const;
  // Synchronize state to the official passive `simulate` viewer. The first
  // call attaches the model and waits for viewer_loop() to enter its UI loop.
  void render();
  void viewer_loop();
  void request_viewer_close();
  bool viewer_should_close() const;
  bool viewer_enabled() const;
  bool viewer_paused() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mujoco
