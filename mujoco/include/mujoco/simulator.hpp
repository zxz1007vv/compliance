#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <string>
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

struct MousePerturbationDebugState {
  bool active = false;
  std::string body_name;
  std::array<double, 3> force_world{{0, 0, 0}};
};

class MujocoSimulator {
 public:
  explicit MujocoSimulator(const TaskProfile& profile, bool enable_viewer = false);
  ~MujocoSimulator();
  MujocoSimulator(const MujocoSimulator&) = delete;
  MujocoSimulator& operator=(const MujocoSimulator&) = delete;
  void reset();
  RobotState state() const;
  // Latch the current end-effector world point and enable a Cartesian
  // spring-damper field around it. Force is clamped independently per axis.
  void start_end_effector_force_field(double stiffness, double damping,
                                      double force_limit);
  void stop_end_effector_force_field();
  bool end_effector_force_field_active() const;
  std::array<double, 3> end_effector_force_field_anchor_world() const;
  std::array<double, 3> end_effector_spring_force_world() const;
  void step(const std::vector<double>& torque);
  double time() const;
  // Base angular velocity about its local Z axis, matching the trained yaw-rate
  // tracking quantity.
  double base_yaw_rate() const;
  // Base linear velocity along its local X axis, matching the trained forward
  // velocity tracking quantity.
  double base_forward_velocity() const;
  double end_effector_contact_force() const;
  // Net contact force acting on the end effector, expressed in world XYZ.
  std::array<double, 3> end_effector_contact_force_world() const;
  // Total external force used by the last physics substep (spring + mouse).
  std::array<double, 3> end_effector_applied_force_world() const;
  MousePerturbationDebugState mouse_perturbation_debug_state() const;
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
