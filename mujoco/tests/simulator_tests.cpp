#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "mujoco/runtime_config.hpp"
#include "mujoco/simulator.hpp"

namespace {

void expect_close(double actual, double expected, const char* message) {
  if (std::abs(actual - expected) > 1.0e-9)
    throw std::runtime_error(message);
}

}  // namespace

int main() {
  try {
    const std::filesystem::path repository_root = MUJOCO_REPOSITORY_ROOT;
    auto profile = mujoco::TaskProfile::Load(
        repository_root / "mujoco/policies/zgwsarm", repository_root);
    mujoco::MujocoSimulator simulation(profile, false);
    std::vector<double> torque(profile.dof_names.size(), 0.0);

    constexpr double stiffness = 200.0;
    constexpr double force_limit = 70.0;
    simulation.start_end_effector_force_field(
        stiffness, 0.0, force_limit, "robot_relative_static");
    if (!simulation.end_effector_force_field_active())
      throw std::runtime_error("force field did not activate");
    const auto anchor = simulation.end_effector_force_field_anchor_world();
    const auto anchor_local =
        simulation.end_effector_force_field_anchor_local();
    const auto initial_ee = simulation.end_effector_debug_state().world_position;
    for (int axis = 0; axis < 3; ++axis)
      expect_close(anchor[axis], initial_ee[axis],
                   "force-field anchor was not latched at current EE point");

    simulation.step(torque);
    const auto displaced_ee = simulation.end_effector_debug_state().world_position;
    simulation.step(torque);
    const auto updated_anchor =
        simulation.end_effector_force_field_anchor_world();
    const auto unchanged_anchor_local =
        simulation.end_effector_force_field_anchor_local();
    const auto spring = simulation.end_effector_spring_force_world();
    const auto spring_debug =
        simulation.end_effector_spring_force_debug_state();
    for (int axis = 0; axis < 3; ++axis) {
      expect_close(unchanged_anchor_local[axis], anchor_local[axis],
                   "robot-relative force anchor changed in its local frame");
      expect_close(spring[axis],
                   std::clamp(stiffness *
                                  (updated_anchor[axis] - displaced_ee[axis]),
                              -force_limit, force_limit),
                   "spring force does not match the robot-relative field");
      expect_close(spring_debug.clipped_world[axis], spring[axis],
                   "spring debug clipped force mismatch");
    }

    const auto wheel_positions = simulation.wheel_positions_base();
    if (wheel_positions.size() != profile.wheel_dof_names.size())
      throw std::runtime_error("wheel diagnostic order/size mismatch");

    simulation.stop_end_effector_force_field();
    if (simulation.end_effector_force_field_active())
      throw std::runtime_error("force field did not stop");
    simulation.step(torque);
    const auto cleared = simulation.end_effector_applied_force_world();
    for (double component : cleared)
      expect_close(component, 0.0, "spring force persisted after stopping field");

    simulation.reset();
    mujoco::ForceAnchorMotionConfig motion;
    motion.velocity_range = {{0.01, 0.01}};
    motion.duration_range = {{10.0, 10.0}};
    simulation.start_end_effector_force_field(
        stiffness, 0.0, force_limit, "robot_relative_moving", motion);
    const auto moving_local_before =
        simulation.end_effector_force_field_anchor_local();
    simulation.step(torque);
    const auto moving_local_after =
        simulation.end_effector_force_field_anchor_local();
    double moving_distance_squared = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double delta = moving_local_after[axis] - moving_local_before[axis];
      moving_distance_squared += delta * delta;
    }
    expect_close(std::sqrt(moving_distance_squared),
                 0.01 * profile.physics_dt,
                 "moving force anchor did not respect its configured speed");

    simulation.stop_end_effector_force_field();
    simulation.reset();
    simulation.start_end_effector_force_field(
        stiffness, 0.0, force_limit, "world_fixed");
    const auto world_fixed_before =
        simulation.end_effector_force_field_anchor_world();
    simulation.step(torque);
    const auto world_fixed_after =
        simulation.end_effector_force_field_anchor_world();
    for (int axis = 0; axis < 3; ++axis)
      expect_close(world_fixed_after[axis], world_fixed_before[axis],
                   "world-fixed force anchor moved with the robot");

    std::cout << "all simulator force-field tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "simulator force test failed: " << error.what() << '\n';
    return 1;
  }
}
