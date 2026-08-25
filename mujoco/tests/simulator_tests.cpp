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
    simulation.start_end_effector_force_field(stiffness, 0.0, force_limit);
    if (!simulation.end_effector_force_field_active())
      throw std::runtime_error("force field did not activate");
    const auto anchor = simulation.end_effector_force_field_anchor_world();
    const auto initial_ee = simulation.end_effector_debug_state().world_position;
    for (int axis = 0; axis < 3; ++axis)
      expect_close(anchor[axis], initial_ee[axis],
                   "force-field anchor was not latched at current EE point");

    simulation.step(torque);
    const auto displaced_ee = simulation.end_effector_debug_state().world_position;
    simulation.step(torque);
    const auto spring = simulation.end_effector_spring_force_world();
    for (int axis = 0; axis < 3; ++axis)
      expect_close(spring[axis],
                   std::clamp(stiffness * (anchor[axis] - displaced_ee[axis]),
                              -force_limit, force_limit),
                   "spring force does not match the fixed-anchor field");

    simulation.stop_end_effector_force_field();
    if (simulation.end_effector_force_field_active())
      throw std::runtime_error("force field did not stop");
    simulation.step(torque);
    const auto cleared = simulation.end_effector_applied_force_world();
    for (double component : cleared)
      expect_close(component, 0.0, "spring force persisted after stopping field");

    std::cout << "all simulator force-field tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "simulator force test failed: " << error.what() << '\n';
    return 1;
  }
}
