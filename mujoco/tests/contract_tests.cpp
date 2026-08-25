#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "mujoco/command.hpp"
#include "mujoco/control.hpp"
#include "mujoco/gamepad.hpp"
#include "mujoco/launcher_config.hpp"
#include "mujoco/observation.hpp"

namespace {
void expect(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

mujoco::TaskProfile small_profile() {
  mujoco::TaskProfile p;
  p.task_name = "test";
  p.policy_output_dim = 2;
  p.frame_dim = 36;
  p.history_length = 3;
  p.policy_input_dim = 108;
  p.physics_dt = 0.002;
  p.decimation = 5;
  p.action_clip = 4;
  p.observation_clip = 100;
  p.dof_position_scale = 1;
  p.dof_velocity_scale = 0.1;
  p.command_names.resize(23);
  p.command_scales.assign(23, 1);
  p.command_defaults.assign(23, 0);
  p.command_defaults[4] = 2;
  p.command_defaults[8] = 0.5;
  p.command_active_low.assign(23, -1);
  p.command_active_high.assign(23, 1);
  p.command_limit_low.assign(23, -2);
  p.command_limit_high.assign(23, 2);
  p.command_limit_low[22] = 0;
  p.command_limit_high[22] = 1;
  p.dof_names = {"joint", "wheel"};
  p.default_dof_positions = {0.5, 0};
  p.joint_lower = {-1, NAN};
  p.joint_upper = {1, NAN};
  p.joint_velocity = {10, 100};
  p.joint_effort = {20, 10};
  p.action_scale_per_dof = {0.25, 0.25};
  p.p_gains = {100, 60};
  p.d_gains = {1, 0.2};
  p.control_kind = {"position_pd", "wheel_torque"};
  p.arm_dof_names = {"joint"};
  p.startup_dog_dof_names = {"wheel"};
  p.startup_fold_positions = {0.0};
  p.startup_stand_positions = {0.0};
  p.startup_p_gains = {60};
  p.startup_d_gains = {0.2};
  p.startup_fold_duration = p.physics_dt;
  p.startup_stand_duration = p.physics_dt;
  p.teleop_deadzone = 0.1;
  p.teleop_precision_scale = 0.25;
  p.teleop_force_limit = 2.0;
  p.teleop_position_rates = {0.2, 0.75, 0.75};
  return p;
}
}  // namespace

int main() {
  try {
    expect(mujoco::ApplyDeadzone(0.05f, 0.1f) == 0.0f, "deadzone failed");
    const auto launch = mujoco::SimulatorConfig::Load(
        std::filesystem::path(MUJOCO_SOURCE_DIR) / "config/zgwsarm_compliance.yaml");
    expect(launch.task_name == "zgwsarm_compliance", "launch task failed");
    expect(launch.policy_backend == "torchscript", "launch backend failed");
    expect(launch.viewer && launch.realtime, "launch booleans failed");
    expect(launch.force_field_enabled,
           "force field must default on");
    expect(launch.force_field_stiffness > 0.0 &&
               launch.force_field_damping >= 0.0 &&
               launch.force_field_limit > 0.0,
           "force-field settings must be physically valid");
    expect(launch.scene_path.filename() == "scene_terrain.xml", "launch scene failed");
    expect(launch.teleop_position_rates && launch.teleop_position_rates->size() == 3,
           "launch teleoperation overrides failed");
    expect(launch.startup_dog_dof_names &&
               launch.startup_dog_dof_names->size() == 16,
           "launch startup joint mapping failed");
    expect(launch.startup_fold_duration && launch.startup_stand_duration &&
               std::abs(*launch.startup_fold_duration - 1.0) < 1e-12 &&
               std::abs(*launch.startup_stand_duration - 1.0) < 1e-12,
           "launch startup phases must each take one second");
    const auto gravity = mujoco::ProjectedGravity({{1, 0, 0, 0}});
    expect(std::abs(gravity[0]) < 1e-7f && std::abs(gravity[1]) < 1e-7f &&
               std::abs(gravity[2] + 1.0f) < 1e-7f,
           "projected gravity failed");

    auto profile = small_profile();
    mujoco::ObservationHistory history(profile.frame_dim, profile.history_length);
    expect(std::all_of(history.values().begin(), history.values().end(),
                       [](float value) { return value == 0.0f; }),
           "history must start at zero");
    std::vector<float> frame(static_cast<std::size_t>(profile.frame_dim), 1.0f);
    history.append(frame);
    expect(history.values().front() == 0.0f && history.values().back() == 1.0f,
           "history append order failed");

    mujoco::RobotState state;
    state.joint_position = {0.5, 0};
    state.joint_velocity = {0, 2};
    mujoco::CommandState commands(profile);
    mujoco::GamepadState pad;
    pad.connected = true;
    pad.left_y = -1;
    commands.update(pad, 0.01);
    expect(commands.input_active() && commands.values()[0] > 0.9f,
           "bumper-free base mapping failed");
    pad.x = true;
    const auto mode_event = commands.update(pad, 0.01);
    expect(mode_event.mode_changed && commands.force_mode(), "X mode mapping failed");
    pad.x = false;
    pad.right_bumper = true;
    commands.update(pad, 0.01);
    expect(std::abs(commands.values()[14] - 2.0f) < 1e-6f,
           "RB force-Z mapping failed");
    pad.right_bumper = false;
    pad.right_trigger = 1.0f;
    commands.update(pad, 0.01);
    expect(commands.values()[14] == 0.0f, "RT must no longer control force-Z");
    pad.right_trigger = 0.0f;
    pad.a = true;
    expect(commands.update(pad, 0.01).request_standup, "A stand-up mapping failed");
    pad.a = false;
    commands.update(pad, 0.01);
    pad.b = true;
    expect(commands.update(pad, 0.01).request_rl, "B RL mapping failed");
    pad.b = false;
    commands.update(pad, 0.01);
    pad.y = true;
    expect(commands.update(pad, 0.01).request_reset, "Y zero-torque mapping failed");

    mujoco::TaskController controller(profile);
    state.joint_position[0] = 0.4;
    auto control = controller.compute({0, 1}, state, commands);
    expect(controller.mode() == mujoco::RobotControlMode::kZeroTorque,
           "controller must start in dog-zero/arm-hold mode");
    expect(std::abs(control.torque[0] - 10.0) < 1e-9 &&
               control.torque[1] == 0.0,
           "startup must hold the arm while the dog stays zero torque");
    state.joint_position[0] = 0.5;
    controller.start_standup(state);
    controller.compute({0, 1}, state, commands);
    controller.compute({0, 1}, state, commands);
    expect(controller.mode() == mujoco::RobotControlMode::kStandby,
           "fold/stand sequence failed");
    expect(controller.start_rl(state), "B-to-RL transition failed");
    expect(controller.mode() == mujoco::RobotControlMode::kRl,
           "B must hand the arm and dog directly to RL");
    control = controller.compute({0, 1}, state, commands);
    expect(std::abs(control.torque[0]) < 1e-9, "position PD failed");
    expect(std::abs(control.torque[1] - 10.0) < 1e-9,
           "wheel torque/clipping failed");
    const auto obs = mujoco::ObservationBuilder(profile).frame(
        state, commands.values(), {0, 1}, commands.clock());
    expect(obs.size() == static_cast<std::size_t>(profile.frame_dim),
           "observation dimension failed");
    std::cout << "all core contract tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "contract test failed: " << error.what() << '\n';
    return 1;
  }
}
