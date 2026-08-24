#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#include "mujoco/command.hpp"
#include "mujoco/control.hpp"
#include "mujoco/gamepad.hpp"
#include "mujoco/launcher_config.hpp"
#include "mujoco/observation.hpp"
#include "mujoco/policy.hpp"
#include "mujoco/runtime_config.hpp"
#include "mujoco/simulator.hpp"

namespace {
volatile std::sig_atomic_t running = 1;
void stop(int) { running = 0; }

struct Options {
  std::optional<mujoco::SimulatorConfig> config;
  std::filesystem::path bundle;
  std::filesystem::path repository_root = std::filesystem::current_path();
  std::filesystem::path scene;
  std::filesystem::path policy;
  long long steps = 0;
  double status_interval_seconds = 1.0;
  bool realtime = true;
  bool viewer = false;
};

constexpr std::array<const char*, 2> kTasks{{"zgwsarm_compliance", "b1_z1_ik"}};

bool supported_task(const std::string& task) {
  return std::find(kTasks.begin(), kTasks.end(), task) != kTasks.end();
}

void print_tasks() {
  for (const char* task : kTasks) std::cout << task << '\n';
}

void print_usage() {
  std::cout
      << "usage: mujoco_sim --task TASK [--config FILE]\n"
         "                  [--bundle DIR] [--repo-root DIR] [--scene XML]\n"
         "                  [--policy PT] [--steps N] [--status-interval SEC]\n"
         "                  [--viewer|--headless] [--realtime|--no-realtime]\n"
         "       mujoco_sim --list-tasks\n\n"
         "The default task config is config/<task>.yaml. Command-line arguments\n"
         "override values from that YAML file.\n";
}

std::filesystem::path task_config(const char* executable, const std::string& task) {
  const std::string filename = task + ".yaml";
  auto candidate = std::filesystem::current_path() / "mujoco/config" / filename;
  if (std::filesystem::is_regular_file(candidate)) return candidate;
  std::error_code error;
  const auto executable_path = std::filesystem::weakly_canonical(executable, error);
  if (!error) {
    candidate = executable_path.parent_path().parent_path() / "config" / filename;
    if (std::filesystem::is_regular_file(candidate)) return candidate;
  }
  return {};
}

Options parse(int argc, char** argv) {
  Options options;
  std::filesystem::path config_path;
  std::string task;
  bool help = false;
  bool list_tasks = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--task" || argument == "--config") {
      if (++index >= argc) throw std::runtime_error("Missing value after " + argument);
      if (argument == "--task") task = argv[index];
      else config_path = argv[index];
    } else if (argument == "--help") {
      help = true;
    } else if (argument == "--list-tasks") {
      list_tasks = true;
    }
  }
  if (list_tasks) {
    print_tasks();
    std::exit(0);
  }
  if (help) {
    print_usage();
    std::exit(0);
  }
  if (task.empty())
    throw std::runtime_error("--task is required; use --list-tasks to see valid tasks");
  if (!supported_task(task)) {
    throw std::runtime_error("Unknown task '" + task +
                             "'; valid tasks are zgwsarm_compliance and b1_z1_ik");
  }
  if (config_path.empty()) {
    config_path = task_config(argv[0], task);
    if (config_path.empty())
      throw std::runtime_error("Cannot find config/" + task + ".yaml");
  }
  options.config = mujoco::SimulatorConfig::Load(config_path);
  if (options.config->task_name != task) {
    throw std::runtime_error("--task '" + task + "' does not match YAML task_name '" +
                             options.config->task_name + "'");
  }
  const auto& config = *options.config;
  options.bundle = config.deployment_bundle;
  options.repository_root = config.repository_root;
  options.scene = config.scene_path;
  options.policy = config.policy_path;
  options.steps = config.steps;
  options.status_interval_seconds = config.status_interval_seconds;
  options.realtime = config.realtime;
  options.viewer = config.viewer;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto next = [&]() -> std::string {
      if (++index >= argc) throw std::runtime_error("Missing value after " + argument);
      return argv[index];
    };
    if (argument == "--task" || argument == "--config") next();
    else if (argument == "--bundle") options.bundle = next();
    else if (argument == "--repo-root") options.repository_root = next();
    else if (argument == "--scene") options.scene = next();
    else if (argument == "--policy") options.policy = next();
    else if (argument == "--steps") options.steps = std::stoll(next());
    else if (argument == "--status-interval")
      options.status_interval_seconds = std::stod(next());
    else if (argument == "--no-realtime") options.realtime = false;
    else if (argument == "--realtime") options.realtime = true;
    else if (argument == "--viewer") options.viewer = true;
    else if (argument == "--headless") options.viewer = false;
    else if (argument == "--help" || argument == "--list-tasks") continue;
    else throw std::runtime_error("Unknown option: " + argument);
  }
  if (options.bundle.empty())
    throw std::runtime_error("No deployment bundle: use --config or --bundle");
  if (options.steps < 0 || options.status_interval_seconds < 0.0)
    throw std::runtime_error("steps and status interval must be nonnegative");
  return options;
}
}  // namespace

int run(int argc, char** argv) {
  try {
    const auto options = parse(argc, argv);
    auto profile = mujoco::TaskProfile::Load(options.bundle, options.repository_root);
    if (options.config) options.config->apply_profile_overrides(profile);
    if (!options.scene.empty()) profile.model_path = std::filesystem::absolute(options.scene);
    if (!options.policy.empty()) profile.policy_path = std::filesystem::absolute(options.policy);
    mujoco::PolicyRunner policy(profile.policy_path, profile.policy_input_dim,
                                profile.policy_output_dim);
    mujoco::MujocoSimulator simulation(profile, options.viewer);
    mujoco::TaskController controller(profile);
    mujoco::CommandState commands(profile);
    mujoco::ObservationBuilder observation(profile);
    mujoco::ObservationHistory history(profile.frame_dim, profile.history_length);
    mujoco::Gamepad gamepad;
    std::vector<float> action(static_cast<std::size_t>(profile.policy_output_dim), 0.0f);

    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    if (options.config) std::cout << "config=" << options.config->source_path << '\n';
    std::cout << "task=" << profile.task_name
              << " contract_checkpoint=" << profile.checkpoint_number
              << " control_hz=" << 1.0 / profile.control_dt() << '\n'
              << "scene=" << profile.model_path << '\n'
              << "policy=" << profile.policy_path << '\n'
              << "gamepad=" << gamepad.status() << '\n'
              << "A=fold/stand PD (" << profile.startup_fold_duration << " s + "
              << profile.startup_stand_duration << " s), B=RL/policy takeover, "
                 "X=position/force, Y=reset/dog zero + arm hold\n"
              << "RB/LB=position radius or force Fz; RT/LT/Start/Back=unused\n"
              << "arm_q_order=";
    for (std::size_t index = 0; index < profile.arm_dof_names.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << profile.arm_dof_names[index];
    }
    std::cout << '\n';
    if (profile.direct_gripper_dof.empty()) {
      std::cout << "gripper=unavailable (this task/model has no actuated gripper DOF)\n";
    } else {
      std::cout << "gripper=" << profile.direct_gripper_dof
                << " (D-pad up/down)\n";
    }

    long long control_steps = 0;
    std::exception_ptr control_error;
    const auto control_loop = [&] {
      try {
        // The official simulate UI owns the main thread.  This initial Load
        // attaches the user-owned model/data after RenderLoop has started.
        simulation.render();
        auto wall_deadline = std::chrono::steady_clock::now();
        auto last_report = wall_deadline;
        while (running && !simulation.viewer_should_close() &&
               (options.steps <= 0 || control_steps < options.steps)) {
          const auto event = commands.update(gamepad.poll(), profile.control_dt());
          if (event.request_reset) {
            simulation.reset();
            controller.reset();
            history.reset();
            std::fill(action.begin(), action.end(), 0.0f);
          } else if (event.request_standup) {
            controller.start_standup(simulation.state());
            history.reset();
            std::fill(action.begin(), action.end(), 0.0f);
          } else if (event.request_rl) {
            if (controller.start_rl(simulation.state())) {
              history.reset();
              std::fill(action.begin(), action.end(), 0.0f);
            } else {
              std::cout << "B ignored: wait until control_state=standby after A completes\n";
            }
          }
          if (simulation.viewer_paused()) {
            simulation.render();
            wall_deadline = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
          }

          if (controller.policy_active())
            action = controller.prepare_action(policy.infer(history.values()));
          else
            std::fill(action.begin(), action.end(), 0.0f);
          for (int substep = 0; substep < profile.decimation; ++substep) {
            const auto state = simulation.state();
            const auto control = controller.compute(action, state, commands);
            const auto safety = mujoco::CheckSafety(profile, state, control.torque);
            if (!safety.safe) throw std::runtime_error("Safety stop: " + safety.reason);
            simulation.step(control.torque);
          }
          commands.advance_clock(profile.control_dt());
          history.append(observation.frame(simulation.state(), commands.values(), action,
                                           commands.clock()));
          ++control_steps;
          simulation.render();

          const auto now = std::chrono::steady_clock::now();
          if (options.status_interval_seconds > 0.0 &&
              now - last_report >=
                  std::chrono::duration<double>(options.status_interval_seconds)) {
            const auto& command = commands.values();
            const auto robot_state = simulation.state();
            const auto ee = simulation.end_effector_debug_state();
            const double command_radius = command[15];
            const double command_pitch = command[16];
            const double command_yaw = command[17];
            const std::array<double, 3> command_arm_position{{
                command_radius * std::cos(command_pitch) * std::cos(command_yaw),
                command_radius * std::cos(command_pitch) * std::sin(command_yaw),
                -command_radius * std::sin(command_pitch),
            }};
            std::cout << std::fixed << std::setprecision(3)
                      << "t=" << simulation.time() << " mode="
                      << (commands.force_mode() ? "force" : "position")
                      << " control_state=" << mujoco::RobotControlModeName(controller.mode())
                      << " gamepad=" << commands.input_active()
                      << " cmd(vx,yaw)=" << command[0] << ',' << command[2] << '\n'
                      << "  ee_cmd_sph(r,p,y)=(" << command_radius << ','
                      << command_pitch << ',' << command_yaw << ")"
                      << " ee_actual_sph=(" << ee.arm_spherical[0] << ','
                      << ee.arm_spherical[1] << ',' << ee.arm_spherical[2] << ")\n"
                      << "  ee_cmd_arm_xyz=(" << command_arm_position[0] << ','
                      << command_arm_position[1] << ',' << command_arm_position[2] << ")"
                      << " ee_actual_arm_xyz=(" << ee.arm_position[0] << ','
                      << ee.arm_position[1] << ',' << ee.arm_position[2] << ")\n"
                      << "  ee_force_cmd_xyz=(" << command[12] << ',' << command[13]
                      << ',' << command[14] << ")"
                      << " ee_contact_N=" << simulation.end_effector_contact_force() << '\n'
                      << "  arm_q(rad)=(";
            for (std::size_t index = 0; index < profile.arm_dof_names.size(); ++index) {
              if (index != 0) std::cout << ',';
              std::cout << robot_state.joint_position[
                  profile.dof_index(profile.arm_dof_names[index])];
            }
            std::cout << ')';
            if (commands.has_wrist_target())
              std::cout << " wrist_target=" << commands.wrist_target();
            if (commands.has_gripper_target())
              std::cout << " gripper_target=" << commands.gripper_target();
            std::cout << '\n';
            last_report = now;
          }
          if (options.realtime) {
            wall_deadline += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(profile.control_dt()));
            std::this_thread::sleep_until(wall_deadline);
          }
        }
      } catch (...) {
        control_error = std::current_exception();
      }
      simulation.request_viewer_close();
    };

    if (simulation.viewer_enabled()) {
      std::thread control_thread(control_loop);
      simulation.viewer_loop();
      control_thread.join();
    } else {
      control_loop();
    }
    if (control_error) std::rethrow_exception(control_error);
    std::cout << "simulation stopped after " << control_steps << " control steps\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "mujoco_sim failed: " << error.what() << '\n';
    return 1;
  }
}

int main(int argc, char** argv) {
  const int result = run(argc, argv);
  std::cout.flush();
  std::cerr.flush();
#ifdef MUJOCO_TORCH_PYTHON_PACKAGE
  // PyTorch 2.0 CUDA Python wheels statically contain LLVM command-line
  // globals.  Mesa's OpenGL driver can load another LLVM copy for the GLFW
  // viewer, and their process-global destructors corrupt each other after all
  // of our objects have already shut down cleanly.  Skip only third-party
  // global destructors for this fallback linkage; official LibTorch builds
  // retain the normal return path.
  std::_Exit(result);
#else
  return result;
#endif
}
