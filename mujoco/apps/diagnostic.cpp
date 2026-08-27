#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mujoco/command.hpp"
#include "mujoco/control.hpp"
#include "mujoco/launcher_config.hpp"
#include "mujoco/observation.hpp"
#include "mujoco/policy.hpp"
#include "mujoco/runtime_config.hpp"
#include "mujoco/simulator.hpp"

namespace {

volatile std::sig_atomic_t running = 1;
void stop(int) { running = 0; }

struct Options {
  std::filesystem::path config;
  std::filesystem::path bundle;
  std::filesystem::path repository_root;
  std::filesystem::path scene;
  std::filesystem::path policy;
  std::string scenario;
  int force_axis = 2;
  std::vector<double> force_values{0.0, 5.0, 10.0, -5.0};
  std::vector<double> yaw_values{0.2, -0.2};
  double segment_seconds = 5.0;
  double settle_seconds = 2.0;
  double ramp_seconds = 1.0;
  double report_seconds = 0.5;
  double stiffness = 200.0;
  double damping = 20.0;
  double force_limit = 30.0;
  bool viewer = false;
  bool realtime = false;
};

struct ForceStatistics {
  long long samples = 0;
  std::array<double, 3> actual_sum{{0.0, 0.0, 0.0}};
  std::array<double, 3> absolute_error_sum{{0.0, 0.0, 0.0}};
  std::array<long long, 3> saturated{{0, 0, 0}};
  long long any_saturated = 0;
  double displacement_sum = 0.0;
  double displacement_max = 0.0;
};

struct WheelSample {
  double left_speed = 0.0;
  double right_speed = 0.0;
  double yaw_hat = 0.0;
};

struct YawStatistics {
  long long samples = 0;
  double actual_sum = 0.0;
  double absolute_error_sum = 0.0;
  double left_sum = 0.0;
  double right_sum = 0.0;
  double yaw_hat_sum = 0.0;
};

void usage() {
  std::cout
      << "usage: mujoco_diagnostic --scenario force|yaw [options]\n\n"
         "Common options:\n"
         "  --config FILE       launcher YAML (default: config/zgwsarm_compliance.yaml)\n"
         "  --bundle DIR        override deployment bundle\n"
         "  --policy FILE       override exported TorchScript policy\n"
         "  --scene FILE        override scene (default: controlled flat scene)\n"
         "  --segment-seconds S duration of each target (default: 5)\n"
         "  --settle-seconds S  zero-command settling before sequence (default: 2)\n"
         "  --report-seconds S  concise status period; 0 disables (default: 0.5)\n"
         "  --viewer|--headless show or hide the official viewer (default: headless)\n"
         "  --realtime|--no-realtime wall-clock pacing (default: no-realtime)\n\n"
         "Force options:\n"
         "  --axis x|y|z        commanded force axis (default: z)\n"
         "  --values CSV        targets in N (default: 0,5,10,-5)\n"
         "  --ramp-seconds S    linear transition time (default: 1)\n"
         "  --stiffness K       virtual spring K in N/m (default: 200)\n"
         "  --damping D         virtual damping D in N*s/m (default: 20)\n"
         "  --force-limit N     world-axis spring clamp (default: 30)\n\n"
         "Yaw options:\n"
         "  --yaw-values CSV    targets in rad/s (default: 0.2,-0.2)\n";
}

std::vector<double> parse_csv(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) throw std::runtime_error("Empty value in CSV sequence");
    values.push_back(std::stod(token));
  }
  if (values.empty()) throw std::runtime_error("Command sequence cannot be empty");
  return values;
}

std::filesystem::path default_config(const char* executable) {
  auto candidate = std::filesystem::current_path() /
                   "mujoco/config/zgwsarm_compliance.yaml";
  if (std::filesystem::is_regular_file(candidate)) return candidate;
  std::error_code error;
  const auto executable_path =
      std::filesystem::weakly_canonical(executable, error);
  if (!error) {
    candidate = executable_path.parent_path().parent_path() /
                "config/zgwsarm_compliance.yaml";
    if (std::filesystem::is_regular_file(candidate)) return candidate;
  }
  throw std::runtime_error("Cannot find config/zgwsarm_compliance.yaml");
}

Options parse(int argc, char** argv) {
  Options options;
  options.config = default_config(argv[0]);
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto next = [&]() -> std::string {
      if (++index >= argc)
        throw std::runtime_error("Missing value after " + argument);
      return argv[index];
    };
    if (argument == "--help") {
      usage();
      std::exit(0);
    } else if (argument == "--config") options.config = next();
  }

  auto config = mujoco::SimulatorConfig::Load(options.config);
  if (config.task_name != "zgwsarm_compliance")
    throw std::runtime_error("mujoco_diagnostic currently supports zgwsarm_compliance only");
  options.bundle = config.deployment_bundle;
  options.repository_root = config.repository_root;
  options.scene = options.repository_root / "mujoco/models/zgwsarm/scene_flat.xml";
  options.policy = config.policy_path;

  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto next = [&]() -> std::string {
      if (++index >= argc)
        throw std::runtime_error("Missing value after " + argument);
      return argv[index];
    };
    if (argument == "--help") continue;
    if (argument == "--config") next();
    else if (argument == "--scenario") options.scenario = next();
    else if (argument == "--bundle") options.bundle = next();
    else if (argument == "--policy") options.policy = next();
    else if (argument == "--scene") options.scene = next();
    else if (argument == "--axis") {
      const std::string axis = next();
      if (axis == "x") options.force_axis = 0;
      else if (axis == "y") options.force_axis = 1;
      else if (axis == "z") options.force_axis = 2;
      else throw std::runtime_error("--axis must be x, y, or z");
    } else if (argument == "--values") options.force_values = parse_csv(next());
    else if (argument == "--yaw-values") options.yaw_values = parse_csv(next());
    else if (argument == "--segment-seconds") options.segment_seconds = std::stod(next());
    else if (argument == "--settle-seconds") options.settle_seconds = std::stod(next());
    else if (argument == "--ramp-seconds") options.ramp_seconds = std::stod(next());
    else if (argument == "--report-seconds") options.report_seconds = std::stod(next());
    else if (argument == "--stiffness") options.stiffness = std::stod(next());
    else if (argument == "--damping") options.damping = std::stod(next());
    else if (argument == "--force-limit") options.force_limit = std::stod(next());
    else if (argument == "--viewer") options.viewer = true;
    else if (argument == "--headless") options.viewer = false;
    else if (argument == "--realtime") options.realtime = true;
    else if (argument == "--no-realtime") options.realtime = false;
    else throw std::runtime_error("Unknown option: " + argument);
  }
  if (options.scenario != "force" && options.scenario != "yaw")
    throw std::runtime_error("--scenario must be force or yaw");
  if (options.segment_seconds <= 0.0 || options.settle_seconds < 0.0 ||
      options.ramp_seconds < 0.0 ||
      (options.scenario == "force" &&
       options.ramp_seconds > options.segment_seconds) ||
      options.report_seconds < 0.0 || options.stiffness <= 0.0 ||
      options.damping < 0.0 || options.force_limit <= 0.0)
    throw std::runtime_error("Invalid diagnostic timing or force-field parameter");
  return options;
}

double norm(const std::array<double, 3>& value) {
  return std::sqrt(value[0] * value[0] + value[1] * value[1] +
                   value[2] * value[2]);
}

double base_yaw(const mujoco::RobotState& state) {
  const double w = state.base_quaternion[0];
  const double x = state.base_quaternion[1];
  const double y = state.base_quaternion[2];
  const double z = state.base_quaternion[3];
  return std::atan2(2.0 * (w * z + x * y),
                    1.0 - 2.0 * (y * y + z * z));
}

std::array<double, 3> world_to_yaw(
    const std::array<double, 3>& value, const mujoco::RobotState& state) {
  const double yaw = base_yaw(state);
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  return {{cosine * value[0] + sine * value[1],
           -sine * value[0] + cosine * value[1], value[2]}};
}

void print_vector(const std::array<double, 3>& value) {
  std::cout << '(' << value[0] << ',' << value[1] << ',' << value[2] << ')';
}

WheelSample wheel_sample(const mujoco::TaskProfile& profile,
                         const mujoco::MujocoSimulator& simulation,
                         const mujoco::RobotState& state) {
  const auto positions = simulation.wheel_positions_base();
  if (positions.size() != profile.wheel_dof_names.size() || positions.empty())
    throw std::runtime_error("Wheel diagnostic dimensions are inconsistent");
  std::vector<double> y;
  std::vector<double> speeds;
  y.reserve(positions.size());
  speeds.reserve(positions.size());
  for (std::size_t index = 0; index < positions.size(); ++index) {
    y.push_back(positions[index][1]);
    speeds.push_back(state.joint_velocity[
                         profile.dof_index(profile.wheel_dof_names[index])] *
                     profile.wheel_radius);
  }
  const double y_mean =
      std::accumulate(y.begin(), y.end(), 0.0) / y.size();
  const double speed_mean =
      std::accumulate(speeds.begin(), speeds.end(), 0.0) / speeds.size();
  double numerator = 0.0;
  double denominator = 1.0e-6;
  double left_sum = 0.0;
  double right_sum = 0.0;
  int left_count = 0;
  int right_count = 0;
  for (std::size_t index = 0; index < y.size(); ++index) {
    const double dy = y[index] - y_mean;
    numerator += dy * (speeds[index] - speed_mean);
    denominator += dy * dy;
    if (y[index] > y_mean) {
      left_sum += speeds[index];
      ++left_count;
    } else {
      right_sum += speeds[index];
      ++right_count;
    }
  }
  return {left_sum / std::max(left_count, 1),
          right_sum / std::max(right_count, 1), -numerator / denominator};
}

void advance_control_step(const mujoco::TaskProfile& profile,
                          mujoco::MujocoSimulator& simulation,
                          mujoco::TaskController& controller,
                          mujoco::CommandState& commands,
                          mujoco::PolicyRunner& policy,
                          mujoco::ObservationBuilder& observation,
                          mujoco::ObservationHistory& history,
                          std::vector<float>& action,
                          const std::function<void()>& after_substep = {}) {
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
    if (after_substep) after_substep();
  }
  commands.advance_clock(profile.control_dt());
  history.append(observation.frame(simulation.state(), commands.values(), action,
                                   commands.clock()));
  simulation.render();
}

void stand_and_enable_policy(const mujoco::TaskProfile& profile,
                             mujoco::MujocoSimulator& simulation,
                             mujoco::TaskController& controller,
                             mujoco::CommandState& commands,
                             mujoco::PolicyRunner& policy,
                             mujoco::ObservationBuilder& observation,
                             mujoco::ObservationHistory& history,
                             std::vector<float>& action) {
  controller.start_standup(simulation.state());
  while (running && controller.mode() != mujoco::RobotControlMode::kStandby)
    advance_control_step(profile, simulation, controller, commands, policy,
                         observation, history, action);
  if (!running || !controller.start_rl(simulation.state()))
    throw std::runtime_error("Could not enter RL control after automatic stand-up");
  history.reset();
  std::fill(action.begin(), action.end(), 0.0f);
}

void pace(bool realtime, std::chrono::steady_clock::time_point& deadline,
          double dt) {
  if (!realtime) return;
  deadline += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(dt));
  std::this_thread::sleep_until(deadline);
}

void print_force_summary(std::size_t segment, double target,
                         const ForceStatistics& stats) {
  const double count = std::max<long long>(stats.samples, 1);
  std::cout << "FORCE SUMMARY segment=" << segment << " target=" << target
            << " N  mean_actual_yaw=(";
  for (int axis = 0; axis < 3; ++axis) {
    if (axis) std::cout << ',';
    std::cout << stats.actual_sum[axis] / count;
  }
  std::cout << ")  mae_xyz=(";
  for (int axis = 0; axis < 3; ++axis) {
    if (axis) std::cout << ',';
    std::cout << stats.absolute_error_sum[axis] / count;
  }
  std::cout << ") N  displacement_mean/max="
            << stats.displacement_sum / count << '/' << stats.displacement_max
            << " m  saturation_world_xyz=(";
  for (int axis = 0; axis < 3; ++axis) {
    if (axis) std::cout << ',';
    std::cout << 100.0 * stats.saturated[axis] / count;
  }
  std::cout << ")% any=" << 100.0 * stats.any_saturated / count << "%\n";
}

void print_yaw_summary(std::size_t segment, double target,
                       const YawStatistics& stats) {
  const double count = std::max<long long>(stats.samples, 1);
  const double actual = stats.actual_sum / count;
  const double left = stats.left_sum / count;
  const double right = stats.right_sum / count;
  const double yaw_hat = stats.yaw_hat_sum / count;
  bool direction_ok = true;
  if (target > 0.0)
    direction_ok = left < 0.0 && right > 0.0 && yaw_hat > 0.0 && actual > 0.0;
  else if (target < 0.0)
    direction_ok = left > 0.0 && right < 0.0 && yaw_hat < 0.0 && actual < 0.0;
  std::cout << "YAW SUMMARY segment=" << segment << " target=" << target
            << " rad/s  actual=" << actual
            << "  mae=" << stats.absolute_error_sum / count
            << "  left/right=" << left << '/' << right << " m/s"
            << "  wheel_yaw_hat=" << yaw_hat
            << "  direction=" << (direction_ok ? "PASS" : "FAIL") << '\n';
}

int execute(const Options& options) {
  auto launcher = mujoco::SimulatorConfig::Load(options.config);
  auto profile = mujoco::TaskProfile::Load(options.bundle,
                                           options.repository_root);
  launcher.apply_profile_overrides(profile);
  profile.model_path = std::filesystem::absolute(options.scene);
  profile.policy_path = std::filesystem::absolute(options.policy);
  mujoco::PolicyRunner policy(profile.policy_path, profile.policy_input_dim,
                              profile.policy_output_dim);
  mujoco::MujocoSimulator simulation(profile, options.viewer);
  mujoco::TaskController controller(profile);
  mujoco::CommandState commands(profile);
  mujoco::ObservationBuilder observation(profile);
  mujoco::ObservationHistory history(profile.frame_dim, profile.history_length);
  std::vector<float> action(static_cast<std::size_t>(profile.policy_output_dim),
                            0.0f);

  std::cout << std::fixed << std::setprecision(3)
            << "diagnostic=" << options.scenario
            << " contract_checkpoint=" << profile.checkpoint_number
            << " scene=" << profile.model_path << '\n'
            << "policy=" << profile.policy_path << '\n';
  if (options.scenario == "force")
    std::cout << "force field: robot_relative_static K=" << options.stiffness
              << " D=" << options.damping
              << " world_axis_limit=" << options.force_limit
              << " N axis=" << "xyz"[options.force_axis]
              << " ramp=" << options.ramp_seconds << " s\n";

  std::exception_ptr error;
  const auto experiment = [&] {
    try {
      simulation.render();
      commands.set_scripted_value(0, 0.0f);
      commands.set_scripted_value(1, 0.0f);
      commands.set_scripted_value(2, 0.0f);
      commands.set_scripted_force_mode(options.scenario == "force");
      stand_and_enable_policy(profile, simulation, controller, commands, policy,
                              observation, history, action);
      if (options.scenario == "force")
        simulation.start_end_effector_force_field(
            options.stiffness, options.damping, options.force_limit,
            "robot_relative_static");

      auto deadline = std::chrono::steady_clock::now();
      const long long settle_steps = static_cast<long long>(
          std::ceil(options.settle_seconds / profile.control_dt()));
      for (long long step = 0; running && step < settle_steps; ++step) {
        advance_control_step(profile, simulation, controller, commands, policy,
                             observation, history, action);
        pace(options.realtime, deadline, profile.control_dt());
      }
      std::cout << "settling complete; scripted command sequence starts now\n";

      double previous_force_target = 0.0;
      const auto& sequence = options.scenario == "force"
                                 ? options.force_values
                                 : options.yaw_values;
      for (std::size_t segment = 0; running && segment < sequence.size();
           ++segment) {
        const double target = sequence[segment];
        const long long control_steps = static_cast<long long>(
            std::ceil(options.segment_seconds / profile.control_dt()));
        ForceStatistics force_stats;
        YawStatistics yaw_stats;
        double next_report = simulation.time();
        for (long long step = 0; running && step < control_steps; ++step) {
          const double elapsed = step * profile.control_dt();
          if (options.scenario == "force") {
            const double alpha = options.ramp_seconds == 0.0
                                     ? 1.0
                                     : std::min(1.0, elapsed / options.ramp_seconds);
            const double command = previous_force_target +
                                   alpha * (target - previous_force_target);
            for (int axis = 0; axis < 3; ++axis)
              commands.set_scripted_value(12 + axis,
                  static_cast<float>(axis == options.force_axis ? command : 0.0));
          } else {
            commands.set_scripted_value(0, 0.0f);
            commands.set_scripted_value(2, static_cast<float>(target));
          }

          const auto sample = [&] {
            const auto state = simulation.state();
            if (options.scenario == "force") {
              const auto debug =
                  simulation.end_effector_spring_force_debug_state();
              const auto actual = world_to_yaw(debug.clipped_world, state);
              std::array<double, 3> command{{
                  commands.values()[12], commands.values()[13],
                  commands.values()[14]}};
              ++force_stats.samples;
              bool any_saturated = false;
              for (int axis = 0; axis < 3; ++axis) {
                force_stats.actual_sum[axis] += actual[axis];
                force_stats.absolute_error_sum[axis] +=
                    std::abs(actual[axis] - command[axis]);
                const bool saturated =
                    std::abs(debug.unclipped_world[axis]) >=
                    options.force_limit - 1.0e-9;
                force_stats.saturated[axis] += saturated;
                any_saturated = any_saturated || saturated;
              }
              force_stats.any_saturated += any_saturated;
              const double displacement = norm(debug.displacement_world);
              force_stats.displacement_sum += displacement;
              force_stats.displacement_max =
                  std::max(force_stats.displacement_max, displacement);
            } else {
              const auto wheel = wheel_sample(profile, simulation, state);
              const double actual = simulation.base_yaw_rate();
              ++yaw_stats.samples;
              yaw_stats.actual_sum += actual;
              yaw_stats.absolute_error_sum += std::abs(actual - target);
              yaw_stats.left_sum += wheel.left_speed;
              yaw_stats.right_sum += wheel.right_speed;
              yaw_stats.yaw_hat_sum += wheel.yaw_hat;
            }
          };
          advance_control_step(profile, simulation, controller, commands, policy,
                               observation, history, action, sample);

          if (options.report_seconds > 0.0 &&
              simulation.time() + 1.0e-9 >= next_report) {
            if (options.scenario == "force") {
              const auto debug =
                  simulation.end_effector_spring_force_debug_state();
              const auto actual = world_to_yaw(debug.clipped_world,
                                               simulation.state());
              std::array<double, 3> command{{commands.values()[12],
                                             commands.values()[13],
                                             commands.values()[14]}};
              std::array<double, 3> absolute_error{{
                  std::abs(actual[0] - command[0]),
                  std::abs(actual[1] - command[1]),
                  std::abs(actual[2] - command[2])}};
              std::cout << "force t=" << simulation.time()
                        << " segment=" << segment << " cmd=";
              print_vector(command);
              std::cout << " actual_yaw=";
              print_vector(actual);
              std::cout << " abs_error=";
              print_vector(absolute_error);
              std::cout << " displacement=" << norm(debug.displacement_world)
                        << " raw_world=";
              print_vector(debug.unclipped_world);
              std::cout << " clipped_world=";
              print_vector(debug.clipped_world);
              std::cout << '\n';
            } else {
              const auto state = simulation.state();
              const auto wheel = wheel_sample(profile, simulation, state);
              const double actual = simulation.base_yaw_rate();
              std::cout << "yaw t=" << simulation.time()
                        << " segment=" << segment << " cmd=" << target
                        << " actual=" << actual
                        << " abs_error=" << std::abs(actual - target)
                        << " left/right=" << wheel.left_speed << '/'
                        << wheel.right_speed
                        << " wheel_yaw_hat=" << wheel.yaw_hat << '\n';
            }
            next_report += options.report_seconds;
          }
          pace(options.realtime, deadline, profile.control_dt());
        }
        if (options.scenario == "force") {
          print_force_summary(segment, target, force_stats);
          previous_force_target = target;
        } else {
          print_yaw_summary(segment, target, yaw_stats);
        }
      }
    } catch (...) {
      error = std::current_exception();
    }
    simulation.request_viewer_close();
  };

  if (simulation.viewer_enabled()) {
    std::thread control_thread(experiment);
    simulation.viewer_loop();
    control_thread.join();
  } else {
    experiment();
  }
  if (error) std::rethrow_exception(error);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, stop);
  std::signal(SIGTERM, stop);
  int result = 0;
  try {
    result = execute(parse(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "mujoco_diagnostic failed: " << error.what() << '\n';
    result = 1;
  }
  std::cout.flush();
  std::cerr.flush();
#ifdef MUJOCO_TORCH_PYTHON_PACKAGE
  std::_Exit(result);
#else
  return result;
#endif
}
