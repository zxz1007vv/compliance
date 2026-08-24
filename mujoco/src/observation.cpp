#include "mujoco/observation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mujoco {

std::array<float, 3> ProjectedGravity(const std::array<double, 4>& q) {
  const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (!std::isfinite(norm) || norm < 1e-12) throw std::runtime_error("Invalid base quaternion");
  const double w = q[0] / norm;
  const double x = q[1] / norm;
  const double y = q[2] / norm;
  const double z = q[3] / norm;
  // R(q)^T * [0, 0, -1], matching Isaac Gym quat_rotate_inverse.
  return {{static_cast<float>(2.0 * (w * y - x * z)),
           static_cast<float>(-2.0 * (w * x + y * z)),
           static_cast<float>(-1.0 + 2.0 * (x * x + y * y))}};
}

std::vector<float> ObservationBuilder::frame(
    const RobotState& state, const std::vector<float>& commands,
    const std::vector<float>& action, const std::array<float, 4>& clock) const {
  const auto n = static_cast<std::size_t>(profile_.policy_output_dim);
  if (state.joint_position.size() != n || state.joint_velocity.size() != n ||
      action.size() != n || commands.size() != 23) {
    throw std::runtime_error("Observation input dimensions do not match task profile");
  }
  std::vector<float> output;
  output.reserve(profile_.frame_dim);
  const auto gravity = ProjectedGravity(state.base_quaternion);
  output.insert(output.end(), gravity.begin(), gravity.end());
  for (std::size_t index = 0; index < commands.size(); ++index) {
    float value = commands[index];
    if (commands[22] >= 0.5f && index >= 15 && index <= 17) value = 0.0f;
    output.push_back(value * static_cast<float>(profile_.command_scales[index]));
  }
  for (std::size_t index = 0; index < n; ++index) {
    const bool zeroed = std::find(profile_.zero_position_dof_indices.begin(),
                                  profile_.zero_position_dof_indices.end(),
                                  static_cast<int>(index)) != profile_.zero_position_dof_indices.end();
    const double error = zeroed ? 0.0 : state.joint_position[index] - profile_.default_dof_positions[index];
    output.push_back(static_cast<float>(error * profile_.dof_position_scale));
  }
  for (double velocity : state.joint_velocity)
    output.push_back(static_cast<float>(velocity * profile_.dof_velocity_scale));
  output.insert(output.end(), action.begin(), action.end());
  output.insert(output.end(), clock.begin(), clock.end());
  if (output.size() != static_cast<std::size_t>(profile_.frame_dim))
    throw std::runtime_error("Built observation frame has the wrong dimension");
  const float clip = static_cast<float>(profile_.observation_clip);
  for (float& value : output) value = std::clamp(value, -clip, clip);
  return output;
}

ObservationHistory::ObservationHistory(int frame_dim, int history_length)
    : frame_dim_(frame_dim), history_length_(history_length),
      values_(static_cast<std::size_t>(frame_dim * history_length), 0.0f) {
  if (frame_dim <= 0 || history_length <= 0) throw std::runtime_error("Invalid observation history shape");
}

void ObservationHistory::reset() { std::fill(values_.begin(), values_.end(), 0.0f); }

void ObservationHistory::append(const std::vector<float>& frame) {
  if (frame.size() != static_cast<std::size_t>(frame_dim_))
    throw std::runtime_error("Observation history received a frame with the wrong size");
  std::move(values_.begin() + frame_dim_, values_.end(), values_.begin());
  std::copy(frame.begin(), frame.end(), values_.end() - frame_dim_);
}

}  // namespace mujoco

