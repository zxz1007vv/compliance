#pragma once

#include <array>
#include <vector>

#include "mujoco/runtime_config.hpp"

namespace mujoco {

struct RobotState {
  // MuJoCo uses scalar-first quaternions: w, x, y, z.
  std::array<double, 4> base_quaternion{{1, 0, 0, 0}};
  std::vector<double> joint_position;
  std::vector<double> joint_velocity;
};

std::array<float, 3> ProjectedGravity(const std::array<double, 4>& quaternion);

class ObservationBuilder {
 public:
  explicit ObservationBuilder(const TaskProfile& profile) : profile_(profile) {}
  std::vector<float> frame(const RobotState& state,
                           const std::vector<float>& commands,
                           const std::vector<float>& action,
                           const std::array<float, 4>& clock) const;

 private:
  const TaskProfile& profile_;
};

class ObservationHistory {
 public:
  ObservationHistory(int frame_dim, int history_length);
  void reset();
  void append(const std::vector<float>& frame);
  const std::vector<float>& values() const { return values_; }

 private:
  int frame_dim_;
  int history_length_;
  std::vector<float> values_;
};

}  // namespace mujoco

