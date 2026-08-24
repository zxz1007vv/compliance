#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "mujoco/policy.hpp"
#include "mujoco/runtime_config.hpp"

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: mujoco_policy_parity BUNDLE_DIR\n";
      return 2;
    }
    const std::filesystem::path bundle = argv[1];
    mujoco::FlatConfig cfg(bundle / "runtime.cfg");
    const int input_dim = cfg.integer("policy_input_dim");
    const int output_dim = cfg.integer("policy_output_dim");
    const double tolerance = cfg.number("golden_absolute_tolerance");
    const auto inputs = mujoco::ReadFloat32File(bundle / "golden_inputs.f32");
    const auto expected = mujoco::ReadFloat32File(bundle / "golden_outputs.f32");
    if (inputs.size() % static_cast<std::size_t>(input_dim) != 0)
      throw std::runtime_error("Golden input length does not divide by policy input dimension");
    const std::size_t cases = inputs.size() / static_cast<std::size_t>(input_dim);
    if (expected.size() != cases * static_cast<std::size_t>(output_dim))
      throw std::runtime_error("Golden output shape does not match policy contract");
    mujoco::PolicyRunner policy(bundle / cfg.get("policy_file"), input_dim, output_dim);
    double maximum_error = 0.0;
    for (std::size_t test = 0; test < cases; ++test) {
      const auto first = inputs.begin() + static_cast<std::ptrdiff_t>(test * input_dim);
      std::vector<float> observation(first, first + input_dim);
      const auto actual = policy.infer(observation);
      for (int index = 0; index < output_dim; ++index) {
        maximum_error = std::max(maximum_error,
            std::abs(static_cast<double>(actual[static_cast<std::size_t>(index)] -
                expected[test * static_cast<std::size_t>(output_dim) + static_cast<std::size_t>(index)])));
      }
    }
    std::cout << "policy parity passed: cases=" << cases
              << " max_abs_error=" << maximum_error
              << " tolerance=" << tolerance << '\n';
    return maximum_error <= tolerance ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "policy parity failed: " << error.what() << '\n';
    return 1;
  }
}
