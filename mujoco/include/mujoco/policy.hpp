#pragma once

#include <filesystem>
#include <memory>
#include <vector>

namespace mujoco {

class PolicyRunner {
 public:
  PolicyRunner(const std::filesystem::path& path, int input_dim, int output_dim);
  ~PolicyRunner();
  PolicyRunner(PolicyRunner&&) noexcept;
  PolicyRunner& operator=(PolicyRunner&&) noexcept;
  PolicyRunner(const PolicyRunner&) = delete;
  PolicyRunner& operator=(const PolicyRunner&) = delete;
  std::vector<float> infer(const std::vector<float>& observation);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::vector<float> ReadFloat32File(const std::filesystem::path& path);

}  // namespace mujoco

