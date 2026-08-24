#include "mujoco/policy.hpp"

#include <torch/script.h>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace mujoco {

struct PolicyRunner::Impl {
  torch::jit::script::Module module;
  int input_dim;
  int output_dim;

  Impl(const std::filesystem::path& path, int input, int output)
      : module(torch::jit::load(path.string(), torch::kCPU)), input_dim(input), output_dim(output) {
    module.eval();
  }
};

PolicyRunner::PolicyRunner(const std::filesystem::path& path, int input_dim, int output_dim)
    : impl_(std::make_unique<Impl>(path, input_dim, output_dim)) {}
PolicyRunner::~PolicyRunner() = default;
PolicyRunner::PolicyRunner(PolicyRunner&&) noexcept = default;
PolicyRunner& PolicyRunner::operator=(PolicyRunner&&) noexcept = default;

std::vector<float> PolicyRunner::infer(const std::vector<float>& observation) {
  if (observation.size() != static_cast<std::size_t>(impl_->input_dim))
    throw std::runtime_error("Policy observation has the wrong dimension");
  torch::NoGradGuard guard;
  auto input = torch::from_blob(const_cast<float*>(observation.data()),
                                {1, impl_->input_dim}, torch::TensorOptions().dtype(torch::kFloat32));
  auto output = impl_->module.forward({input}).toTensor().to(torch::kCPU).contiguous().view({-1});
  if (output.numel() != impl_->output_dim)
    throw std::runtime_error("TorchScript policy returned the wrong action dimension");
  std::vector<float> result(static_cast<std::size_t>(impl_->output_dim));
  const float* values = output.data_ptr<float>();
  for (int index = 0; index < impl_->output_dim; ++index) {
    if (!std::isfinite(values[index])) throw std::runtime_error("TorchScript policy returned a non-finite action");
    result[static_cast<std::size_t>(index)] = values[index];
  }
  return result;
}

std::vector<float> ReadFloat32File(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("Cannot open float32 file: " + path.string());
  const auto bytes = stream.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
    throw std::runtime_error("Invalid float32 file length: " + path.string());
  std::vector<float> values(static_cast<std::size_t>(bytes / sizeof(float)));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(values.data()), bytes);
  if (!stream) throw std::runtime_error("Cannot read float32 file: " + path.string());
  return values;
}

}  // namespace mujoco

