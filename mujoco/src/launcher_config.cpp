#include "mujoco/launcher_config.hpp"

#include <yaml.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace mujoco {
namespace {

class YamlDocument {
 public:
  explicit YamlDocument(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open YAML config: " + path.string());
    source_.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    if (!yaml_parser_initialize(&parser_))
      throw std::runtime_error("Cannot initialize the YAML parser");
    parser_initialized_ = true;
    yaml_parser_set_input_string(
        &parser_, reinterpret_cast<const unsigned char*>(source_.data()), source_.size());
    if (!yaml_parser_load(&parser_, &document_)) {
      const std::string problem = parser_.problem ? parser_.problem : "unknown YAML error";
      const std::size_t line = parser_.problem_mark.line + 1;
      yaml_parser_delete(&parser_);
      parser_initialized_ = false;
      throw std::runtime_error("Invalid YAML config " + path.string() + ":" +
                               std::to_string(line) + ": " + problem);
    }
    document_loaded_ = true;
  }

  ~YamlDocument() {
    if (document_loaded_) yaml_document_delete(&document_);
    if (parser_initialized_) yaml_parser_delete(&parser_);
  }

  yaml_document_t* get() { return &document_; }

 private:
  std::string source_;
  yaml_parser_t parser_{};
  yaml_document_t document_{};
  bool parser_initialized_ = false;
  bool document_loaded_ = false;
};

std::string scalar(yaml_node_t* node, const std::string& name) {
  if (!node || node->type != YAML_SCALAR_NODE)
    throw std::runtime_error("YAML key '" + name + "' must be a scalar");
  return std::string(reinterpret_cast<char*>(node->data.scalar.value),
                     node->data.scalar.length);
}

yaml_node_t* mapping_value(yaml_document_t* document, yaml_node_t* mapping,
                           const std::string& key, bool required = true) {
  if (!mapping || mapping->type != YAML_MAPPING_NODE)
    throw std::runtime_error("YAML parent of '" + key + "' must be a mapping");
  for (yaml_node_pair_t* pair = mapping->data.mapping.pairs.start;
       pair < mapping->data.mapping.pairs.top; ++pair) {
    yaml_node_t* key_node = yaml_document_get_node(document, pair->key);
    if (key_node && key_node->type == YAML_SCALAR_NODE && scalar(key_node, key) == key)
      return yaml_document_get_node(document, pair->value);
  }
  if (required) throw std::runtime_error("Missing YAML key: " + key);
  return nullptr;
}

yaml_node_t* mapping(yaml_document_t* document, yaml_node_t* parent,
                     const std::string& key) {
  yaml_node_t* result = mapping_value(document, parent, key);
  if (result->type != YAML_MAPPING_NODE)
    throw std::runtime_error("YAML key '" + key + "' must be a mapping");
  return result;
}

std::string optional_string(yaml_document_t* document, yaml_node_t* parent,
                            const std::string& key, const std::string& fallback) {
  yaml_node_t* node = mapping_value(document, parent, key, false);
  return node ? scalar(node, key) : fallback;
}

long long integer(yaml_document_t* document, yaml_node_t* parent,
                  const std::string& key, long long fallback, bool required = false) {
  yaml_node_t* node = mapping_value(document, parent, key, required);
  if (!node) return fallback;
  const std::string value = scalar(node, key);
  std::size_t parsed = 0;
  const long long result = std::stoll(value, &parsed);
  if (parsed != value.size()) throw std::runtime_error("Invalid integer for YAML key: " + key);
  return result;
}

double number(yaml_document_t* document, yaml_node_t* parent,
              const std::string& key, double fallback) {
  yaml_node_t* node = mapping_value(document, parent, key, false);
  if (!node) return fallback;
  const std::string value = scalar(node, key);
  std::size_t parsed = 0;
  const double result = std::stod(value, &parsed);
  if (parsed != value.size()) throw std::runtime_error("Invalid number for YAML key: " + key);
  return result;
}

bool boolean(yaml_document_t* document, yaml_node_t* parent,
             const std::string& key, bool fallback) {
  yaml_node_t* node = mapping_value(document, parent, key, false);
  if (!node) return fallback;
  std::string value = scalar(node, key);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) { return std::tolower(character); });
  if (value == "true" || value == "yes" || value == "on" || value == "1") return true;
  if (value == "false" || value == "no" || value == "off" || value == "0") return false;
  throw std::runtime_error("Invalid boolean for YAML key: " + key);
}

std::optional<double> optional_number(yaml_document_t* document,
                                      yaml_node_t* parent,
                                      const std::string& key) {
  return mapping_value(document, parent, key, false)
             ? std::optional<double>(number(document, parent, key, 0.0))
             : std::nullopt;
}

std::optional<std::vector<double>> optional_numbers(yaml_document_t* document,
                                                    yaml_node_t* parent,
                                                    const std::string& key) {
  yaml_node_t* node = mapping_value(document, parent, key, false);
  if (!node) return std::nullopt;
  if (node->type != YAML_SEQUENCE_NODE)
    throw std::runtime_error("YAML key '" + key + "' must be a sequence");
  std::vector<double> result;
  for (yaml_node_item_t* item = node->data.sequence.items.start;
       item < node->data.sequence.items.top; ++item) {
    yaml_node_t* value_node = yaml_document_get_node(document, *item);
    const std::string value = scalar(value_node, key);
    std::size_t parsed = 0;
    result.push_back(std::stod(value, &parsed));
    if (parsed != value.size())
      throw std::runtime_error("Invalid sequence number for YAML key: " + key);
  }
  return result;
}

std::optional<std::vector<std::string>> optional_strings(yaml_document_t* document,
                                                         yaml_node_t* parent,
                                                         const std::string& key) {
  yaml_node_t* node = mapping_value(document, parent, key, false);
  if (!node) return std::nullopt;
  if (node->type != YAML_SEQUENCE_NODE)
    throw std::runtime_error("YAML key '" + key + "' must be a sequence");
  std::vector<std::string> result;
  for (yaml_node_item_t* item = node->data.sequence.items.start;
       item < node->data.sequence.items.top; ++item) {
    result.push_back(scalar(yaml_document_get_node(document, *item), key));
  }
  return result;
}

std::filesystem::path resolve_path(const std::filesystem::path& directory,
                                   const std::string& value) {
  if (value.empty()) return {};
  const std::filesystem::path path(value);
  return std::filesystem::absolute(path.is_absolute() ? path : directory / path)
      .lexically_normal();
}

void require_nonnegative(const std::optional<double>& value, const char* name) {
  if (value && *value < 0.0)
    throw std::runtime_error(std::string(name) + " must be nonnegative");
}

}  // namespace

SimulatorConfig SimulatorConfig::Load(const std::filesystem::path& path) {
  const std::filesystem::path absolute_path = std::filesystem::absolute(path).lexically_normal();
  const std::filesystem::path directory = absolute_path.parent_path();
  YamlDocument yaml(absolute_path);
  yaml_document_t* document = yaml.get();
  yaml_node_t* root = yaml_document_get_root_node(document);
  if (!root || root->type != YAML_MAPPING_NODE)
    throw std::runtime_error("YAML config root must be a mapping");
  if (integer(document, root, "schema_version", 0, true) != 1)
    throw std::runtime_error("Unsupported simulator config schema");

  yaml_node_t* paths = mapping(document, root, "paths");
  yaml_node_t* runtime = mapping(document, root, "runtime");
  yaml_node_t* teleoperation = mapping(document, root, "teleoperation");
  yaml_node_t* startup = mapping_value(document, root, "startup", false);
  if (startup && startup->type != YAML_MAPPING_NODE)
    throw std::runtime_error("YAML key 'startup' must be a mapping");
  SimulatorConfig result;
  result.source_path = absolute_path;
  result.task_name = scalar(mapping_value(document, root, "task_name"), "task_name");
  result.repository_root = resolve_path(
      directory, optional_string(document, paths, "repository_root", "../.."));
  result.deployment_bundle = resolve_path(
      directory, scalar(mapping_value(document, paths, "deployment_bundle"),
                        "deployment_bundle"));
  result.scene_path = resolve_path(
      directory, optional_string(document, paths, "scene_path", ""));
  result.policy_backend = optional_string(document, paths, "policy_backend", "torchscript");
  result.policy_path = resolve_path(
      directory, optional_string(document, paths, "policy_path", ""));
  result.viewer = boolean(document, runtime, "viewer", true);
  result.realtime = boolean(document, runtime, "realtime", true);
  result.steps = integer(document, runtime, "steps", 0);
  result.status_interval_seconds = number(document, runtime, "status_interval_seconds", 1.0);
  result.teleop_deadzone = optional_number(document, teleoperation, "deadzone");
  result.teleop_force_limit = optional_number(document, teleoperation, "force_limit");
  result.teleop_position_rates = optional_numbers(document, teleoperation, "position_rates");
  result.teleop_wrist_rate = optional_number(document, teleoperation, "wrist_rate");
  result.teleop_gripper_rate = optional_number(document, teleoperation, "gripper_rate");
  result.force_field_enabled = boolean(
      document, teleoperation, "force_field_enabled", true);
  result.force_field_stiffness = number(
      document, teleoperation, "force_field_stiffness", 200.0);
  result.force_field_damping = number(
      document, teleoperation, "force_field_damping", 6.0);
  result.force_field_limit = number(
      document, teleoperation, "force_field_limit", 70.0);
  if (startup) {
    result.startup_fold_duration = optional_number(document, startup, "fold_duration_seconds");
    result.startup_stand_duration = optional_number(document, startup, "stand_duration_seconds");
    result.startup_dog_dof_names = optional_strings(document, startup, "dog_joint_names");
    result.startup_fold_positions = optional_numbers(document, startup, "fold_positions");
    result.startup_stand_positions = optional_numbers(document, startup, "stand_positions");
    result.startup_p_gains = optional_numbers(document, startup, "kp");
    result.startup_d_gains = optional_numbers(document, startup, "kd");
  }

  if (result.task_name.empty()) throw std::runtime_error("task_name must not be empty");
  if (result.deployment_bundle.empty())
    throw std::runtime_error("deployment_bundle must not be empty");
  if (result.policy_backend != "torchscript") {
    throw std::runtime_error(
        "Unsupported policy_backend '" + result.policy_backend +
        "'; this build uses LibTorch and expects a TorchScript .pt policy");
  }
  if (result.policy_path.extension() == ".onnx") {
    throw std::runtime_error(
        "policy_path points to ONNX, but this build uses the TorchScript backend");
  }
  if (result.steps < 0 || result.status_interval_seconds < 0.0)
    throw std::runtime_error("steps and status_interval_seconds must be nonnegative");
  if (result.teleop_deadzone &&
      (*result.teleop_deadzone < 0.0 || *result.teleop_deadzone >= 1.0))
    throw std::runtime_error("teleoperation.deadzone must be in [0, 1)");
  require_nonnegative(result.teleop_force_limit, "teleoperation.force_limit");
  require_nonnegative(result.teleop_wrist_rate, "teleoperation.wrist_rate");
  require_nonnegative(result.teleop_gripper_rate, "teleoperation.gripper_rate");
  if (result.force_field_stiffness <= 0.0 ||
      result.force_field_damping < 0.0 || result.force_field_limit <= 0.0) {
    throw std::runtime_error(
        "force-field stiffness/limit must be positive and damping nonnegative");
  }
  if ((result.startup_fold_duration && *result.startup_fold_duration <= 0.0) ||
      (result.startup_stand_duration && *result.startup_stand_duration <= 0.0)) {
    throw std::runtime_error("startup interpolation durations must be positive");
  }
  if (result.teleop_position_rates) {
    if (result.teleop_position_rates->size() != 3 ||
        std::any_of(result.teleop_position_rates->begin(),
                    result.teleop_position_rates->end(),
                    [](double value) { return value < 0.0; })) {
      throw std::runtime_error(
          "teleoperation.position_rates must contain 3 nonnegative values");
    }
  }
  return result;
}

void SimulatorConfig::apply_profile_overrides(TaskProfile& profile) const {
  if (profile.task_name != task_name) {
    throw std::runtime_error("Config task_name '" + task_name +
                             "' does not match deployment bundle task '" +
                             profile.task_name + "'");
  }
  if (teleop_deadzone) profile.teleop_deadzone = *teleop_deadzone;
  if (teleop_force_limit) profile.teleop_force_limit = *teleop_force_limit;
  if (teleop_position_rates) profile.teleop_position_rates = *teleop_position_rates;
  if (teleop_wrist_rate) profile.teleop_wrist_rate = *teleop_wrist_rate;
  if (teleop_gripper_rate) profile.teleop_gripper_rate = *teleop_gripper_rate;
  if (startup_fold_duration) profile.startup_fold_duration = *startup_fold_duration;
  if (startup_stand_duration) profile.startup_stand_duration = *startup_stand_duration;
  if (startup_dog_dof_names) profile.startup_dog_dof_names = *startup_dog_dof_names;
  if (startup_fold_positions) profile.startup_fold_positions = *startup_fold_positions;
  if (startup_stand_positions) profile.startup_stand_positions = *startup_stand_positions;
  if (startup_p_gains) profile.startup_p_gains = *startup_p_gains;
  if (startup_d_gains) profile.startup_d_gains = *startup_d_gains;
  profile.validate();
}

}  // namespace mujoco
