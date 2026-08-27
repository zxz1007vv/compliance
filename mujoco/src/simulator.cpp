#include "mujoco/simulator.hpp"

#include <mujoco/mujoco.h>
#ifdef MUJOCO_HAS_CLASSIC_VIEWER
#include "glfw_adapter.h"
#include "simulate.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace mujoco {

struct MujocoSimulator::Impl {
  const TaskProfile& profile;
  mjModel* model = nullptr;
  mjData* data = nullptr;
  std::vector<int> joint_ids;
  std::vector<int> qpos_addresses;
  std::vector<int> dof_addresses;
  std::vector<int> actuator_ids;
  int base_free_joint = -1;
  int base_body = -1;
  int end_effector_body = -1;
  std::array<double, 3> force_field_latched_anchor_local{{0.0, 0.0, 0.0}};
  std::array<double, 3> force_field_anchor_local{{0.0, 0.0, 0.0}};
  std::array<double, 3> force_field_anchor_velocity_local{{0.0, 0.0, 0.0}};
  std::array<double, 3> force_field_anchor_world{{0.0, 0.0, 0.0}};
  std::array<double, 3> spring_force_world{{0.0, 0.0, 0.0}};
  std::array<double, 3> last_spring_force_world{{0.0, 0.0, 0.0}};
  std::array<double, 3> spring_force_unclipped_world{{0.0, 0.0, 0.0}};
  std::array<double, 3> last_spring_force_unclipped_world{{0.0, 0.0, 0.0}};
  std::array<double, 3> last_applied_force_world{{0.0, 0.0, 0.0}};
  MousePerturbationDebugState last_mouse_perturbation;
  double force_field_stiffness = 0.0;
  double force_field_damping = 0.0;
  double force_field_limit = 0.0;
  std::string force_anchor_mode = "world_fixed";
  ForceAnchorMotionConfig force_anchor_motion;
  double force_anchor_motion_end_time = 0.0;
  std::mt19937 force_anchor_random{0};
  bool force_field_active = false;
  bool spring_force_present = false;
#ifdef MUJOCO_HAS_CLASSIC_VIEWER
  mjvCamera camera{};
  mjvOption visual_options{};
  mjvPerturb perturb{};
  std::unique_ptr<Simulate> viewer;
  bool viewer_attached = false;
#endif

  double command_base_yaw() const {
    const int base_qpos = model->jnt_qposadr[base_free_joint];
    const double qw = data->qpos[base_qpos + 3];
    const double qx = data->qpos[base_qpos + 4];
    const double qy = data->qpos[base_qpos + 5];
    const double qz = data->qpos[base_qpos + 6];
    const double norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (norm <= 0.0)
      throw std::runtime_error("Cannot compute force anchor from zero quaternion");
    const double w = qw / norm;
    const double x = qx / norm;
    const double y = qy / norm;
    const double z = qz / norm;
    return std::atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z));
  }

  std::array<double, 3> command_base_origin_world() const {
    const int base_qpos = model->jnt_qposadr[base_free_joint];
    return {{data->qpos[base_qpos], data->qpos[base_qpos + 1],
             profile.command_base_height}};
  }

  void update_force_anchor_world() {
    if (force_anchor_mode != "robot_relative_static" &&
        force_anchor_mode != "robot_relative_moving")
      return;
    const auto origin = command_base_origin_world();
    const double yaw = command_base_yaw();
    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);
    force_field_anchor_world = {{
        origin[0] + cos_yaw * force_field_anchor_local[0] -
                        sin_yaw * force_field_anchor_local[1],
        origin[1] + sin_yaw * force_field_anchor_local[0] +
                        cos_yaw * force_field_anchor_local[1],
        origin[2] + force_field_anchor_local[2],
    }};
  }

  void sample_force_anchor_motion() {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::array<double, 3> direction{{normal(force_anchor_random),
                                     normal(force_anchor_random),
                                     normal(force_anchor_random)}};
    const double norm = std::sqrt(direction[0] * direction[0] +
                                  direction[1] * direction[1] +
                                  direction[2] * direction[2]);
    if (norm <= 0.0) direction = {{1.0, 0.0, 0.0}};
    const double safe_norm = norm <= 0.0 ? 1.0 : norm;
    std::uniform_real_distribution<double> speed(
        force_anchor_motion.velocity_range[0],
        force_anchor_motion.velocity_range[1]);
    const double sampled_speed = speed(force_anchor_random);
    for (int axis = 0; axis < 3; ++axis)
      force_field_anchor_velocity_local[axis] =
          direction[axis] / safe_norm * sampled_speed;
    std::uniform_real_distribution<double> duration(
        force_anchor_motion.duration_range[0],
        force_anchor_motion.duration_range[1]);
    force_anchor_motion_end_time = data->time + duration(force_anchor_random);
  }

  void advance_force_anchor_motion() {
    if (force_anchor_mode != "robot_relative_moving") return;
    if (data->time >= force_anchor_motion_end_time)
      sample_force_anchor_motion();
    for (int axis = 0; axis < 3; ++axis) {
      double offset = force_field_anchor_local[axis] -
                      force_field_latched_anchor_local[axis] +
                      force_field_anchor_velocity_local[axis] *
                          model->opt.timestep;
      if (std::abs(offset) > force_anchor_motion.offset_limit[axis]) {
        force_field_anchor_velocity_local[axis] *= -1.0;
        offset = std::clamp(offset, -force_anchor_motion.offset_limit[axis],
                            force_anchor_motion.offset_limit[axis]);
      }
      force_field_anchor_local[axis] =
          force_field_latched_anchor_local[axis] + offset;
    }
  }

  explicit Impl(const TaskProfile& task, bool enable_viewer) : profile(task) {
    std::array<char, 2048> error{};
    model = mj_loadXML(profile.model_path.string().c_str(), nullptr, error.data(), error.size());
    if (!model) throw std::runtime_error("MuJoCo model load failed: " + std::string(error.data()));
    data = mj_makeData(model);
    if (!data) throw std::runtime_error("MuJoCo data allocation failed");
    model->opt.timestep = profile.physics_dt;
    base_body = mj_name2id(model, mjOBJ_BODY, profile.base_body.c_str());
    if (base_body < 0) throw std::runtime_error("Base body is missing from model: " + profile.base_body);
    // Some URDFs have a massless root followed by fixed base/trunk links.  They
    // become one floating articulation in Isaac Gym, while the generated MJCF
    // keeps the fixed hierarchy.  There must still be exactly one free joint.
    for (int joint = 0; joint < model->njnt; ++joint) {
      if (model->jnt_type[joint] != mjJNT_FREE) continue;
      const int joint_body = model->jnt_bodyid[joint];
      for (int ancestor = base_body; ancestor > 0; ancestor = model->body_parentid[ancestor]) {
        if (ancestor == joint_body) base_free_joint = joint;
      }
    }
    if (base_free_joint < 0) throw std::runtime_error("Model has no floating-base free joint");
    end_effector_body = mj_name2id(model, mjOBJ_BODY, profile.end_effector_body.c_str());
    if (end_effector_body < 0)
      throw std::runtime_error("End-effector body is missing from model: " + profile.end_effector_body);

    for (const auto& name : profile.dof_names) {
      const int joint = mj_name2id(model, mjOBJ_JOINT, name.c_str());
      if (joint < 0) throw std::runtime_error("Model joint missing: " + name);
      if (model->jnt_type[joint] != mjJNT_HINGE)
        throw std::runtime_error("Policy joint is not a one-DOF hinge: " + name);
      int actuator = -1;
      for (int candidate = 0; candidate < model->nu; ++candidate) {
        if (model->actuator_trntype[candidate] == mjTRN_JOINT &&
            model->actuator_trnid[2 * candidate] == joint) {
          actuator = candidate;
          break;
        }
      }
      if (actuator < 0) throw std::runtime_error("No direct motor actuator for joint: " + name);
      joint_ids.push_back(joint);
      qpos_addresses.push_back(model->jnt_qposadr[joint]);
      dof_addresses.push_back(model->jnt_dofadr[joint]);
      actuator_ids.push_back(actuator);
    }
    reset();
#ifdef MUJOCO_HAS_CLASSIC_VIEWER
    if (enable_viewer) {
      mjv_defaultCamera(&camera);
      mjv_defaultFreeCamera(model, &camera);
      mjv_defaultOption(&visual_options);
      mjv_defaultPerturb(&perturb);
      viewer = std::make_unique<Simulate>(
          std::make_unique<GlfwAdapter>(), &camera, &visual_options, &perturb,
          /*is_passive=*/true);
    }
#else
    if (enable_viewer)
      throw std::runtime_error(
          "Viewer requested but the official MuJoCo simulate UI was not compiled");
#endif
  }

  ~Impl() {
#ifdef MUJOCO_HAS_CLASSIC_VIEWER
    if (viewer) {
      viewer->exitrequest.store(1);
      viewer.reset();
    }
#endif
    if (data) mj_deleteData(data);
    if (model) mj_deleteModel(model);
  }

  void reset() {
    mj_resetData(model, data);
    const int base_qpos = model->jnt_qposadr[base_free_joint];
    const int base_dof = model->jnt_dofadr[base_free_joint];
    for (int axis = 0; axis < 3; ++axis) {
      data->qpos[base_qpos + axis] = profile.initial_base_position[axis];
      data->qvel[base_dof + axis] = profile.initial_base_linear_velocity[axis];
      data->qvel[base_dof + 3 + axis] = profile.initial_base_angular_velocity[axis];
    }
    for (int component = 0; component < 4; ++component)
      data->qpos[base_qpos + 3 + component] = profile.initial_base_quaternion_wxyz[component];
    for (std::size_t index = 0; index < qpos_addresses.size(); ++index)
      data->qpos[qpos_addresses[index]] = profile.default_dof_positions[index];
    force_field_latched_anchor_local = {{0.0, 0.0, 0.0}};
    force_field_anchor_local = {{0.0, 0.0, 0.0}};
    force_field_anchor_velocity_local = {{0.0, 0.0, 0.0}};
    force_field_anchor_world = {{0.0, 0.0, 0.0}};
    spring_force_world = {{0.0, 0.0, 0.0}};
    last_spring_force_world = {{0.0, 0.0, 0.0}};
    spring_force_unclipped_world = {{0.0, 0.0, 0.0}};
    last_spring_force_unclipped_world = {{0.0, 0.0, 0.0}};
    last_applied_force_world = {{0.0, 0.0, 0.0}};
    last_mouse_perturbation = {};
    force_field_stiffness = 0.0;
    force_field_damping = 0.0;
    force_field_limit = 0.0;
    force_anchor_mode = "world_fixed";
    force_anchor_motion = {};
    force_anchor_motion_end_time = 0.0;
    force_field_active = false;
    spring_force_present = false;
    mj_forward(model, data);
  }
};

MujocoSimulator::MujocoSimulator(const TaskProfile& profile, bool enable_viewer)
    : impl_(std::make_unique<Impl>(profile, enable_viewer)) {}
MujocoSimulator::~MujocoSimulator() = default;
void MujocoSimulator::reset() { impl_->reset(); }

RobotState MujocoSimulator::state() const {
  RobotState result;
  const int qadr = impl_->model->jnt_qposadr[impl_->base_free_joint];
  result.base_quaternion = {{impl_->data->qpos[qadr + 3], impl_->data->qpos[qadr + 4],
                             impl_->data->qpos[qadr + 5], impl_->data->qpos[qadr + 6]}};
  result.joint_position.resize(impl_->qpos_addresses.size());
  result.joint_velocity.resize(impl_->dof_addresses.size());
  for (std::size_t index = 0; index < impl_->qpos_addresses.size(); ++index) {
    result.joint_position[index] = impl_->data->qpos[impl_->qpos_addresses[index]];
    result.joint_velocity[index] = impl_->data->qvel[impl_->dof_addresses[index]];
  }
  return result;
}

void MujocoSimulator::start_end_effector_force_field(
    double stiffness, double damping, double force_limit,
    const std::string& anchor_mode, ForceAnchorMotionConfig motion) {
  if (stiffness <= 0.0 || damping < 0.0 || force_limit <= 0.0)
    throw std::runtime_error("Invalid end-effector force-field parameters");
  if (anchor_mode != "world_fixed" &&
      anchor_mode != "robot_relative_static" &&
      anchor_mode != "robot_relative_moving")
    throw std::runtime_error("Invalid end-effector force anchor mode: " +
                             anchor_mode);
  if (motion.velocity_range[0] < 0.0 ||
      motion.velocity_range[1] < motion.velocity_range[0] ||
      motion.duration_range[0] <= 0.0 ||
      motion.duration_range[1] < motion.duration_range[0] ||
      std::any_of(motion.offset_limit.begin(), motion.offset_limit.end(),
                  [](double value) { return value <= 0.0; }))
    throw std::runtime_error("Invalid moving force-anchor parameters");
  stop_end_effector_force_field();
  impl_->force_anchor_mode = anchor_mode;
  impl_->force_anchor_motion = motion;
  for (int axis = 0; axis < 3; ++axis) {
    impl_->force_field_anchor_world[axis] =
        impl_->data->xpos[3 * impl_->end_effector_body + axis];
  }
  if (anchor_mode == "robot_relative_static" ||
      anchor_mode == "robot_relative_moving") {
    const auto origin = impl_->command_base_origin_world();
    const double yaw = impl_->command_base_yaw();
    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);
    const double dx = impl_->force_field_anchor_world[0] - origin[0];
    const double dy = impl_->force_field_anchor_world[1] - origin[1];
    impl_->force_field_anchor_local = {{
        cos_yaw * dx + sin_yaw * dy,
        -sin_yaw * dx + cos_yaw * dy,
        impl_->force_field_anchor_world[2] - origin[2],
    }};
    impl_->force_field_latched_anchor_local =
        impl_->force_field_anchor_local;
    if (anchor_mode == "robot_relative_moving")
      impl_->sample_force_anchor_motion();
  } else {
    impl_->force_field_latched_anchor_local = {{0.0, 0.0, 0.0}};
    impl_->force_field_anchor_local = {{0.0, 0.0, 0.0}};
    impl_->force_field_anchor_velocity_local = {{0.0, 0.0, 0.0}};
  }
  impl_->force_field_stiffness = stiffness;
  impl_->force_field_damping = damping;
  impl_->force_field_limit = force_limit;
  impl_->force_field_active = true;
}

void MujocoSimulator::stop_end_effector_force_field() {
  if (impl_->spring_force_present) {
    mjtNum* applied =
        impl_->data->xfrc_applied + 6 * impl_->end_effector_body;
    for (int axis = 0; axis < 3; ++axis)
      applied[axis] -= impl_->spring_force_world[axis];
  }
  impl_->spring_force_world = {{0.0, 0.0, 0.0}};
  impl_->last_spring_force_world = {{0.0, 0.0, 0.0}};
  impl_->spring_force_unclipped_world = {{0.0, 0.0, 0.0}};
  impl_->last_spring_force_unclipped_world = {{0.0, 0.0, 0.0}};
  impl_->force_field_latched_anchor_local = {{0.0, 0.0, 0.0}};
  impl_->force_field_anchor_local = {{0.0, 0.0, 0.0}};
  impl_->force_field_anchor_velocity_local = {{0.0, 0.0, 0.0}};
  impl_->force_field_anchor_world = {{0.0, 0.0, 0.0}};
  impl_->force_anchor_motion_end_time = 0.0;
  impl_->force_field_active = false;
  impl_->spring_force_present = false;
}

bool MujocoSimulator::end_effector_force_field_active() const {
  return impl_->force_field_active;
}

std::array<double, 3>
MujocoSimulator::end_effector_force_field_anchor_local() const {
  return impl_->force_field_anchor_local;
}

std::array<double, 3>
MujocoSimulator::end_effector_force_field_anchor_velocity_local() const {
  return impl_->force_field_anchor_velocity_local;
}

std::array<double, 3>
MujocoSimulator::end_effector_force_field_anchor_world() const {
  return impl_->force_field_anchor_world;
}

std::array<double, 3>
MujocoSimulator::end_effector_force_field_displacement_world() const {
  if (!impl_->force_field_active) return {{0.0, 0.0, 0.0}};
  std::array<double, 3> result{};
  for (int axis = 0; axis < 3; ++axis) {
    result[axis] = impl_->force_field_anchor_world[axis] -
                   impl_->data->xpos[3 * impl_->end_effector_body + axis];
  }
  return result;
}

std::array<double, 3> MujocoSimulator::end_effector_spring_force_world() const {
  return impl_->last_spring_force_world;
}

SpringForceDebugState
MujocoSimulator::end_effector_spring_force_debug_state() const {
  return {end_effector_force_field_displacement_world(),
          impl_->last_spring_force_unclipped_world,
          impl_->last_spring_force_world};
}

std::vector<std::array<double, 3>>
MujocoSimulator::wheel_positions_base() const {
  std::vector<std::array<double, 3>> result;
  result.reserve(impl_->profile.wheel_dof_names.size());
  const double* base_position =
      impl_->data->xpos + 3 * impl_->base_body;
  const double* base_rotation =
      impl_->data->xmat + 9 * impl_->base_body;
  for (const auto& wheel_name : impl_->profile.wheel_dof_names) {
    const std::size_t dof_index = impl_->profile.dof_index(wheel_name);
    const int body = impl_->model->jnt_bodyid[impl_->joint_ids[dof_index]];
    const double* wheel_position = impl_->data->xpos + 3 * body;
    const std::array<double, 3> relative_world{{
        wheel_position[0] - base_position[0],
        wheel_position[1] - base_position[1],
        wheel_position[2] - base_position[2],
    }};
    // MuJoCo stores a row-major local-to-world rotation. R^T maps world to
    // the base frame and matches Isaac Gym's quat_rotate_inverse.
    result.push_back({{
        base_rotation[0] * relative_world[0] +
            base_rotation[3] * relative_world[1] +
            base_rotation[6] * relative_world[2],
        base_rotation[1] * relative_world[0] +
            base_rotation[4] * relative_world[1] +
            base_rotation[7] * relative_world[2],
        base_rotation[2] * relative_world[0] +
            base_rotation[5] * relative_world[1] +
            base_rotation[8] * relative_world[2],
    }});
  }
  return result;
}

void MujocoSimulator::step(const std::vector<double>& torque) {
  if (torque.size() != impl_->actuator_ids.size())
    throw std::runtime_error("Torque vector has the wrong dimension");
  std::fill(impl_->data->ctrl, impl_->data->ctrl + impl_->model->nu, 0.0);
  for (std::size_t index = 0; index < torque.size(); ++index)
    impl_->data->ctrl[impl_->actuator_ids[index]] = torque[index];
  mjtNum* applied =
      impl_->data->xfrc_applied + 6 * impl_->end_effector_body;
  // xfrc_applied persists in headless mode. Remove our previous sample before
  // calculating the field again; Viewer Sync clears it independently.
  if (impl_->spring_force_present) {
    for (int axis = 0; axis < 3; ++axis)
      applied[axis] -= impl_->spring_force_world[axis];
  }
  impl_->spring_force_world = {{0.0, 0.0, 0.0}};
  impl_->spring_force_unclipped_world = {{0.0, 0.0, 0.0}};
  impl_->spring_force_present = false;
  if (impl_->force_field_active) {
    impl_->advance_force_anchor_motion();
    impl_->update_force_anchor_world();
    std::array<mjtNum, 6> velocity_world{};
    mj_objectVelocity(impl_->model, impl_->data, mjOBJ_BODY,
                      impl_->end_effector_body, velocity_world.data(), 0);
    for (int axis = 0; axis < 3; ++axis) {
      const double displacement =
          impl_->force_field_anchor_world[axis] -
          impl_->data->xpos[3 * impl_->end_effector_body + axis];
      impl_->spring_force_unclipped_world[axis] =
          impl_->force_field_stiffness * displacement -
          impl_->force_field_damping * velocity_world[3 + axis];
      impl_->spring_force_world[axis] = std::clamp(
          impl_->spring_force_unclipped_world[axis],
          -impl_->force_field_limit, impl_->force_field_limit);
      applied[axis] += impl_->spring_force_world[axis];
    }
    impl_->spring_force_present = true;
  }
  impl_->last_spring_force_world = impl_->spring_force_world;
  impl_->last_spring_force_unclipped_world =
      impl_->spring_force_unclipped_world;
  for (int axis = 0; axis < 3; ++axis) {
    impl_->last_applied_force_world[axis] =
        impl_->data->xfrc_applied[6 * impl_->end_effector_body + axis];
  }
#ifdef MUJOCO_HAS_CLASSIC_VIEWER
  impl_->last_mouse_perturbation = {};
  if (impl_->viewer && impl_->perturb.active && impl_->perturb.select > 0) {
    const int body = impl_->perturb.select;
    impl_->last_mouse_perturbation.active = true;
    const char* body_name = mj_id2name(impl_->model, mjOBJ_BODY, body);
    impl_->last_mouse_perturbation.body_name =
        body_name ? body_name : "<unnamed>";
    for (int axis = 0; axis < 3; ++axis) {
      impl_->last_mouse_perturbation.force_world[axis] =
          impl_->data->xfrc_applied[6 * body + axis];
      if (body == impl_->end_effector_body && impl_->spring_force_present) {
        impl_->last_mouse_perturbation.force_world[axis] -=
            impl_->spring_force_world[axis];
      }
    }
  }
#endif
  mj_step(impl_->model, impl_->data);
  // Isaac Gym applies each URDF maxJointVelocity in the PhysX articulation.
  // MuJoCo hinge joints do not have an equivalent velocity attribute, so
  // reproduce that trained transition contract explicitly after each step.
  bool state_projected = false;
  for (std::size_t index = 0; index < impl_->dof_addresses.size(); ++index) {
    const double limit = impl_->profile.joint_velocity[index];
    double& velocity = impl_->data->qvel[impl_->dof_addresses[index]];
    if (std::isfinite(limit) && limit > 0.0) {
      const double bounded = std::clamp(velocity, -limit, limit);
      state_projected = state_projected || bounded != velocity;
      velocity = bounded;
    }
    // PhysX enforces the URDF hard stops in the articulation solver.  MuJoCo
    // constraints are intentionally soft and a low-inertia gripper can cross
    // its stop within one 5 ms B1 step, so project only actual URDF-limited
    // hinges back to that same hard contract.
    double& position = impl_->data->qpos[impl_->qpos_addresses[index]];
    if (std::isfinite(impl_->profile.joint_lower[index]) &&
        position < impl_->profile.joint_lower[index]) {
      position = impl_->profile.joint_lower[index];
      velocity = std::max(velocity, 0.0);
      state_projected = true;
    }
    if (std::isfinite(impl_->profile.joint_upper[index]) &&
        position > impl_->profile.joint_upper[index]) {
      position = impl_->profile.joint_upper[index];
      velocity = std::min(velocity, 0.0);
      state_projected = true;
    }
  }
  if (state_projected) mj_forward(impl_->model, impl_->data);
}

double MujocoSimulator::time() const { return impl_->data->time; }

double MujocoSimulator::base_yaw_rate() const {
  std::array<mjtNum, 6> velocity_local{};
  mj_objectVelocity(impl_->model, impl_->data, mjOBJ_BODY, impl_->base_body,
                    velocity_local.data(), 1);
  // mj_objectVelocity returns angular XYZ followed by linear XYZ.
  return velocity_local[2];
}

double MujocoSimulator::base_forward_velocity() const {
  std::array<mjtNum, 6> velocity_local{};
  mj_objectVelocity(impl_->model, impl_->data, mjOBJ_BODY, impl_->base_body,
                    velocity_local.data(), 1);
  return velocity_local[3];
}

double MujocoSimulator::end_effector_contact_force() const {
  double magnitude = 0.0;
  std::array<mjtNum, 6> wrench{};
  for (int index = 0; index < impl_->data->ncon; ++index) {
    const mjContact& contact = impl_->data->contact[index];
    const int body1 = impl_->model->geom_bodyid[contact.geom1];
    const int body2 = impl_->model->geom_bodyid[contact.geom2];
    if (body1 != impl_->end_effector_body && body2 != impl_->end_effector_body) continue;
    mj_contactForce(impl_->model, impl_->data, index, wrench.data());
    magnitude += std::abs(wrench[0]);
  }
  return magnitude;
}

std::array<double, 3> MujocoSimulator::end_effector_contact_force_world() const {
  std::array<double, 3> force_world{{0.0, 0.0, 0.0}};
  std::array<mjtNum, 6> wrench_contact{};
  for (int index = 0; index < impl_->data->ncon; ++index) {
    const mjContact& contact = impl_->data->contact[index];
    const int body1 = impl_->model->geom_bodyid[contact.geom1];
    const int body2 = impl_->model->geom_bodyid[contact.geom2];
    if (body1 != impl_->end_effector_body && body2 != impl_->end_effector_body)
      continue;

    mj_contactForce(
        impl_->model, impl_->data, index, wrench_contact.data());
    // contact.frame stores the normal and two tangent axes as rows. MuJoCo's
    // wrench acts on geom2; reverse it when the end effector is geom1.
    const double end_effector_sign =
        body2 == impl_->end_effector_body ? 1.0 : -1.0;
    for (int world_axis = 0; world_axis < 3; ++world_axis) {
      double component = 0.0;
      for (int contact_axis = 0; contact_axis < 3; ++contact_axis) {
        component += contact.frame[3 * contact_axis + world_axis] *
                     wrench_contact[contact_axis];
      }
      force_world[world_axis] += end_effector_sign * component;
    }
  }
  return force_world;
}

std::array<double, 3> MujocoSimulator::end_effector_applied_force_world() const {
  return impl_->last_applied_force_world;
}

MousePerturbationDebugState MujocoSimulator::mouse_perturbation_debug_state() const {
  return impl_->last_mouse_perturbation;
}

EndEffectorDebugState MujocoSimulator::end_effector_debug_state() const {
  EndEffectorDebugState result;
  for (int axis = 0; axis < 3; ++axis)
    result.world_position[axis] =
        impl_->data->xpos[3 * impl_->end_effector_body + axis];

  // Reproduce CommandLifecycleMixin::get_measured_ee_pos_spherical(): the
  // command origin follows base X/Y and yaw, but has a fixed world Z.
  const int base_qpos = impl_->model->jnt_qposadr[impl_->base_free_joint];
  const double qw = impl_->data->qpos[base_qpos + 3];
  const double qx = impl_->data->qpos[base_qpos + 4];
  const double qy = impl_->data->qpos[base_qpos + 5];
  const double qz = impl_->data->qpos[base_qpos + 6];
  const double quaternion_norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  if (quaternion_norm <= std::numeric_limits<double>::epsilon())
    throw std::runtime_error("Cannot compute end-effector debug frame from zero quaternion");
  const double w = qw / quaternion_norm;
  const double x = qx / quaternion_norm;
  const double y = qy / quaternion_norm;
  const double z = qz / quaternion_norm;
  const double base_yaw = std::atan2(2.0 * (w * z + x * y),
                                     1.0 - 2.0 * (y * y + z * z));

  const double world_dx = result.world_position[0] - impl_->data->qpos[base_qpos];
  const double world_dy = result.world_position[1] - impl_->data->qpos[base_qpos + 1];
  const double world_dz = result.world_position[2] - impl_->profile.command_base_height;
  const double cos_base = std::cos(base_yaw);
  const double sin_base = std::sin(base_yaw);
  const std::array<double, 3> base_position{{
      cos_base * world_dx + sin_base * world_dy,
      -sin_base * world_dx + cos_base * world_dy,
      world_dz,
  }};

  const double translated_x =
      base_position[0] - impl_->profile.arm_mount_translation[0];
  const double translated_y =
      base_position[1] - impl_->profile.arm_mount_translation[1];
  const double translated_z =
      base_position[2] - impl_->profile.arm_mount_translation[2];
  const double cos_mount = std::cos(impl_->profile.arm_mount_yaw);
  const double sin_mount = std::sin(impl_->profile.arm_mount_yaw);
  result.arm_position = {{
      cos_mount * translated_x + sin_mount * translated_y,
      -sin_mount * translated_x + cos_mount * translated_y,
      translated_z,
  }};

  const double radius = std::sqrt(result.arm_position[0] * result.arm_position[0] +
                                  result.arm_position[1] * result.arm_position[1] +
                                  result.arm_position[2] * result.arm_position[2]);
  const double pitch = radius > std::numeric_limits<double>::epsilon()
                           ? -std::asin(std::clamp(result.arm_position[2] / radius,
                                                  -1.0, 1.0))
                           : 0.0;
  result.arm_spherical =
      {{radius, pitch, std::atan2(result.arm_position[1], result.arm_position[0])}};
  return result;
}

void MujocoSimulator::render() {
#ifdef MUJOCO_HAS_CLASSIC_VIEWER
  if (!impl_->viewer) return;
  if (!impl_->viewer_attached) {
    impl_->viewer->Load(impl_->model, impl_->data,
                        impl_->profile.model_path.string().c_str());
    impl_->viewer_attached = true;
  }
  impl_->viewer->Sync();
  // Passive Sync clears xfrc_applied and writes the current mouse perturbation.
  impl_->spring_force_present = false;
#endif
}

void MujocoSimulator::viewer_loop() {
#ifdef MUJOCO_HAS_CLASSIC_VIEWER
  if (impl_->viewer) impl_->viewer->RenderLoop();
#endif
}

void MujocoSimulator::request_viewer_close() {
#ifdef MUJOCO_HAS_CLASSIC_VIEWER
  if (impl_->viewer) impl_->viewer->exitrequest.store(1);
#endif
}

bool MujocoSimulator::viewer_should_close() const {
#ifdef MUJOCO_HAS_CLASSIC_VIEWER
  return impl_->viewer && impl_->viewer->exitrequest.load() != 0;
#else
  return false;
#endif
}

bool MujocoSimulator::viewer_enabled() const {
#ifdef MUJOCO_HAS_CLASSIC_VIEWER
  return impl_->viewer != nullptr;
#else
  return false;
#endif
}

bool MujocoSimulator::viewer_paused() const {
#ifdef MUJOCO_HAS_CLASSIC_VIEWER
  if (!impl_->viewer) return false;
  const MutexLock lock(impl_->viewer->mtx);
  return impl_->viewer->run == 0;
#else
  return false;
#endif
}

}  // namespace mujoco
