#include "mujoco/simulator.hpp"

#include <mujoco/mujoco.h>
#ifdef MUJOCO_HAS_CLASSIC_VIEWER
#include "glfw_adapter.h"
#include "simulate.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
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
  int end_effector_body = -1;
#ifdef MUJOCO_HAS_CLASSIC_VIEWER
  mjvCamera camera{};
  mjvOption visual_options{};
  mjvPerturb perturb{};
  std::unique_ptr<Simulate> viewer;
  bool viewer_attached = false;
#endif

  explicit Impl(const TaskProfile& task, bool enable_viewer) : profile(task) {
    std::array<char, 2048> error{};
    model = mj_loadXML(profile.model_path.string().c_str(), nullptr, error.data(), error.size());
    if (!model) throw std::runtime_error("MuJoCo model load failed: " + std::string(error.data()));
    data = mj_makeData(model);
    if (!data) throw std::runtime_error("MuJoCo data allocation failed");
    model->opt.timestep = profile.physics_dt;
    const int base_body = mj_name2id(model, mjOBJ_BODY, profile.base_body.c_str());
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

void MujocoSimulator::step(const std::vector<double>& torque) {
  if (torque.size() != impl_->actuator_ids.size())
    throw std::runtime_error("Torque vector has the wrong dimension");
  std::fill(impl_->data->ctrl, impl_->data->ctrl + impl_->model->nu, 0.0);
  for (std::size_t index = 0; index < torque.size(); ++index)
    impl_->data->ctrl[impl_->actuator_ids[index]] = torque[index];
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
