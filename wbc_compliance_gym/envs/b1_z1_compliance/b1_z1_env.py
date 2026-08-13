"""B1+Z1 task entry point.

The subclass deliberately adds no simulation behavior. It gives the existing
whole-body environment a task-correct home while preserving its exact step
implementation.
"""

from wbc_compliance_gym.envs.base.velocity_tracking_env import VelocityTrackingEnv


class B1Z1Env(VelocityTrackingEnv):
    pass
