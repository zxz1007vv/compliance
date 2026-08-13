"""B1+Z1 task entry point.

The subclass deliberately adds no simulation behavior. It gives the existing
whole-body environment a task-correct home while preserving its exact step
implementation and the historical import path.
"""

from b1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv


class B1Z1Env(VelocityTrackingEasyEnv):
    pass
