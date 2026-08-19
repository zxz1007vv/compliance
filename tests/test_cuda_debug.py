import io
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace

import isaacgym  # noqa: F401 - must precede torch imports in this project.
import torch

from wbc_compliance_gym.utils.cuda_debug import CudaPhysicsDebugger


class CudaPhysicsDebuggerTest(unittest.TestCase):
    def test_debugger_is_disabled_by_default(self):
        self.assertIsNone(
            CudaPhysicsDebugger.from_environment("cpu", 4, environ={})
        )

    def test_environment_options_are_validated(self):
        with self.assertRaisesRegex(ValueError, "COMPLIANCE_CUDA_DEBUG"):
            CudaPhysicsDebugger.from_environment(
                "cpu", 4, environ={"COMPLIANCE_CUDA_DEBUG": "sometimes"}
            )
        with self.assertRaisesRegex(ValueError, "COMPLIANCE_CUDA_DEBUG_INTERVAL"):
            CudaPhysicsDebugger.from_environment(
                "cpu",
                4,
                environ={
                    "COMPLIANCE_CUDA_DEBUG": "1",
                    "COMPLIANCE_CUDA_DEBUG_INTERVAL": "0",
                },
            )

    def test_non_finite_values_report_indices_and_environment(self):
        debugger = CudaPhysicsDebugger.from_environment(
            "cpu",
            2,
            environ={"COMPLIANCE_CUDA_DEBUG": "1"},
        )
        values = torch.tensor([[0.0, float("nan")], [float("inf"), 1.0]])
        stderr = io.StringIO()

        with redirect_stderr(stderr), self.assertRaises(FloatingPointError):
            debugger.check_tensors("before_simulate", {"actions": values}, 7, 2)

        output = stderr.getvalue()
        self.assertIn("tensor=actions", output)
        self.assertIn("step=7 substep=2", output)
        self.assertIn("env_ids=[0, 1]", output)

    def test_periodic_stats_and_startup_config_are_reported(self):
        debugger = CudaPhysicsDebugger.from_environment(
            "cpu",
            2,
            environ={
                "COMPLIANCE_CUDA_DEBUG": "true",
                "COMPLIANCE_CUDA_DEBUG_INTERVAL": "5",
            },
        )
        cfg = SimpleNamespace(
            sim=SimpleNamespace(
                dt=0.005,
                substeps=1,
                physx=SimpleNamespace(
                    max_gpu_contact_pairs=2 ** 23,
                    default_buffer_size_multiplier=5,
                    contact_collection=2,
                ),
            ),
            control=SimpleNamespace(decimation=4),
            asset=SimpleNamespace(self_collisions=0),
        )
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            debugger.log_startup(cfg)
            debugger.check_tensors(
                "step_outputs", {"rewards": torch.tensor([-2.0, 3.0])}, 10
            )

        output = stderr.getvalue()
        self.assertIn("[cuda-debug] enabled", output)
        self.assertIn("num_envs=2", output)
        self.assertIn("max_gpu_contact_pairs=8388608", output)
        self.assertIn("rewards[min=-2,max=3,abs=3]", output)

    def test_named_dof_and_contact_outliers_are_retained(self):
        debugger = CudaPhysicsDebugger.from_environment(
            "cpu",
            2,
            environ={"COMPLIANCE_CUDA_DEBUG": "1"},
        )
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            debugger.report_dof_diagnostics(
                positions=torch.tensor([[0.0, 0.0], [1.5, -3.0]]),
                velocities=torch.tensor([[0.0, 0.0], [12.0, -25.0]]),
                requested_targets=torch.tensor([[0.0, 3.0], [2.0, 0.0]]),
                hard_clamped_targets=torch.tensor([[0.0, 2.0], [1.0, 0.0]]),
                applied_targets=torch.tensor([[0.0, 1.5], [0.5, 0.0]]),
                hard_limits=torch.tensor([[-1.0, 1.0], [-2.0, 2.0]]),
                velocity_limits=torch.tensor([10.0, 20.0]),
                dof_names=("hip", "calf"),
                step=0,
                substep=0,
            )
            debugger.report_contact_diagnostics(
                torch.tensor(
                    [
                        [[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]],
                        [[0.0, 4.0, 3.0], [0.0, 0.0, 1.0]],
                    ]
                ),
                ("base", "foot"),
                step=0,
                substep=0,
                ignored_body_indices=torch.tensor([1]),
            )

        self.assertIn("env=1,item=calf,value=1", debugger.last_dof_summary)
        self.assertIn("env=1,item=hip,value=1", debugger.last_dof_summary)
        self.assertIn("env=1,item=calf,value=5", debugger.last_dof_summary)
        self.assertIn("hard_target_clamp", debugger.last_dof_summary)
        self.assertIn("target_slew_limit", debugger.last_dof_summary)
        self.assertIn("env=1,item=base,value=5", debugger.last_contact_summary)
        self.assertIn("nonignored_bodies_gt1N=2", debugger.last_contact_summary)
        self.assertIn("envs_with_nonignored_contact=2", debugger.last_contact_summary)
        self.assertIn("hard_limit_violation", stderr.getvalue())

    def test_action_saturation_reports_per_dof_rates(self):
        debugger = CudaPhysicsDebugger.from_environment(
            "cpu",
            2,
            environ={"COMPLIANCE_CUDA_DEBUG": "1"},
        )

        debugger.report_action_diagnostics(
            torch.tensor([[4.0, 0.0], [-4.0, 2.0]]),
            ("hip", "arm"),
            clip=4.0,
            step=0,
        )

        self.assertIn("saturation_rate=50.000%", debugger.last_action_summary)
        self.assertIn("item=hip,value=100", debugger.last_action_summary)

    def test_safety_reset_counts_are_retained(self):
        debugger = CudaPhysicsDebugger.from_environment(
            "cpu",
            3,
            environ={"COMPLIANCE_CUDA_DEBUG": "1"},
        )
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            debugger.report_safety_resets(
                {
                    "velocity": torch.tensor([False, True, False]),
                    "nonfoot_contact": torch.tensor([True, False, True]),
                },
                step=0,
            )

        self.assertIn("velocity=1", debugger.last_safety_summary)
        self.assertIn("nonfoot_contact=2", debugger.last_safety_summary)
        self.assertIn("safety_resets", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
