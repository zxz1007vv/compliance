import hashlib
import json
import tempfile
import unittest
import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from wbc_compliance_rl.algorithms.ppo_cse import PPO, PPO_Args
from wbc_compliance_rl.modules.actor_critic import AC_Args, ActorCritic
from wbc_compliance_rl.storage.rollout_storage import RolloutStorage
from wbc_compliance_rl.utils.policy_export import export_policy_as_jit


EXPECTED_MODEL_SPEC_SHA256 = (
    "b5c1d0bb801d0cde86c11b843ae5b39b31431f2f9069539157bb203cde90716a"
)
EXPECTED_PARAMETER_COUNT = 1_497_271


class RLStructureContracts(unittest.TestCase):
    def setUp(self):
        AC_Args.init_noise_std = 1.0
        AC_Args.actor_hidden_dims = [512, 256, 128]
        AC_Args.critic_hidden_dims = [512, 256, 128]
        AC_Args.adaptation_module_branch_hidden_dims = [256, 128]
        AC_Args.activation = "elu"
        AC_Args.adaptation_labels = [
            "motion_loss",
            "dynamics_loss",
            "force_loss",
            "friction_loss",
            "gripper_pos_loss",
            "gripper_target_pos_loss",
        ]
        AC_Args.adaptation_dims = [3, 3, 3, 1, 3, 3]
        AC_Args.adaptation_weights = [1, 1, 0.05, 1, 10, 1]
        AC_Args.use_decoder = False

    @staticmethod
    def make_model():
        return ActorCritic(87, 16, 870, 19)

    def test_model_state_dict_contract_is_unchanged(self):
        torch.manual_seed(1234)
        model = self.make_model()
        spec = [(name, list(tensor.shape)) for name, tensor in model.state_dict().items()]
        digest = hashlib.sha256(
            json.dumps(spec, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        self.assertEqual(digest, EXPECTED_MODEL_SPEC_SHA256)
        self.assertEqual(
            sum(tensor.numel() for tensor in model.state_dict().values()),
            EXPECTED_PARAMETER_COUNT,
        )

    def test_full_and_legacy_state_dicts_load_strictly(self):
        torch.manual_seed(1234)
        source = self.make_model()
        target = self.make_model()

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            legacy_path = directory / "legacy.pt"
            complete_path = directory / "complete.pt"
            torch.save(source.state_dict(), legacy_path)
            torch.save({"model_state_dict": source.state_dict(), "iter": 17}, complete_path)

            legacy = torch.load(legacy_path, map_location="cpu")
            target.load_state_dict(legacy, strict=True)
            complete = torch.load(complete_path, map_location="cpu")
            target.load_state_dict(complete["model_state_dict"], strict=True)

        for name, tensor in source.state_dict().items():
            torch.testing.assert_close(tensor, target.state_dict()[name])

    def test_explicit_policy_config_is_numerically_identical(self):
        explicit_cfg = SimpleNamespace(
            **{
                key: copy.deepcopy(value)
                for key, value in vars(AC_Args).items()
                if not key.startswith("_") and not callable(value)
            }
        )
        torch.manual_seed(23)
        legacy_model = self.make_model()
        torch.manual_seed(23)
        explicit_model = ActorCritic(87, 16, 870, 19, cfg=explicit_cfg)

        for name, tensor in legacy_model.state_dict().items():
            torch.testing.assert_close(
                tensor, explicit_model.state_dict()[name], rtol=0, atol=0
            )

        observations = torch.randn(5, 870)
        torch.testing.assert_close(
            legacy_model.get_student_latent(observations),
            explicit_model.get_student_latent(observations),
            rtol=0,
            atol=0,
        )

    def test_exported_policy_matches_python_inference(self):
        torch.manual_seed(91)
        model = self.make_model().eval()
        observations = torch.randn(3, 870)
        with torch.no_grad():
            latent = model.adaptation_module(observations)
            expected = model.actor_body(torch.cat((observations, latent), dim=-1))

        with tempfile.TemporaryDirectory() as directory:
            output_path = export_policy_as_jit(model, directory)
            exported = torch.jit.load(str(output_path)).eval()
            with torch.no_grad():
                actual = exported(observations)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_policy_export_preserves_training_mode_and_rng(self):
        torch.manual_seed(314)
        model = self.make_model().train()
        rng_before = torch.get_rng_state().clone()

        with tempfile.TemporaryDirectory() as directory:
            export_policy_as_jit(model, directory)

        self.assertTrue(model.training)
        self.assertTrue(model.actor_body.training)
        self.assertTrue(model.adaptation_module.training)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))

    def test_rollout_and_ppo_keep_training_shapes(self):
        torch.manual_seed(7)
        model = self.make_model()
        algorithm = PPO(model, device="cpu")
        algorithm.init_storage(4, 2, [87], [16], [870], [19])

        self.assertEqual(tuple(algorithm.storage.observations.shape), (2, 4, 87))
        self.assertEqual(
            tuple(algorithm.storage.privileged_observations.shape), (2, 4, 16)
        )
        self.assertEqual(
            tuple(algorithm.storage.observation_histories.shape), (2, 4, 870)
        )
        self.assertEqual(tuple(algorithm.storage.actions.shape), (2, 4, 19))

    def test_ppo_step_does_not_require_optional_curriculum_bins(self):
        torch.manual_seed(8)
        algorithm = PPO(self.make_model(), device="cpu")
        algorithm.init_storage(4, 2, [87], [16], [870], [19])
        algorithm.act(
            torch.zeros(4, 87),
            torch.zeros(4, 16),
            torch.zeros(4, 870),
        )

        algorithm.process_env_step(
            torch.zeros(4),
            torch.zeros(4, dtype=torch.bool),
            {},
        )

        self.assertEqual(1, algorithm.storage.step)


if __name__ == "__main__":
    unittest.main()
