# Package namespace migration

## Result

The simulator/task package is now `wbc_compliance_gym`; the reinforcement-
learning package is `wbc_compliance_rl`. The former `b1_gym` and
`b1_gym_learn` directories were removed rather than retained as forwarding
packages.

Canonical task-facing paths:

```text
wbc_compliance_gym/
├── envs/base/
├── envs/b1_z1_compliance/{b1_z1_config.py,b1_z1_env.py}
├── robots/configs/
├── commands/
├── curriculum/
├── rewards/b1_z1.py
└── sensors/sensors.py

wbc_compliance_rl/
├── algorithms/
├── modules/
├── runners/
├── storage/
├── logging/
└── utils/
```

The former robot-named environment folders (`envs/b1`, `envs/go1`, and
`envs/z1`) were removed. Robot configuration fragments now live under
`robots/configs`; the generic velocity-tracking environment now lives under
`envs/base`.

## Compatibility evidence

- After the resource directory rename, the environment config fingerprint is
  `e05f1682ed01fd52b847ebd14e4cf185de228ede91e9f42bbdf1ecdba53b5f4f`.
  The only intentional difference from the previous fingerprint is
  `Cfg.asset.file`: `resources/robots/b1` became `resources/robots/b1_z1`.
- Policy/algorithm/runner fingerprint remains
  `3971bb0ad9795963582e9f46a46121722055b5390850bee43e38798b6596df9b`.
- All 30 current unit/contract tests pass.
- The historical `2026-08-13_11-32-29_wbc_release` checkpoint strict-loads,
  exports to TorchScript, and completes a GPU headless play smoke test.

Checkpoint files contain state dictionaries and plain configuration data; they
do not pickle classes from the removed package namespaces. The serialized
`MINI_GYM_ROOT_DIR` asset placeholder is intentionally retained inside robot
configuration values. Historical saved runs remain unchanged on disk; the
config loader migrates their old asset path in memory.

After pulling the migration into an existing environment, refresh the editable
installation once:

```bash
python -m pip install -e . --no-deps
```

## Policy export ownership

`OnPolicyRunner.save()` automatically exports
`exported/policies/policy_<iteration>.pt` and refreshes `policy_latest.pt` after
every checkpoint save. `play.py` only evaluates and no longer performs a
duplicate export. `scripts/export_policy.py` remains available for manually
exporting a historical or selected checkpoint.
