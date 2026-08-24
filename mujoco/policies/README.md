# Policy deployment bundles

This directory is for generated C++ policy bundles and is ignored by Git except
for this README. The simulator currently uses LibTorch, so the neural-network
file is an exported TorchScript `policy.pt`, not an ONNX file. Each task folder
also contains `runtime.cfg`, `manifest.json`, golden parity vectors, checksums,
and the source training configuration; keep these files together.

From the repository root, export the current policies with:

```bash
python scripts/export_deployment_bundle.py \
  --run-dir logs/zgwsarm_compliance/<run> --checkpoint latest \
  --output-dir mujoco/policies/zgwsarm

python scripts/export_deployment_bundle.py \
  --run-dir logs/b1_z1_ik/<run> --checkpoint latest \
  --output-dir mujoco/policies/b1_z1
```

Do not copy only `policy.pt`: `runtime.cfg` is the exact trained interface used
to validate observation dimensions, joint order, action scaling, PD gains,
limits, and control timing.
