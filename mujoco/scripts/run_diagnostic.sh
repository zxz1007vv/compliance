#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../.." && pwd)"
diagnostic_binary="${repository_root}/mujoco/build/mujoco_diagnostic"

if [[ ! -x "${diagnostic_binary}" ]]; then
  echo "mujoco_diagnostic is not built. Run: mujoco/scripts/configure.sh" >&2
  exit 1
fi

exec "${diagnostic_binary}" "$@"
