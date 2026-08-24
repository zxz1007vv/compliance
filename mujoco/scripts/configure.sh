#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../.." && pwd)"
build_dir="${1:-${repository_root}/mujoco/build}"
torch_root="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)')"
torch_cxx11_abi="$(python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"

cmake_args=(
  -S "${repository_root}/mujoco"
  -B "${build_dir}"
  -DMUJOCO_TORCH_ROOT="${torch_root}"
  -DMUJOCO_TORCH_CXX11_ABI="${torch_cxx11_abi}"
  -DMUJOCO_BUILD_SIMULATOR=ON
  -DMUJOCO_ENABLE_SDL2=ON
  -DCMAKE_BUILD_TYPE=Release
)
if [[ -n "${MUJOCO_ROOT:-}" ]]; then
  cmake_args+=(-DMUJOCO_ROOT="${MUJOCO_ROOT}")
fi
if [[ -n "${MUJOCO_YAML_ROOT:-}" ]]; then
  cmake_args+=(-DMUJOCO_YAML_ROOT="${MUJOCO_YAML_ROOT}")
fi

cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --parallel
ctest --test-dir "${build_dir}" --output-on-failure
