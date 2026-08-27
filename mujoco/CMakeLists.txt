cmake_minimum_required(VERSION 3.18)
project(mujoco_sim2sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MUJOCO_BUILD_SIMULATOR "Build the MuJoCo simulator executable" ON)
option(MUJOCO_ENABLE_SDL2 "Enable SDL2 game-controller input" ON)
option(MUJOCO_BUILD_TESTS "Build C++ contract/parity tests" ON)
set(MUJOCO_TORCH_ROOT "" CACHE PATH
    "Torch Python package root; bypasses CUDA checks and links CPU inference libraries directly")
set(MUJOCO_TORCH_CXX11_ABI "0" CACHE STRING "_GLIBCXX_USE_CXX11_ABI value of the Torch package")
set(MUJOCO_ROOT "" CACHE PATH "MuJoCo binary distribution root")
set(MUJOCO_YAML_ROOT "" CACHE PATH "libyaml development package root")

find_path(MUJOCO_YAML_INCLUDE_DIR yaml.h
  HINTS ${MUJOCO_YAML_ROOT} $ENV{MUJOCO_YAML_ROOT} PATH_SUFFIXES include)
find_library(MUJOCO_YAML_LIBRARY NAMES yaml
  HINTS ${MUJOCO_YAML_ROOT} $ENV{MUJOCO_YAML_ROOT} PATH_SUFFIXES lib)
if(NOT MUJOCO_YAML_INCLUDE_DIR OR NOT MUJOCO_YAML_LIBRARY)
  message(FATAL_ERROR "libyaml was not found; install libyaml-dev or set MUJOCO_YAML_ROOT")
endif()
add_library(mujoco_yaml UNKNOWN IMPORTED)
set_target_properties(mujoco_yaml PROPERTIES
  IMPORTED_LOCATION "${MUJOCO_YAML_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_YAML_INCLUDE_DIR}")

if(MUJOCO_TORCH_ROOT)
  if(NOT EXISTS "${MUJOCO_TORCH_ROOT}/include/torch/script.h" OR
     NOT EXISTS "${MUJOCO_TORCH_ROOT}/lib/libtorch_cpu.so")
    message(FATAL_ERROR "MUJOCO_TORCH_ROOT does not point to a complete Torch package")
  endif()
  add_library(mujoco_torch_cpu SHARED IMPORTED)
  set_target_properties(mujoco_torch_cpu PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_TORCH_ROOT}/lib/libtorch_cpu.so"
    INTERFACE_INCLUDE_DIRECTORIES
      "${MUJOCO_TORCH_ROOT}/include;${MUJOCO_TORCH_ROOT}/include/torch/csrc/api/include")
  add_library(mujoco_c10 SHARED IMPORTED)
  set_target_properties(mujoco_c10 PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_TORCH_ROOT}/lib/libc10.so"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_TORCH_ROOT}/include")
  set(MUJOCO_TORCH_LIBRARIES mujoco_torch_cpu mujoco_c10)
  # The repository's PyTorch 2.0 conda environment is built with the old ABI.
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${MUJOCO_TORCH_CXX11_ABI})
  add_compile_definitions(MUJOCO_TORCH_PYTHON_PACKAGE=1)
  set(CMAKE_BUILD_RPATH "${MUJOCO_TORCH_ROOT}/lib")
else()
  find_package(Torch REQUIRED)
  set(MUJOCO_TORCH_LIBRARIES ${TORCH_LIBRARIES})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

add_library(mujoco_core
  src/runtime_config.cpp
  src/launcher_config.cpp
  src/command.cpp
  src/observation.cpp
  src/control.cpp
  src/gamepad.cpp
  src/policy.cpp)
target_include_directories(mujoco_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mujoco_core PUBLIC ${MUJOCO_TORCH_LIBRARIES} mujoco_yaml)
target_compile_options(mujoco_core PRIVATE -Wall -Wextra -Wpedantic)

if(MUJOCO_ENABLE_SDL2)
  find_package(SDL2 QUIET CONFIG)
  if(SDL2_FOUND)
    target_compile_definitions(mujoco_core PUBLIC MUJOCO_HAS_SDL2=1)
    if(TARGET SDL2::SDL2)
      target_link_libraries(mujoco_core PUBLIC SDL2::SDL2)
    else()
      target_include_directories(mujoco_core PUBLIC ${SDL2_INCLUDE_DIRS})
      target_link_libraries(mujoco_core PUBLIC ${SDL2_LIBRARIES})
    endif()
  else()
    message(WARNING "SDL2 was not found: gamepad support will use the diagnostic stub")
  endif()
endif()

add_executable(mujoco_policy_parity apps/policy_parity.cpp)
target_link_libraries(mujoco_policy_parity PRIVATE mujoco_core)

add_executable(mujoco_gamepad_probe apps/gamepad_probe.cpp)
target_link_libraries(mujoco_gamepad_probe PRIVATE mujoco_core)

if(MUJOCO_BUILD_TESTS)
  enable_testing()
  add_executable(mujoco_contract_tests tests/contract_tests.cpp)
  target_link_libraries(mujoco_contract_tests PRIVATE mujoco_core)
  target_compile_definitions(mujoco_contract_tests PRIVATE
    MUJOCO_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME mujoco_contract_tests COMMAND mujoco_contract_tests)
endif()

if(MUJOCO_BUILD_SIMULATOR)
  find_package(mujoco QUIET CONFIG)
  if(NOT TARGET mujoco::mujoco)
    find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h HINTS ${MUJOCO_ROOT} $ENV{MUJOCO_ROOT} PATH_SUFFIXES include)
    find_library(MUJOCO_LIBRARY NAMES mujoco HINTS ${MUJOCO_ROOT} $ENV{MUJOCO_ROOT} PATH_SUFFIXES lib bin)
    if(MUJOCO_INCLUDE_DIR AND MUJOCO_LIBRARY)
      add_library(mujoco::mujoco UNKNOWN IMPORTED)
      set_target_properties(mujoco::mujoco PROPERTIES
        IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}")
    endif()
  endif()

  if(TARGET mujoco::mujoco)
    add_library(mujoco_simulator src/simulator.cpp)
    target_include_directories(mujoco_simulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(mujoco_simulator PUBLIC mujoco_core mujoco::mujoco)
    target_compile_options(mujoco_simulator PRIVATE
      -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
    if(MUJOCO_BUILD_TESTS)
      add_executable(mujoco_simulator_tests tests/simulator_tests.cpp)
      target_link_libraries(mujoco_simulator_tests PRIVATE mujoco_simulator)
      target_compile_definitions(mujoco_simulator_tests PRIVATE
        MUJOCO_REPOSITORY_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
      add_test(NAME mujoco_simulator_tests COMMAND mujoco_simulator_tests)
    endif()
    find_package(glfw3 QUIET)
    set(MUJOCO_SIMULATE_SOURCE_DIR "")
    foreach(candidate
        "${MUJOCO_ROOT}/simulate"
        "$ENV{MUJOCO_ROOT}/simulate"
        "${MUJOCO_INCLUDE_DIR}/../simulate")
      if(EXISTS "${candidate}/simulate.cc" AND EXISTS "${candidate}/glfw_adapter.cc")
        get_filename_component(MUJOCO_SIMULATE_SOURCE_DIR "${candidate}" ABSOLUTE)
        break()
      endif()
    endforeach()
    if(TARGET glfw AND MUJOCO_SIMULATE_SOURCE_DIR)
      find_package(Threads REQUIRED)
      add_library(mujoco_classic_viewer
        "${MUJOCO_SIMULATE_SOURCE_DIR}/simulate.cc"
        "${MUJOCO_SIMULATE_SOURCE_DIR}/platform_ui_adapter.cc"
        "${MUJOCO_SIMULATE_SOURCE_DIR}/glfw_adapter.cc"
        "${MUJOCO_SIMULATE_SOURCE_DIR}/glfw_dispatch.cc"
        third_party/lodepng_compat/lodepng.cpp)
      target_include_directories(mujoco_classic_viewer PUBLIC
        "${MUJOCO_SIMULATE_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lodepng_compat")
      target_link_libraries(mujoco_classic_viewer PUBLIC
        mujoco::mujoco glfw Threads::Threads)
      target_compile_options(mujoco_classic_viewer PRIVATE
        -Wno-unused-parameter -Wno-missing-field-initializers)
      target_compile_definitions(mujoco_simulator PUBLIC MUJOCO_HAS_CLASSIC_VIEWER=1)
      target_link_libraries(mujoco_simulator PUBLIC mujoco_classic_viewer)
    else()
      message(WARNING
        "GLFW or the MuJoCo simulate sources were not found: --viewer is unavailable. "
        "Use the complete MuJoCo binary release, whose root contains simulate/.")
    endif()

    add_executable(mujoco_sim apps/mujoco_sim.cpp)
    target_link_libraries(mujoco_sim PRIVATE mujoco_simulator)

    add_executable(mujoco_diagnostic apps/diagnostic.cpp)
    target_link_libraries(mujoco_diagnostic PRIVATE mujoco_simulator)
  else()
    message(WARNING "MuJoCo was not found: mujoco_sim is disabled; core/parity tests remain available")
  endif()
endif()
