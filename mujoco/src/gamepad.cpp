#include "mujoco/gamepad.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef MUJOCO_HAS_SDL2
#include <SDL.h>
#endif

namespace mujoco {

float ApplyDeadzone(float value, float deadzone) {
  const float magnitude = std::abs(value);
  if (magnitude <= deadzone) return 0.0f;
  return std::copysign((magnitude - deadzone) / (1.0f - deadzone), value);
}

struct Gamepad::Impl {
#ifdef MUJOCO_HAS_SDL2
  SDL_GameController* controller = nullptr;
  SDL_JoystickID instance = -1;
  bool initialized = false;
  std::string message = "no controller";

  void open_first() {
    if (controller) return;
    for (int index = 0; index < SDL_NumJoysticks(); ++index) {
      if (!SDL_IsGameController(index)) continue;
      controller = SDL_GameControllerOpen(index);
      if (controller) {
        instance = SDL_JoystickInstanceID(SDL_GameControllerGetJoystick(controller));
        message = SDL_GameControllerName(controller);
        return;
      }
    }
    message = "no SDL2-compatible controller (F710 switch should be in X mode)";
  }

  void close() {
    if (controller) SDL_GameControllerClose(controller);
    controller = nullptr;
    instance = -1;
  }
#else
  std::string message = "SDL2 support was not compiled";
#endif
};

Gamepad::Gamepad() : impl_(std::make_unique<Impl>()) {
#ifdef MUJOCO_HAS_SDL2
  if (SDL_InitSubSystem(SDL_INIT_GAMECONTROLLER | SDL_INIT_JOYSTICK) != 0) {
    impl_->message = std::string("SDL2 initialization failed: ") + SDL_GetError();
    return;
  }
  impl_->initialized = true;
  SDL_GameControllerEventState(SDL_ENABLE);
  impl_->open_first();
#endif
}

Gamepad::~Gamepad() {
#ifdef MUJOCO_HAS_SDL2
  impl_->close();
  if (impl_->initialized) SDL_QuitSubSystem(SDL_INIT_GAMECONTROLLER | SDL_INIT_JOYSTICK);
#endif
}

GamepadState Gamepad::poll() {
  GamepadState state;
#ifdef MUJOCO_HAS_SDL2
  if (!impl_->initialized) return state;
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_CONTROLLERDEVICEADDED) impl_->open_first();
    if (event.type == SDL_CONTROLLERDEVICEREMOVED && event.cdevice.which == impl_->instance) {
      impl_->close();
      impl_->message = "controller disconnected";
      impl_->open_first();
    }
  }
  if (!impl_->controller || !SDL_GameControllerGetAttached(impl_->controller)) return state;
  const auto axis = [&](SDL_GameControllerAxis id) {
    const int value = SDL_GameControllerGetAxis(impl_->controller, id);
    return std::clamp(value / 32767.0f, -1.0f, 1.0f);
  };
  const auto trigger = [&](SDL_GameControllerAxis id) {
    return std::clamp(axis(id), 0.0f, 1.0f);
  };
  const auto button = [&](SDL_GameControllerButton id) {
    return SDL_GameControllerGetButton(impl_->controller, id) != 0;
  };
  state.connected = true;
  state.left_x = axis(SDL_CONTROLLER_AXIS_LEFTX);
  state.left_y = axis(SDL_CONTROLLER_AXIS_LEFTY);
  state.right_x = axis(SDL_CONTROLLER_AXIS_RIGHTX);
  state.right_y = axis(SDL_CONTROLLER_AXIS_RIGHTY);
  state.left_trigger = trigger(SDL_CONTROLLER_AXIS_TRIGGERLEFT);
  state.right_trigger = trigger(SDL_CONTROLLER_AXIS_TRIGGERRIGHT);
  state.a = button(SDL_CONTROLLER_BUTTON_A);
  state.b = button(SDL_CONTROLLER_BUTTON_B);
  state.x = button(SDL_CONTROLLER_BUTTON_X);
  state.y = button(SDL_CONTROLLER_BUTTON_Y);
  state.left_bumper = button(SDL_CONTROLLER_BUTTON_LEFTSHOULDER);
  state.right_bumper = button(SDL_CONTROLLER_BUTTON_RIGHTSHOULDER);
  state.dpad_left = button(SDL_CONTROLLER_BUTTON_DPAD_LEFT);
  state.dpad_right = button(SDL_CONTROLLER_BUTTON_DPAD_RIGHT);
  state.dpad_up = button(SDL_CONTROLLER_BUTTON_DPAD_UP);
  state.dpad_down = button(SDL_CONTROLLER_BUTTON_DPAD_DOWN);
#endif
  return state;
}

bool Gamepad::available() const {
#ifdef MUJOCO_HAS_SDL2
  return impl_->controller && SDL_GameControllerGetAttached(impl_->controller);
#else
  return false;
#endif
}

std::string Gamepad::status() const { return impl_->message; }

}  // namespace mujoco
