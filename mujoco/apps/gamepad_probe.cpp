#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

#include "mujoco/gamepad.hpp"

int main() {
  mujoco::Gamepad gamepad;
  std::cout << "gamepad: " << gamepad.status() << '\n'
            << "Press Ctrl-C to stop. Use F710 X mode.\n";
  while (true) {
    const auto state = gamepad.poll();
    std::cout << '\r' << (state.connected ? "connected" : "disconnected")
              << std::fixed << std::setprecision(2)
              << " LX=" << state.left_x << " LY=" << state.left_y
              << " RX=" << state.right_x << " RY=" << state.right_y
              << " LT=" << state.left_trigger << " RT=" << state.right_trigger
              << " A=" << state.a << " B=" << state.b
              << " X=" << state.x << " Y=" << state.y
              << " LB=" << state.left_bumper << " RB=" << state.right_bumper
              << " DPad(LRUD)=" << state.dpad_left << state.dpad_right
              << state.dpad_up << state.dpad_down
              << "        " << std::flush;
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
  }
}
