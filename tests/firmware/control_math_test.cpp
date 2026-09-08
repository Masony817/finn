#include <cmath>
#include <cstdio>
#include <limits>

#include "control_math.h"

namespace {

bool nearlyEqual(const float actual, const float expected) {
  return std::fabs(actual - expected) < 1.0e-6f;
}

bool check(const bool condition, const char* const description) {
  if (condition) return true;
  std::fprintf(stderr, "failed: %s\n", description);
  return false;
}

}  // namespace

int main() {
  const FinnLqrControl::CommandLimits limits = {0.6f, 1.0f, 0.5f, 3.0f, 500U};
  FinnLqrControl::CommandArbiter arbiter;

  if (!check(
          FinnLqrControl::submitCommand(&arbiter, 2.0f, -2.0f, 100U, limits),
          "finite commands are accepted")) {
    return 1;
  }
  if (!check(
          nearlyEqual(arbiter.requested_forward_m_s, 0.6f) &&
              nearlyEqual(arbiter.requested_yaw_rad_s, -1.0f),
          "commands are clamped to the reviewed limits")) {
    return 1;
  }

  if (!check(
          FinnLqrControl::stepCommandArbiter(&arbiter, 100U, 0.1f, limits),
          "fresh command steps")) {
    return 1;
  }
  if (!check(
          nearlyEqual(arbiter.forward_m_s, 0.05f) && nearlyEqual(arbiter.yaw_rad_s, -0.3f) &&
              !arbiter.stale,
          "command output is slew limited")) {
    return 1;
  }

  if (!check(
          FinnLqrControl::stepCommandArbiter(&arbiter, 600U, 0.1f, limits),
          "command is held through its timeout boundary")) {
    return 1;
  }
  if (!check(
          nearlyEqual(arbiter.forward_m_s, 0.1f) && nearlyEqual(arbiter.yaw_rad_s, -0.6f) &&
              !arbiter.stale,
          "intermittent commands do not disturb balance before timeout")) {
    return 1;
  }

  if (!check(
          FinnLqrControl::stepCommandArbiter(&arbiter, 601U, 0.1f, limits),
          "expired command ramps down")) {
    return 1;
  }
  if (!check(
          nearlyEqual(arbiter.forward_m_s, 0.05f) && nearlyEqual(arbiter.yaw_rad_s, -0.3f) &&
              arbiter.stale,
          "expired command fails soft by slewing to zero")) {
    return 1;
  }

  const uint32_t last_command_ms = arbiter.last_command_ms;
  if (!check(
          !FinnLqrControl::submitCommand(
              &arbiter, std::numeric_limits<float>::quiet_NaN(), 0.0f, 700U, limits),
          "NaN command is rejected")) {
    return 1;
  }
  if (!check(
          arbiter.last_command_ms == last_command_ms &&
              FinnLqrControl::isFinite(arbiter.forward_m_s) &&
              FinnLqrControl::isFinite(arbiter.yaw_rad_s),
          "invalid command cannot refresh or poison the held command")) {
    return 1;
  }

  const FinnLqrControl::WheelTorques priority =
      FinnLqrControl::allocateWheelTorques(0.8f, 0.9f, 1.0f);
  if (!check(
          priority.valid && nearlyEqual(priority.common_nm, 0.8f) &&
              nearlyEqual(priority.yaw_nm, 0.2f),
          "yaw receives only balance headroom")) {
    return 1;
  }
  const FinnLqrControl::WheelTorques saturated =
      FinnLqrControl::allocateWheelTorques(2.0f, -0.5f, 1.0f);
  if (!check(
          saturated.valid && nearlyEqual(saturated.common_nm, 1.0f) &&
              nearlyEqual(saturated.yaw_nm, 0.0f),
          "saturated balance torque leaves no yaw authority")) {
    return 1;
  }
  const FinnLqrControl::WheelTorques invalid = FinnLqrControl::allocateWheelTorques(
      std::numeric_limits<float>::quiet_NaN(), 0.1f, 1.0f);
  if (!check(
          !invalid.valid && nearlyEqual(invalid.common_nm, 0.0f) && nearlyEqual(invalid.yaw_nm, 0.0f),
          "NaN torque allocation is invalid and inert")) {
    return 1;
  }
  FinnLqrControl::ActiveControlState active_state = {
      1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 25.0f, 0.0f, 0.0f, 25.0f, 0.0f, 0.0f, 0.0f,
  };
  if (!check(
          FinnLqrControl::activeControlStateIsFinite(active_state),
          "finite active control state is accepted")) {
    return 1;
  }
  active_state.left_velocity_rev_s = std::numeric_limits<float>::infinity();
  if (!check(
          !FinnLqrControl::activeControlStateIsFinite(active_state),
          "non-finite sensor state is rejected before control arithmetic")) {
    return 1;
  }
  return 0;
}
