#pragma once

#include <math.h>
#include <stdint.h>

namespace FinnLqrControl {

struct CommandLimits {
  float max_forward_vel_m_s;
  float max_yaw_rate_rad_s;
  float forward_accel_limit_m_s2;
  float yaw_accel_limit_rad_s2;
  uint32_t timeout_ms;
};

struct CommandArbiter {
  float requested_forward_m_s = 0.0f;
  float requested_yaw_rad_s = 0.0f;
  float forward_m_s = 0.0f;
  float yaw_rad_s = 0.0f;
  uint32_t last_command_ms = 0;
  bool has_command = false;
  bool stale = true;
};

struct WheelTorques {
  float common_nm = 0.0f;
  float yaw_nm = 0.0f;
  bool valid = false;
};

struct ActiveControlState {
  float quat_real;
  float quat_i;
  float quat_j;
  float quat_k;
  float gyro_x_rad_s;
  float gyro_y_rad_s;
  float pitch_zero_raw_rad;
  float left_position_rev;
  float left_velocity_rev_s;
  float left_temperature_c;
  float right_position_rev;
  float right_velocity_rev_s;
  float right_temperature_c;
  float start_left_position_rev;
  float start_right_position_rev;
  float reference_forward_pos_m;
};

inline bool isFinite(const float value) { return ::isfinite(value); }

inline bool activeControlStateIsFinite(const ActiveControlState& state) {
  return isFinite(state.quat_real) && isFinite(state.quat_i) && isFinite(state.quat_j) &&
         isFinite(state.quat_k) && isFinite(state.gyro_x_rad_s) &&
         isFinite(state.gyro_y_rad_s) && isFinite(state.pitch_zero_raw_rad) &&
         isFinite(state.left_position_rev) && isFinite(state.left_velocity_rev_s) &&
         isFinite(state.left_temperature_c) && isFinite(state.right_position_rev) &&
         isFinite(state.right_velocity_rev_s) && isFinite(state.right_temperature_c) &&
         isFinite(state.start_left_position_rev) && isFinite(state.start_right_position_rev) &&
         isFinite(state.reference_forward_pos_m);
}

inline float clampFloat(const float value, const float lower, const float upper) {
  if (value < lower) return lower;
  if (value > upper) return upper;
  return value;
}

inline float slewFloat(const float current, const float target, const float max_step) {
  return current + clampFloat(target - current, -max_step, max_step);
}

inline bool commandLimitsAreValid(const CommandLimits& limits) {
  return isFinite(limits.max_forward_vel_m_s) && isFinite(limits.max_yaw_rate_rad_s) &&
         isFinite(limits.forward_accel_limit_m_s2) && isFinite(limits.yaw_accel_limit_rad_s2) &&
         limits.max_forward_vel_m_s >= 0.0f && limits.max_yaw_rate_rad_s >= 0.0f &&
         limits.forward_accel_limit_m_s2 >= 0.0f && limits.yaw_accel_limit_rad_s2 >= 0.0f;
}

inline void resetCommandArbiter(CommandArbiter* const arbiter) {
  *arbiter = CommandArbiter();
}

inline bool submitCommand(
    CommandArbiter* const arbiter, const float forward_m_s, const float yaw_rad_s,
    const uint32_t now_ms, const CommandLimits& limits) {
  if (!isFinite(forward_m_s) || !isFinite(yaw_rad_s) || !commandLimitsAreValid(limits)) {
    return false;
  }
  arbiter->requested_forward_m_s =
      clampFloat(forward_m_s, -limits.max_forward_vel_m_s, limits.max_forward_vel_m_s);
  arbiter->requested_yaw_rad_s =
      clampFloat(yaw_rad_s, -limits.max_yaw_rate_rad_s, limits.max_yaw_rate_rad_s);
  arbiter->last_command_ms = now_ms;
  arbiter->has_command = true;
  arbiter->stale = false;
  return true;
}

inline bool stepCommandArbiter(
    CommandArbiter* const arbiter, const uint32_t now_ms, const float dt_s,
    const CommandLimits& limits) {
  if (!isFinite(dt_s) || dt_s < 0.0f || !commandLimitsAreValid(limits) ||
      !isFinite(arbiter->requested_forward_m_s) || !isFinite(arbiter->requested_yaw_rad_s) ||
      !isFinite(arbiter->forward_m_s) || !isFinite(arbiter->yaw_rad_s)) {
    resetCommandArbiter(arbiter);
    return false;
  }
  const bool expired = !arbiter->has_command || (now_ms - arbiter->last_command_ms) > limits.timeout_ms;
  if (expired) {
    arbiter->stale = true;
    arbiter->requested_forward_m_s = 0.0f;
    arbiter->requested_yaw_rad_s = 0.0f;
  }
  arbiter->forward_m_s = slewFloat(
      arbiter->forward_m_s, arbiter->requested_forward_m_s, limits.forward_accel_limit_m_s2 * dt_s);
  arbiter->yaw_rad_s = slewFloat(
      arbiter->yaw_rad_s, arbiter->requested_yaw_rad_s, limits.yaw_accel_limit_rad_s2 * dt_s);
  return isFinite(arbiter->forward_m_s) && isFinite(arbiter->yaw_rad_s);
}

inline WheelTorques allocateWheelTorques(
    const float tau_balance_nm, const float tau_yaw_nm, const float torque_limit_nm) {
  WheelTorques result;
  if (!isFinite(tau_balance_nm) || !isFinite(tau_yaw_nm) || !isFinite(torque_limit_nm) ||
      torque_limit_nm < 0.0f) {
    return result;
  }
  result.common_nm = clampFloat(tau_balance_nm, -torque_limit_nm, torque_limit_nm);
  const float yaw_headroom_nm = torque_limit_nm - fabsf(result.common_nm);
  result.yaw_nm = clampFloat(tau_yaw_nm, -yaw_headroom_nm, yaw_headroom_nm);
  result.valid = true;
  return result;
}

}  // namespace FinnLqrControl
