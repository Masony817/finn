#include <cstdio>

#include "control_math.h"

int main() {
  const FinnLqrControl::CommandLimits limits = {0.6f, 1.0f, 0.5f, 3.0f, 500U};
  FinnLqrControl::CommandArbiter arbiter;
  unsigned int now_ms;
  int submit;
  float forward, yaw;
  while (std::scanf("%u %d %f %f", &now_ms, &submit, &forward, &yaw) == 4) {
    if (submit) {
      FinnLqrControl::submitCommand(&arbiter, forward, yaw, now_ms, limits);
    }
    if (!FinnLqrControl::stepCommandArbiter(&arbiter, now_ms, 0.01f, limits)) return 1;
    std::printf("%.9g %.9g %d\n", arbiter.forward_m_s, arbiter.yaw_rad_s, arbiter.stale);
  }
  return 0;
}
