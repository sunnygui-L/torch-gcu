#pragma once

#if defined(__ANDROID__) || defined(__linux__)

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <limits>

#endif /* __ANDROID__ || __linux__ */

#include <torch_gcu/csrc/profiler/perf.h>

namespace torch_gcu::profiler::impl::linux_perf {

/*
 * PerfEvent
 * ---------
 */

inline void PerfEvent::Disable() const {
#if defined(__ANDROID__) || defined(__linux__)
  ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
#endif /* __ANDROID__ || __linux__ */
}

inline void PerfEvent::Enable() const {
#if defined(__ANDROID__) || defined(__linux__)
  ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
#endif /* __ANDROID__ || __linux__ */
}

inline void PerfEvent::Reset() const {
#if defined(__ANDROID__) || defined(__linux__)
  ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
#endif /* __ANDROID__ || __linux__ */
}

/*
 * PerfProfiler
 * ------------
 */

inline uint64_t PerfProfiler::CalcDelta(uint64_t start, uint64_t end) const {
  if (end < start) {  // overflow
    return end + (std::numeric_limits<uint64_t>::max() - start);
  }
  // not possible to wrap around start for a 64b cycle counter
  return end - start;
}

inline void PerfProfiler::StartCounting() const {
  for (auto& e : events_) {
    e.Enable();
  }
}

inline void PerfProfiler::StopCounting() const {
  for (auto& e : events_) {
    e.Disable();
  }
}

}  // namespace torch_gcu::profiler::impl::linux_perf
