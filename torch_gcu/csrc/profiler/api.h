#pragma once

#include "torch_gcu/csrc/profiler/orchestration/observer.h"

// There are some components which use these symbols. Until we migrate them
// we have to mirror them in the old autograd namespace.

namespace torch_gcu::autograd::profiler {
using torch_gcu::profiler::impl::ActivityType;
using torch_gcu::profiler::impl::getProfilerConfig;
using torch_gcu::profiler::impl::ProfilerConfig;
using torch_gcu::profiler::impl::profilerEnabled;
using torch_gcu::profiler::impl::ProfilerState;
}  // namespace torch_gcu::autograd::profiler
