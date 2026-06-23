#include <torch_gcu/csrc/profiler/orchestration/observer.h>
#include <torch_gcu/csrc/profiler/util.h>

#include <utility>

namespace torch_gcu {
namespace profiler {
namespace impl {

using GlobalManager = GlobalStateManager<ProfilerStateBase>;

// ----------------------------------------------------------------------------
// -- Profiler base class -----------------------------------------------------
// ----------------------------------------------------------------------------
/*explicit*/ ProfilerStateBase::ProfilerStateBase(ProfilerConfig config)
    : c10::MemoryReportingInfoBase(), config_(std::move(config)) {}

ProfilerStateBase::~ProfilerStateBase() {
  if (handle_) {
    auto handle = handle_;
    removeCallback();
    SOFT_ASSERT(false, "Leaked callback handle: ", handle);
  }
}

/*static*/ ProfilerStateBase* ProfilerStateBase::get(bool global) {
  auto* out =
      global ? GlobalManager::get()
             : static_cast<ProfilerStateBase*>(c10::ThreadLocalDebugInfo::get(
                   c10::DebugInfoKind::PROFILER_STATE));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!out || out->config().global() == global);
  return out;
}

/*static*/ void ProfilerStateBase::push(
    std::shared_ptr<ProfilerStateBase>&& state) {
  TORCH_INTERNAL_ASSERT(state != nullptr);
  if (state->config().global()) {
    GlobalManager::push(std::move(state));
  } else {
    c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);
  }
}

namespace {
std::shared_ptr<ProfilerStateBase> popTLS() {
  // If there is no active thread local profiler then we simply return null.
  // However if there is an active profiler but it is not the top
  // `DebugInfoBase`then `c10::ThreadLocalDebugInfo::_pop` will throw.
  // TODO(robieta): make `noexcept` version.
  return c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE)
             ? std::static_pointer_cast<ProfilerStateBase>(
                   c10::ThreadLocalDebugInfo::_pop(
                       c10::DebugInfoKind::PROFILER_STATE))
             : nullptr;
}
}  // namespace

/*static*/ std::shared_ptr<ProfilerStateBase> ProfilerStateBase::pop(
    bool global) {
  auto out = global ? GlobalManager::pop() : popTLS();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!out || out->config().global() == global);
  return out;
}

void ProfilerStateBase::setCallbackHandle(at::CallbackHandle handle) {
  if (handle_) {
    at::removeCallback(handle_);
    SOFT_ASSERT(false,
                "ProfilerStateBase already has a registered callback. "
                "Removing to avoid leaked callback.");
  }

  handle_ = handle;
}

void ProfilerStateBase::removeCallback() {
  if (handle_) {
    at::removeCallback(handle_);
    handle_ = 0;
  }
}

bool profilerEnabled() {
  auto* state_ptr = ProfilerStateBase::get(/*global=*/false);
  return state_ptr && !state_ptr->config().disabled();
}

TORCH_API ActiveProfilerType profilerType() {
  auto* state_ptr = ProfilerStateBase::get(/*global=*/false);
  return state_ptr == nullptr ? ActiveProfilerType::NONE
                              : state_ptr->profilerType();
}

torch_gcu::profiler::impl::ProfilerConfig getProfilerConfig() {
  auto* state_ptr = ProfilerStateBase::get(/*global=*/false);
  TORCH_CHECK(state_ptr,
              "Tried to access profiler config, but profiler is not enabled!");
  return state_ptr->config();
}

}  // namespace impl
}  // namespace profiler
}  // namespace torch_gcu
