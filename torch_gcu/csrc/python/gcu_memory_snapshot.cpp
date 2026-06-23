#include "python/gcu_memory_snapshot.h"

#include <ATen/Context.h>

#include "gcu/gcu_caching_allocator.h"
#ifdef USE_KINETO_GCU
#include "profiler/combined_traceback.h"
#else
#include <torch/csrc/profiler/combined_traceback.h>
using CapturedTraceback = torch::CapturedTraceback;
#endif

namespace torch_gcu {

namespace {

std::shared_ptr<c10::GatheredContext> gather() {
  return CapturedTraceback::gather(true, true, false);
}

std::shared_ptr<c10::GatheredContext> gather_with_cpp() {
  return CapturedTraceback::gather(true, true, true);
}

}  // namespace

void _record_memory_history(bool enabled, bool record_context,
                            int64_t trace_alloc_max_entries,
                            bool trace_alloc_record_context,
                            bool record_cpp_context) {
  GCUCachingAllocator::CreateContextFn recorder = gather;
  if (enabled && record_cpp_context) {
    recorder = gather_with_cpp;
// warm up C++ stack unwinding
#ifdef USE_KINETO_GCU
    unwind::unwind();
#else
    torch::unwind::unwind();
#endif
  }
  auto when = GCUCachingAllocator::RecordContext::NEVER;
  if (trace_alloc_record_context) {
    when = GCUCachingAllocator::RecordContext::ALLOC;
  } else if (record_context) {
    when = GCUCachingAllocator::RecordContext::STATE;
  }
  at::globalContext().lazyInitDevice(at::kPrivateUse1);
  GCUCachingAllocator::recordHistory(enabled, recorder, trace_alloc_max_entries,
                                     when);
}

static void checkOptionIn(const std::string& option,
                          std::initializer_list<std::string> valid,
                          const char* error) {
  TORCH_CHECK(valid.end() != std::find(valid.begin(), valid.end(), option),
              error);
}

void _record_memory_history(c10::optional<std::string> enabled,
                            c10::optional<std::string> context,
                            std::string stacks, size_t max_entries) {
  if (enabled) {
    checkOptionIn(*enabled, {"state", "all"},
                  "expected state to be 'state', 'all', or None");
  }
  if (context) {
    checkOptionIn(*context, {"state", "alloc", "all"},
                  "expected context to be 'state', 'alloc', 'all', or None");
  }
  checkOptionIn(stacks, {"python", "all"},
                "expected stacks to be 'python', or 'all'");

  GCUCachingAllocator::CreateContextFn recorder = gather;
  if (enabled && stacks == "all") {
    recorder = gather_with_cpp;
// warm up C++ stack unwinding
#ifdef USE_KINETO_GCU
    unwind::unwind();
#else
    torch::unwind::unwind();
#endif
  }
  max_entries = (enabled && *enabled == "all") ? max_entries : 1;
  auto when = GCUCachingAllocator::RecordContext::NEVER;
  if (context) {
    if (context == "all") {
      when = GCUCachingAllocator::RecordContext::ALL;
    } else if (context == "alloc") {
      when = GCUCachingAllocator::RecordContext::ALLOC;
    } else if (context == "state") {
      when = GCUCachingAllocator::RecordContext::STATE;
    }
  }
  at::globalContext().lazyInitDevice(at::kPrivateUse1);
  GCUCachingAllocator::recordHistory(enabled.has_value(), recorder, max_entries,
                                     when);
}

}  // namespace torch_gcu
