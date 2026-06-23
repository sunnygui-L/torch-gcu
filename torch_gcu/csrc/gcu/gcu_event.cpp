#include "gcu/gcu_event.h"
#include "gcu/gcu_exception.h"
#include "gcu/gcu_guard.h"

namespace torch_gcu {

GCUEvent::GCUEvent(c10::DeviceIndex device_index,
                   const topsIpcEventHandle_t* handle) {
  device_index_ = device_index;
  GCUGuard guard(device_index_);

  C10_GCU_CHECK(topsIpcOpenEventHandle(&event_, *handle));
  is_created_ = true;
}

GCUEvent::~GCUEvent() {
  try {
    if (is_created_) {
      GCUGuard guard(device_index_);
      // const c10::impl::PyInterpreter* interp = GCUTrace::get_trace();
      // if (C10_UNLIKELY(interp)) {
      //   (*interp)->trace_gcu_event_deletion(
      //       reinterpret_cast<uintptr_t>(event_));
      // }
      C10_GCU_CHECK(topsEventDestroy(event_));
    }
  } catch (...) { /* No throw */
  }
}

bool GCUEvent::query() const {
  if (!is_created_) {
    return true;
  }

  topsError_t err = topsEventQuery(event_);
  if (err == topsSuccess) {
    return true;
  } else if (err != topsErrorNotReady) {
    C10_GCU_CHECK(err);
  } else {
    // ignore and clear the error if not ready
    (void)topsGetLastError();
  }

  return false;
}

void GCUEvent::record(const GCUStream& stream) {
  if (!is_created_) {
    createEvent(stream.device_index());
  }

  TORCH_CHECK(device_index_ == stream.device_index(), "Event device ",
              device_index_, " does not match recording stream's device ",
              stream.device_index(), ".");
  GCUGuard guard(device_index_);
  C10_GCU_CHECK(topsEventRecord(event_, stream));
  // const c10::impl::PyInterpreter* interp = GCUTrace::get_trace();
  // if (C10_UNLIKELY(interp)) {
  //   (*interp)->trace_gcu_event_record(
  //       reinterpret_cast<uintptr_t>(event_),
  //       reinterpret_cast<uintptr_t>(stream.stream()));
  // }
  was_recorded_ = true;
}

void GCUEvent::block(const GCUStream& stream) {
  if (is_created_) {
    GCUGuard guard(stream.device_index());
    C10_GCU_CHECK(topsStreamWaitEvent(stream, event_, 0));
    // const c10::impl::PyInterpreter* interp = GCUTrace::get_trace();
    // if (C10_UNLIKELY(interp)) {
    //   (*interp)->trace_gcu_event_wait(
    //       reinterpret_cast<uintptr_t>(event_),
    //       reinterpret_cast<uintptr_t>(stream.stream()));
    // }
  }
}

float GCUEvent::elapsed_time(const GCUEvent& other) const {
  TORCH_CHECK(is_created_ && other.isCreated(),
              "Both events must be recorded before calculating elapsed time.");
  float time_ms = 0;
  // raise topsErrorNotReady if either event is recorded but not yet completed
  C10_GCU_CHECK(topsEventElapsedTime(&time_ms, event_, other.event_));
  return time_ms;
}

// Note: topsEventSynchronize can be safely called from any device
void GCUEvent::synchronize() const {
  if (is_created_) {
    // const c10::impl::PyInterpreter* interp = GCUTrace::get_trace();
    // if (C10_UNLIKELY(interp)) {
    //   (*interp)->trace_gcu_event_synchronization(
    //       reinterpret_cast<uintptr_t>(event_));
    // }
    C10_GCU_CHECK(topsEventSynchronize(event_));
  }
}

void GCUEvent::ipc_handle(topsIpcEventHandle_t* handle) {
  if (!is_created_) {
    // this GCUEvent object was initially constructed from flags but event_
    // is not created yet.
    createEvent(getCurrentGCUStream().device_index());
  }
  GCUGuard guard(device_index_);
  C10_GCU_CHECK(topsIpcGetEventHandle(handle, event_));
}

void GCUEvent::createEvent(c10::DeviceIndex device_index) {
  device_index_ = device_index;
  GCUGuard guard(device_index_);
  C10_GCU_CHECK(topsEventCreateWithFlags(&event_, flags_));
  // const c10::impl::PyInterpreter* interp = GCUTrace::get_trace();
  // if (C10_UNLIKELY(interp)) {
  //   (*interp)->trace_gcu_event_creation(reinterpret_cast<uintptr_t>(event_));
  // }
  is_created_ = true;
}

}  // namespace torch_gcu