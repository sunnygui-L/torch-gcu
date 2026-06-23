#include <ATen/MapAllocator.h>
#include <gcu/gcu_guard.h>
#include <python/gcu_ipc_types.h>

#include <atomic>
#include <map>
#include <mutex>
#include <string>

#include "gcu/gcu_functions.h"
#include "gcu/gcu_guard_impl.h"

namespace torch_gcu {

namespace {

void warnProducerTerminatedBeforeSharedTensorsReleased() {
  static bool warned = false;
  if (!warned) {
    LOG(WARNING) << "Producer process has been terminated before all shared "
                    "GCU tensors released. See Note [Sharing GCU tensors]";
    warned = true;
  }
}

struct GcuIPCGlobalEntities {
  // This class is used as a singleton (see gcu_ipc_global_entities)
  // This variable is used to track its lifetime to avoid accessing it
  // after it was destroyed which would lead to segmentation faults
  // Note that a trvial type is used which doesn't suffer from construction
  // and destruction order issues
  static bool alive;

  std::mutex ref_counters_mutex_;
  std::atomic<int64_t> sync_events_used_{0};
  std::map<std::string, std::shared_ptr<GcuIPCRefCountersFile>>
      ref_counters_files_;
  std::shared_ptr<GcuIPCRefCountersFile> next_available_ref_counters_file_;
  GcuIPCSentDataLimbo GcuIPCSentDataLimbo_;
  GcuIPCGlobalEntities() { alive = true; }
  ~GcuIPCGlobalEntities() {
    GcuIPCSentDataLimbo_.collect();
    safe_clean_current_file();
    if (next_available_ref_counters_file_) {
      warnProducerTerminatedBeforeSharedTensorsReleased();
    }
    alive = false;
  }
  void safe_clean_current_file() {
    std::lock_guard<std::mutex> lock(ref_counters_mutex_);
    if (next_available_ref_counters_file_ &&
        next_available_ref_counters_file_->offsets_in_use() == 0) {
      ref_counters_files_.erase(next_available_ref_counters_file_->handle());
      next_available_ref_counters_file_.reset();
    }
  }
};

bool GcuIPCGlobalEntities::alive = false;
GcuIPCGlobalEntities gcu_ipc_global_entities;

GcuIPCSentDataLimbo::~GcuIPCSentDataLimbo() {
  collect();
  if (size() > 0) {
    warnProducerTerminatedBeforeSharedTensorsReleased();
  }
}

bool GcuIPCSentDataLimbo::collect() {
  bool freed_memory = false;
  std::vector<std::unique_ptr<GcuIPCSentData>> reset_blocks;
  {  // Begin critical section to modify shared blocks
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    std::vector<std::unique_ptr<GcuIPCSentData>> kept_blocks;
    for (auto& sd : shared_blocks_) {
      if (sd->counter_value() > 0) {
        kept_blocks.push_back(std::move(sd));
      } else {
        freed_memory = true;
        reset_blocks.push_back(std::move(sd));
      }
    }
    shared_blocks_ = std::move(kept_blocks);
  }
  // Need to reset blocks out of the critical section here, otherwise it
  // deadlocks.
  for (auto& sd : reset_blocks) {
    sd.reset();
  }
  return freed_memory;
}

void GcuIPCSentDataLimbo::add(std::unique_ptr<GcuIPCSentData> shared_block) {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  static bool warned = false;
  if (shared_blocks_.size() > GCU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO && !warned) {
    LOG(WARNING) << "Producer process tried to deallocate over "
                 << GCU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO
                 << " memory blocks referred by consumer processes. "
                    "Deallocation might be significantly slowed down. "
                 << "We assume it will never going to be the case, but if it "
                    "is, please file but to https://github.com/pytorch/pytorch";
    warned = true;
  }
  shared_blocks_.push_back(std::move(shared_block));
}

uint64_t GcuIPCSentDataLimbo::size() {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  return shared_blocks_.size();
}

void GcuIPCSentDataDelete(void* ptr) {
  std::unique_ptr<GcuIPCSentData> sent_data(static_cast<GcuIPCSentData*>(ptr));
  if (!GcuIPCGlobalEntities::alive) {
    return;
  }
  if (sent_data->counter_value() > 0) {
    gcu_ipc_global_entities.GcuIPCSentDataLimbo_.add(std::move(sent_data));
  }
  gcu_ipc_global_entities.GcuIPCSentDataLimbo_.collect();
}

void ReturnRefCounter(const std::string& handle, uint64_t offset /* unused */) {
  if (!GcuIPCGlobalEntities::alive) {
    return;
  }
  std::lock_guard<std::mutex> lock(gcu_ipc_global_entities.ref_counters_mutex_);
  auto& map = gcu_ipc_global_entities.ref_counters_files_;
  auto it = map.find(handle);
  if (it != map.end()) {
    it->second->return_offset(offset);
    if (it->second->offsets_in_use() == 0 && !it->second->have_offsets()) {
      map.erase(handle);
    }
  }
}

}  // namespace

GcuIPCSentData::GcuIPCSentData(std::string handle, uint64_t offset,
                               uint64_t* counter_ptr, at::Device device)
    : handle_(std::move(handle)),
      offset_(offset),
      counter_ptr_(counter_ptr),
      original_ptr_(),
      device_(device) {
  // GCU have the unofficial limit on the number of recorded blocking
  // interprocess events, to prevent using of all events, we are switching to
  // StreamSync before limit reached.
  //
  //  ```python
  //  import torch
  //  a = [ torch.gcu.Event(
  //      enable_timing=False, blocking=True, interprocess=True) for i in
  //      range(30000) ]
  //  [i.record() for i in a]
  //  ```
  //

  // Workaround: GCU dot NOT support
  // if (gcu_ipc_global_entities.sync_events_used_.load() <
  //     GCU_IPC_MAXIMUM_EVENTS_TO_USE) {
  //   // TODO: More efficient would be to create event inside of main thread
  //   (at
  //   // the moment of the queue.put). The reason this is more efficient is
  //   // because the main thread may have queued extra work on the stream,
  //   which
  //   // this event will consequently wait for (uselessly).
  //   gcu_ipc_global_entities.sync_events_used_++;
  //   C10_GCU_CHECK(cudaEventCreateWithFlags(
  //       &event_,
  //       topsEventDisableTiming | topsEventInterprocess |
  //           topsEventBlockingSync));
  //   C10_GCU_CHECK(topsEventRecord(
  //       event_, torch_gcu::getCurrentGCUStream(device.index())));
  //   event_sync_required_ = true;
  // } else {
  //   auto stream = torch_gcu::getCurrentGCUStream(device.index());
  //   torch_gcu::stream_synchronize(stream);
  //   event_ = nullptr;
  //   event_sync_required_ = false;
  // }
  auto stream = torch_gcu::getCurrentGCUStream(device.index());
  torch_gcu::stream_synchronize(stream);
  event_sync_required_ = false;
}

GcuIPCSentData::~GcuIPCSentData() {
  ReturnRefCounter(handle_, offset_);
  try {
    if (event_sync_required_) {
      torch_gcu::GCUGuard device_guard(device_.index());
      C10_GCU_CHECK(topsEventDestroy(event_));
      if (!GcuIPCGlobalEntities::alive) {
        return;
      }
      gcu_ipc_global_entities.sync_events_used_--;
    }
  } catch (...) { /* No throw */
  }
}

uint64_t GcuIPCSentData::counter_value() { return *counter_ptr_; }

at::DataPtr GetNewRefCountedSentData(void* data, at::Device device) {
  {
    std::lock_guard<std::mutex> lock(
        gcu_ipc_global_entities.ref_counters_mutex_);
    if (!gcu_ipc_global_entities.next_available_ref_counters_file_) {
      std::string ref_counter_handle = at::NewProcessWideShmHandle();

      int flags =
          at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
      at::DataPtr sptr = at::RefcountedMapAllocator::makeDataPtr(
          ref_counter_handle.c_str(), flags,
          sizeof(int64_t) * GCU_IPC_REF_COUNTER_FILE_SIZE, nullptr);
      auto rc = std::make_shared<GcuIPCRefCountersFile>(
          ref_counter_handle, GCU_IPC_REF_COUNTER_FILE_SIZE, std::move(sptr));
      gcu_ipc_global_entities.ref_counters_files_[ref_counter_handle] = rc;
      gcu_ipc_global_entities.next_available_ref_counters_file_ = rc;
    }
  }
  gcu_ipc_global_entities.next_available_ref_counters_file_->set_counter(1);
  auto sent_data = new GcuIPCSentData(
      gcu_ipc_global_entities.next_available_ref_counters_file_->handle(),
      gcu_ipc_global_entities.next_available_ref_counters_file_->get_offset(),
      gcu_ipc_global_entities.next_available_ref_counters_file_->counter_ptr(),
      device);

  gcu_ipc_global_entities.next_available_ref_counters_file_->rotate_offset();
  if (!gcu_ipc_global_entities.next_available_ref_counters_file_
           ->have_offsets()) {
    gcu_ipc_global_entities.next_available_ref_counters_file_.reset();
  }
  return at::DataPtr(data, sent_data, GcuIPCSentDataDelete, device);
}

bool GcuIPCCollect() {
  if (!GcuIPCGlobalEntities::alive) {
    return true;
  }
  bool freed_memory = gcu_ipc_global_entities.GcuIPCSentDataLimbo_.collect();
  if (gcu_ipc_global_entities.GcuIPCSentDataLimbo_.size() == 0) {
    gcu_ipc_global_entities.safe_clean_current_file();
  }
  return freed_memory;
}

}  // namespace torch_gcu

namespace torch_gcu {
namespace {
REGISTER_FREE_MEMORY_CALLBACK("gcu_ipc_collect", GcuIPCCollectCallback);
}
}  // namespace torch_gcu
