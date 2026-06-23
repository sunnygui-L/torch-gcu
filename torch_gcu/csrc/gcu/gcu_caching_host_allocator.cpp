#include "gcu/gcu_caching_host_allocator.h"

#include <ATen/DeviceGuard.h>

#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gcu/gcu_event.h"
#include "gcu/gcu_functions.h"
#include "gcu/gcu_graphs_c10_utils.h"

namespace torch_gcu {

namespace {

struct BlockSize {
  size_t size_{0};
  void* ptr_{nullptr};
};

struct Block {
  size_t size_{0};
  void* ptr_{nullptr};

  std::mutex mutex_;
  bool allocated_{false};
  size_t event_count_{0};
  std::unordered_set<GCUStream> streams_;
};

// Note as gcu caching allocator.
class EventPool {
 public:
  using Event = std::unique_ptr<GCUEvent, std::function<void(GCUEvent*)>>;
  EventPool() : pools_(device_count()) {}

  Event get(at::DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<at::DeviceIndex>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](GCUEvent* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<GCUEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // Allocate a new event that will be returned to the pool on destruction.
    return Event(std::make_unique<GCUEvent>(topsEventDisableTiming).release(),
                 destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<GCUEvent>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

// Used for heterogenous lookup support in the free list.
struct BlockComparator {
  using is_transparent = void;
  bool operator()(const Block* a, const Block* b) const {
    if (a->size_ != b->size_) {
      return a->size_ < b->size_;
    }
    return (uintptr_t)a->ptr_ < (uintptr_t)b->ptr_;
  }

  bool operator()(const Block* a, BlockSize b) const {
    if (a->size_ != b.size_) {
      return a->size_ < b.size_;
    }
    return (uintptr_t)a->ptr_ < (uintptr_t)b.ptr_;
  }

  bool operator()(BlockSize a, const Block* b) const {
    if (a.size_ != b->size_) {
      return a.size_ < b->size_;
    }
    return (uintptr_t)a.ptr_ < (uintptr_t)b->ptr_;
  }
};

/**
 * Note [TOPSHostAllocator design]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * We have three key data structures - the free list which stores blocks that
 * are not currently used, the block list which stores all blocks that have been
 * allocated, and the event queue which stores TOPS events and their
 * corresponding blocks.
 *
 * Each of these are protected by a separate mutex. The key design principles
 * are to 1) only hold each mutex for the minimal amount of time possible, 2)
 * never do any possible expensive operations (such as TOPS runtime API calls)
 * while holding the lock.
 *
 * There are three public methods: allocate, free, and record_event. In the
 * allocate path, we first check to see if we can service our request from this
 * free list, and otherwise we create a new block with topsHostAlloc. In the
 * free path, we insert events (if required) into the event queue, and if
 * possible insert our block back into the free list. In allocate, we first
 * eagerly query events until we find one that is not ready, and insert the
 * corresponding block onto the free list if all the events recorded for a
 * block are ready. In the record_event path, we simply insert the given
 * stream into the set of streams tracked by the specified block. This set of
 * streams is then consumed in the free path.
 *
 * Some of the invariants here are less strict than they could be - for example,
 * we do not enforce that free(Block* block) => block->event_count == 0. This is
 * for compatibility reasons, and we can explore enforcing these in subsequent
 * versions.
 */
class TOPSHostAllocator {
 public:
  std::pair<void*, void*> allocate(size_t size) {
    if (size == 0) {
      return {nullptr, nullptr};
    }

    process_events();

    // First, try to allocate from the free list.
    {
      std::lock_guard<std::mutex> g(free_list_mutex_);
      auto it = free_list_.lower_bound(BlockSize{size, nullptr});
      if (it != free_list_.end()) {
        auto block = *it;
        block->allocated_ = true;
        free_list_.erase(it);
        return {block->ptr_, reinterpret_cast<void*>(block)};
      }
    }
    // Then, create a new block.
    // Pinned memory pointers allocated by any device can be directly used by
    // any other device, regardless of the current device at the time of
    // allocation, since we assume unified addressing. So we grab any existing
    // primary context, if available. See pytorch/pytorch#21081.
    at::OptionalDeviceGuard device_guard(
        (at::Device(at::DeviceType::PrivateUse1)));
    auto primary_ctx_device_index = getDeviceIndexWithPrimaryContext();
    if (primary_ctx_device_index.has_value()) {
      device_guard.reset_device(
          at::Device(at::DeviceType::PrivateUse1, *primary_ctx_device_index));
    }

    // Round up the allocation to the nearest power of two to improve reuse.
    void* ptr = nullptr;
    if (torch_gcu::currentStreamCaptureStatusMayInitCtx() ==
        torch_gcu::CaptureStatus::None) {
      C10_GCU_CHECK(topsHostMalloc(&ptr, c10::llvm::PowerOf2Ceil(size),
                                   topsHostMallocDefault));
    } else {
      // It's ok to capture topsHostMalloc, as long as we never topsHostFree
      // those addresses before replay. Capturing topsHostMalloc behaves nicely:
      // it gives the graph new VA, but is ignored (won't leakily allocate new
      // memory) in replays.
      torch_gcu::GCUStreamCaptureModeGuard g{topsStreamCaptureModeRelaxed};
      C10_GCU_CHECK(topsHostMalloc(&ptr, c10::llvm::PowerOf2Ceil(size),
                                   topsHostMallocDefault));
    }
    // C10_GCU_CHECK(topsHostMalloc(&ptr, c10::llvm::PowerOf2Ceil(size),
    //                              topsHostMallocDefault));
    auto block = new Block();
    block->size_ = c10::llvm::PowerOf2Ceil(size);
    block->ptr_ = ptr;
    block->allocated_ = true;

    {
      std::lock_guard<std::mutex> g(blocks_mutex_);
      blocks_.insert(block);
      ptr_to_block_.insert({block->ptr_, block});
    }
    return {block->ptr_, reinterpret_cast<void*>(block)};
  }

  void free(void* ctx) {
    if (!ctx) {
      return;
    }

    // Note: we can assume that free is correctly paired with alloc,
    // and thus we do not need to look up the ctx in blocks_.
    auto* block = reinterpret_cast<Block*>(ctx);

    c10::optional<std::vector<EventPool::Event>> events;
    {
      std::lock_guard<std::mutex> g(block->mutex_);
      block->allocated_ = false;
      if (block->streams_.empty()) {
        TORCH_INTERNAL_ASSERT(block->event_count_ == 0);
      } else {
        events = std::vector<EventPool::Event>();
        events->reserve(block->streams_.size());
        for (auto stream : block->streams_) {
          auto event = event_pool_.get(stream.device_index());
          event->record(stream);
          events->push_back(std::move(event));
        }
        block->event_count_ += events->size();
        block->streams_.clear();
      }
    }

    if (!events) {
      std::lock_guard<std::mutex> g(free_list_mutex_);
      free_list_.insert(block);
    } else {
      std::lock_guard<std::mutex> g(gcu_events_mutex_);
      for (auto&& event : *events) {
        gcu_events_.emplace_front(std::move(event), block);
      }
    }
  }

  bool record_event(void* ptr, void* ctx, GCUStream stream) {
    auto* block = reinterpret_cast<Block*>(ctx);

    // Note: we need to check if the passed-in `ctx` is valid. This is because
    // `record_event` (via `CachingHostAllocator_recordEvent`) can be invoked on
    // an arbitrary tensor, and is not guaranteed to correspond to a pinned
    // memory allocation. Therefore, we need to check that `ctx` is valid before
    // proceeding.
    {
      std::lock_guard<std::mutex> g(blocks_mutex_);
      if (blocks_.find(block) != blocks_.end()) {
        // Now we know this object is safe to access.
        std::lock_guard<std::mutex> gb(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }
      auto it = ptr_to_block_.find(ptr);
      if (it != ptr_to_block_.end()) {
        block = it->second;
        std::lock_guard<std::mutex> g(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }
    }

    return false;
  }

  void empty_cache() {
    // Flush any available blocks into the free_list.
    process_events();

    // Release cached events from the event pool.
    event_pool_.empty_cache();

    // Remove all elements from the free list, remove them from the blocks
    // list, and free the associated pinned memory allocation. This requires
    // concurrently holding both the free list mutex and the blocks mutex, and
    // is the only function that concurrently holds multiple mutexes.
    std::lock(free_list_mutex_, blocks_mutex_);
    std::lock_guard<std::mutex> gf(free_list_mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> gb(blocks_mutex_, std::adopt_lock);

    std::vector<Block*> blocks_to_remove(free_list_.begin(), free_list_.end());
    free_list_.clear();
    for (auto* block : blocks_to_remove) {
      blocks_.erase(block);
      ptr_to_block_.erase(block->ptr_);
      C10_GCU_CHECK(topsHostFree(block->ptr_));
      delete block;
    }
  }

  void copy_data(void* dest, const void* src, std::size_t count) const {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for TOPSHostAllocator");
  }

 private:
  void process_events() {
    while (true) {
      // Avoid calling topsEventDestroy while holding a mutex, so move
      // intermediate events out of the lock into this object.
      c10::optional<std::pair<EventPool::Event, Block*>> processed;

      {
        std::lock_guard<std::mutex> g(gcu_events_mutex_);
        if (!gcu_events_.empty()) {
          processed = std::move(gcu_events_.back());
          gcu_events_.pop_back();
        }
      }

      if (!processed) {
        return;
      }

      // otherwise, query the event
      {
        // now, see if we can handle this element
        auto& event = processed->first;
        topsError_t err = topsEventQuery(*event);
        if (err == topsErrorNotReady) {
          (void)topsGetLastError();  // clear TOPS error
          // push the event onto the back of the queue if it's not
          // ready. TODO: do we need some debouncing logic to avoid allocating
          // threads repeatedly spinning on an event?
          {
            std::lock_guard<std::mutex> g(gcu_events_mutex_);
            gcu_events_.push_back(std::move(*processed));
          }
          return;
        } else if (err != topsSuccess) {
          C10_GCU_CHECK(err);
        }
      }

      // Process the events.
      TORCH_INTERNAL_ASSERT(processed);
      auto* block = processed->second;
      bool available = false;
      {
        std::lock_guard<std::mutex> g(block->mutex_);
        TORCH_INTERNAL_ASSERT(!block->allocated_)
        block->event_count_--;
        if (block->event_count_ == 0) {
          available = true;
        }
      }

      if (available) {
        std::lock_guard<std::mutex> g(free_list_mutex_);
        free_list_.insert(block);
      }
    }
  }

  EventPool event_pool_;

  alignas(64) std::mutex blocks_mutex_;
  std::unordered_set<Block*> blocks_;
  std::unordered_map<void*, Block*> ptr_to_block_;
  // Note: sharding this mutex seems to be profitable in heavily multi-threaded
  // scenarios.
  alignas(64) std::mutex free_list_mutex_;
  // Note: an alternative datastructure can yield significant wins here in
  // microbenchmarks.
  std::set<Block*, BlockComparator> free_list_;

  alignas(64) std::mutex gcu_events_mutex_;
  std::deque<std::pair<EventPool::Event, Block*>> gcu_events_;
};

}  // namespace

static TOPSHostAllocator& getTOPSHostAllocator() {
  // leak and don't worry about shutdown.
  static auto* r = new TOPSHostAllocator();
  return *r;
}

static void TOPSHostAllocatorDeleter(void* ctx) {
  getTOPSHostAllocator().free(ctx);
}

bool CachingHostAllocator_recordEvent(void* ptr, void* ctx, GCUStream stream) {
  return getTOPSHostAllocator().record_event(ptr, ctx, stream);
}

void CachingHostAllocator_emptyCache() { getTOPSHostAllocator().empty_cache(); }

struct TOPSHostAllocatorWrapper final : public at::Allocator {
  at::DataPtr allocate(size_t size) override {
    auto ptr_and_ctx = getTOPSHostAllocator().allocate(size);
    return {ptr_and_ctx.first, ptr_and_ctx.second, &TOPSHostAllocatorDeleter,
            at::DeviceType::CPU};
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    getTOPSHostAllocator().copy_data(dest, src, count);
  }
};

static TOPSHostAllocatorWrapper tops_host_allocator;

at::Allocator* getCachingHostAllocator() { return &tops_host_allocator; }

}  // namespace torch_gcu
