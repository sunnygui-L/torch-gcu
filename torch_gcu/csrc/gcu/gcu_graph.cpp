#include "gcu/gcu_graph.h"

#include <ATen/Functions.h>

#include <cstdint>

#include "c10/core/DeviceType.h"
#include "gcu/gcu_caching_allocator.h"
#include "gcu/gcu_exception.h"
#include "gcu/gcu_functions.h"
#include "gcu/gcu_generator_impl.h"
#include "gcu/gcu_graphs_c10_utils.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_macros.h"
#include "gcu/gcu_utils.h"

namespace torch_gcu {

GCUGraph* current_gcu_graph_ = nullptr;

static bool _gcu_graphs_debug = false;
constexpr int kSynchronizeBusyWaitMillis = 10;

TORCH_GCU_API MempoolId_t graph_pool_handle() {
  // uuid count starts at 1. 0 is reserved to mean "wasn't set by
  // graph_pool_handle".
  static std::atomic<CaptureId_t> uid{1};
  // Sets just the second value, to distinguish it from MempoolId_ts created
  // from topsStreamGetCaptureInfo id_s in capture_begin.
  return {0, uid++};
}

// Get the expected id of a capture sequence so that we can call
// beginAllocateStreamToPool before starting a graph capture
CaptureId_t capture_sequence_id() {
  // id starts at 1:
  // Ensures uuid count starts at 1. 0 is reserved to mean "not set by
  // topsStreamGetCaptureInfo". (But how do we know GetCaptureInfo never sets
  // id_ to 0? Because that's the current behavior, and I asked gcu devs to
  // keep it that way, and they agreed.)
  static std::atomic<CaptureId_t> uuid{1};
  return uuid++;
}

void GCUGraph::register_generator_state(const at::Generator& generator) {
  auto* gen = at::check_generator<GCUGeneratorImpl>(generator);
  gen->register_graph(this);

  ExtraGenData data;
  uint64_t size = sizeof(uint64_t);
  auto options = at::TensorOptions().device(at::kPrivateUse1).dtype(at::kInt);
  // GCU workaround: use {2} x int32 to represent a single int64 value.
  data.seed_extragraph = at::empty({2}, options);
  data.offset_extragraph = at::empty({2}, options);
  AT_GCU_CHECK(topsHostMalloc(&data.host_seed, c10::llvm::PowerOf2Ceil(size),
                              topsHostMallocDefault));
  AT_GCU_CHECK(topsHostMalloc(&data.host_offset, c10::llvm::PowerOf2Ceil(size),
                              topsHostMallocDefault));
  extra_generators_[gen] = std::move(data);
}

/**
 * Note [GCU Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native GCU ops (like RNG ops
 * with CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [GCU Graph-safe RNG states] in GCUGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write
 * capture bindings naively in an extension, they likely won't interact with the
 * native ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with GCU graph capture] in gcu_caching_allocator.cpp
 * describes memory management for captures.
 */

std::atomic<int> GCUGraph::pending_event_queries = 0;

// Track any outstanding event queries that could happen e.g., in a ECCL
// watchdog so that they can be resolved before the capture begins. Note that
// event queries are not allowed during a graph capture in the default capture
// mode.
void GCUGraph::inc_pending_event_queries() { pending_event_queries++; }

void GCUGraph::dec_pending_event_queries() {
  TORCH_INTERNAL_ASSERT(pending_event_queries > 0,
                        "Attempted to decrement the number of outstanding "
                        "events to be queried, but it was <= 0.");
  pending_event_queries--;
}

int GCUGraph::num_pending_event_queries() { return pending_event_queries; }

GCUGraph::GCUGraph()
    // GCUStreams may not be default-constructed.
    : capture_stream_(torch_gcu::getCurrentGCUStream()) {
  uint64_t size = sizeof(uint64_t);
  AT_GCU_CHECK(topsHostMalloc(&host_current_seed, c10::llvm::PowerOf2Ceil(size),
                              topsHostMallocDefault));
  AT_GCU_CHECK(topsHostMalloc(&host_offset_val, c10::llvm::PowerOf2Ceil(size),
                              topsHostMallocDefault));
}

void GCUGraph::capture_begin(MempoolId_t pool /*=0*/,
                             topsStreamCaptureMode capture_mode) {
  TORCH_CHECK(!has_graph_exec_,
              "This GCUGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  // For now, a GCUGraph instance only accommodates the default generator on
  // the device that's current when capture begins. If any op in the captured
  // region uses a non-default generator, or a generator on another device, the
  // offending generator will throw an error. These restrictions simplify
  // GCUGraph, but could be relaxed in the future: in principle, the underlying
  // Gcu calls do permit cross-device ops to be captured.
  auto* gen = at::get_generator_or_default<GCUGeneratorImpl>(
      c10::nullopt, torch_gcu::getDefaultGCUGenerator());

  // Workaround here.
  // Since GCU do NOT support 64-bit, we use 32-bit tensor in shape {2} to
  // simulate 64-bit tensor in shape {1}
  // auto options =
  // at::TensorOptions().device(at::kPrivateUse1).dtype(at::kLong);
  // seed_extragraph_ = at::empty({1}, options);
  // offset_extragraph_ = at::empty({1}, options);
  // seed_extragraph_.fill_(int64_t(gen->current_seed()));
  // gen->capture_prologue(seed_extragraph_.data_ptr<int64_t>(),
  //                       offset_extragraph_.mutable_data_ptr<int64_t>());
  auto options = at::TensorOptions().device(at::kPrivateUse1).dtype(at::kInt);
  seed_extragraph_ = at::empty({2}, options);
  offset_extragraph_ = at::empty({2}, options);
  int64_t current_seed = gen->current_seed();
  AT_GCU_CHECK(topsMemcpy(gcu_data_ptr(seed_extragraph_), &current_seed,
                          sizeof(int64_t), topsMemcpyHostToDevice));
  int64_t* casted_seed_extragraph_ptr =
      reinterpret_cast<int64_t*>(seed_extragraph_.data_ptr());
  int64_t* casted_offset_extragraph_ =
      reinterpret_cast<int64_t*>(offset_extragraph_.mutable_data_ptr());
  gen->capture_prologue(casted_seed_extragraph_ptr, casted_offset_extragraph_);

  // Initialize capture state for each extra registered generator.
  for (auto& [extra_gen, data] : extra_generators_) {
    int64_t extra_seed = extra_gen->current_seed();
    AT_GCU_CHECK(topsMemcpy(gcu_data_ptr(data.seed_extragraph), &extra_seed,
                            sizeof(int64_t), topsMemcpyHostToDevice));
    int64_t* extra_seed_ptr =
        reinterpret_cast<int64_t*>(data.seed_extragraph.data_ptr());
    int64_t* extra_offset_ptr =
        reinterpret_cast<int64_t*>(data.offset_extragraph.mutable_data_ptr());
    extra_gen->capture_prologue(extra_seed_ptr, extra_offset_ptr);
  }

  auto stream = torch_gcu::getCurrentGCUStream();

  TORCH_CHECK(stream != torch_gcu::getDefaultGCUStream(),
              "GCU graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_gen_ = gen;
  capture_dev_ = torch_gcu::current_device();

  id_ = capture_sequence_id();

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be
    // nonzero. If pool was created by graph_pool_handle, second should be
    // nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Use our own id_ as our
    // mempool_id_. Sets just the first value, to distinguish it from
    // MempoolId_ts created by graph_pool_handle().
    mempool_id_ = {id_, 0};
  }

  // Addendum: beginAllocateToPool is now called before
  // topsStreamBeginCapture to prevent an autograd thread's free() call
  // triggering an invalid topsEventRecord in the caching allocator due to the
  // capture status being updated _after_ a capture had already started.
  torch_gcu::GCUCachingAllocator::beginAllocateToPool(
      capture_dev_, mempool_id_, [this](topsStream_t stream) {
        topsStreamCaptureStatus status;
        CaptureId_t stream_capture_id;
        AT_GCU_CHECK(
            topsStreamGetCaptureInfo(stream, &status, &stream_capture_id));
        return status ==
                   topsStreamCaptureStatus::topsStreamCaptureStatusActive &&
               stream_capture_id == capture_id_;
      });

  // At this point, any ECCL watchdogs should be aware that we are in capture
  // mode
  // and therefore should not enqueue any additional work that could be
  // event-queried. We still must wait on any existing work that has not been
  // cleaned up.
  while (num_pending_event_queries()) {
    TORCH_WARN_ONCE(
        "Waiting for pending ECCL work to finish before starting graph "
        "capture.");
    std::this_thread::sleep_for(
        std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
  }

  current_gcu_graph_ = this;
  // topsStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe GCU API calls during capture.
  AT_GCU_CHECK(topsStreamBeginCapture(capture_stream_, capture_mode));

  topsStreamCaptureStatus status;
  AT_GCU_CHECK(topsStreamGetCaptureInfo(stream, &status, &capture_id_));
  TORCH_INTERNAL_ASSERT(status ==
                        topsStreamCaptureStatus::topsStreamCaptureStatusActive);

  TORCH_INTERNAL_ASSERT(id_ > 0);
}

void GCUGraph::capture_end() {
  auto stream = torch_gcu::getCurrentGCUStream();

  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

  AT_GCU_CHECK(topsStreamEndCapture(capture_stream_, &graph_));

  torch_gcu::GCUCachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  TORCH_CHECK(graph_ != NULL, "Invalid capture.");
  has_graph_ = true;

  // Workaround(GCU): Memory allocated by topsMallocAsync (topsruntime memory
  // pool) can NOT be auto-freed after catputring
  // AT_GCU_CHECK(topsGraphInstantiateWithFlags(
  //     &graph_exec_, graph_, topsGraphInstantiateFlagAutoFreeOnLaunch));
  AT_GCU_CHECK(topsGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));

  has_graph_exec_ = true;

  auto* gen = at::get_generator_or_default<GCUGeneratorImpl>(
      c10::nullopt, torch_gcu::getDefaultGCUGenerator());
  TORCH_CHECK(gen == capture_gen_,
              "Default GCU RNG generator on current device at capture end "
              "is different from default generator on current device "
              "when capture began");
  wholegraph_increment_ = gen->capture_epilogue();

  // Finalize capture state for each extra registered generator.
  for (auto& [extra_gen, data] : extra_generators_) {
    data.wholegraph_increment = extra_gen->capture_epilogue();
  }

  size_t numGCUGraphNodes = 0;
  AT_GCU_CHECK(topsGraphGetNodes(graph_, NULL, &numGCUGraphNodes));
  if (numGCUGraphNodes == 0) {
    TORCH_WARN("The GCU Graph is empty. This usually means that the graph was ",
               "attempted to be captured on wrong device or stream.");
  }

  // check if debug path is set
  if (!_gcu_graphs_debug) {
    // Now that we've instantiated graph_ into graph_exec_,
    // we don't need graph_ anymore.
    AT_GCU_CHECK(topsGraphDestroy(graph_));
    has_graph_ = false;
  } else {
    TORCH_WARN(
        "DEBUG: TORCH_GCUGRAPHS_DEBUG_PATH detected. graph_ will not be freed "
        "until debug_dump is called.");
  }

  current_gcu_graph_ = nullptr;
}

void GCUGraph::replay() {
  TORCH_CHECK(
      has_graph_exec_,
      "Called GCUGraph::replay without a preceding successful capture.");

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  // Just like any RNG consumer kernel!
  auto* gen = at::get_generator_or_default<GCUGeneratorImpl>(
      c10::nullopt, torch_gcu::getDefaultGCUGenerator());
  PhiloxGcuState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_gcu_state(wholegraph_increment_);
  }

  // Workaround here.
  // Since GCU do NOT support 64-bit, we use 32-bit tensor in shape {2} to
  // simulate 64-bit tensor in shape {1}
  // seed_extragraph_.fill_(int64_t(gen->current_seed()));
  // offset_extragraph_.fill_(int64_t(rng_engine_inputs.offset_.val));
  auto stream = getCurrentGCUStream();
  int64_t current_seed = gen->current_seed();
  int64_t offset_val = rng_engine_inputs.offset_.val;
  uint64_t size = sizeof(uint64_t);

  memcpy(host_current_seed, &current_seed, size);
  memcpy(host_offset_val, &offset_val, size);
  AT_GCU_CHECK(topsMemcpyAsync(gcu_data_ptr(seed_extragraph_),
                               host_current_seed, size, topsMemcpyHostToDevice,
                               stream));
  AT_GCU_CHECK(topsMemcpyAsync(gcu_data_ptr(offset_extragraph_),
                               host_offset_val, size, topsMemcpyHostToDevice,
                               stream));

  // Update seed/offset tensors for each extra registered generator.
  for (auto& [extra_gen, data] : extra_generators_) {
    PhiloxGcuState extra_rng_inputs;
    {
      std::lock_guard<std::mutex> lock(extra_gen->mutex_);
      extra_rng_inputs = extra_gen->philox_gcu_state(data.wholegraph_increment);
    }
    int64_t extra_seed = extra_gen->current_seed();
    int64_t extra_offset = extra_rng_inputs.offset_.val;
    memcpy(data.host_seed, &extra_seed, size);
    memcpy(data.host_offset, &extra_offset, size);
    AT_GCU_CHECK(topsMemcpyAsync(gcu_data_ptr(data.seed_extragraph),
                                 data.host_seed, size, topsMemcpyHostToDevice,
                                 stream));
    AT_GCU_CHECK(topsMemcpyAsync(gcu_data_ptr(data.offset_extragraph),
                                 data.host_offset, size, topsMemcpyHostToDevice,
                                 stream));
  }

  // graph_exec_ may be replayed in any stream.
  AT_GCU_CHECK(topsGraphLaunch(graph_exec_, stream));

  if (this->has_py_host_functions_) {
    // 关键：如果 graph 包含 py_host_functions，必须在这里同步，在 GIL
    // 被重新获取之前 如果等到 Python 层调用 synchronize()，GIL
    // 已经被重新获取，仍然会死锁
    AT_GCU_CHECK(topsStreamSynchronize(stream));
  }

  int version;
  AT_GCU_CHECK(topsDriverGetVersion(&version));
}

void GCUGraph::enable_debug_mode() { _gcu_graphs_debug = true; }

void GCUGraph::debug_dump(const std::string& debug_path) {
  if (_gcu_graphs_debug) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling topsGraphDebugDotPrint() with ", debug_path);
      C10_GCU_CHECK_WARN(topsGraphDebugDotPrint(
          graph_, debug_path.c_str(), 1 << 10));  // most verbose output
      AT_GCU_CHECK(topsGraphDestroy(graph_));
    }
  } else {
    TORCH_WARN(
        "GCU Graphs debug not enabled, set with "
        "torch_gcu._C._gcu_enable_graphs_debug_mode");
  }
}

void GCUGraph::reset() {
  // I'd prefer these checks throw exceptions, not print warnings,
  // but the destructor calls reset(), and at least one CI build
  // refuses to compile with a throwing destructor.
  //
  // Instead of calling reset() in the destructor to clean up, I could
  // call reset() in the __del__ method of a thin Python wrapper,
  // in which case reset would be allowed to throw exceptions.
  // But Stackoverflow does not like user-defined __del__.
  // __del__ prevents Graph instances from EVER being garbage collected
  // if they participate in a reference cycle.
  // And exceptions thrown in __del__ only print a warning anyway.
  //
  // Calling reset() in the C++ destructor, with warnings instead of exceptions
  // if calls fail, is the compromise we chose.
  //
  // If capture_begin, the capture, or capture_end failed at some point, this
  // GCUGraph, the generator, and the allocator could end up in all kinds of
  // weird states depending where failure occurred. If the user catches the
  // failure exception in a script, or is running in REPL or (god forbid) a
  // Jupyter notebook, I don't see an easy way for reset() to gracefully fix all
  // such possible error states.
  if (has_graph_ || has_graph_exec_) {
    // notifyCaptureDestroy may throw. How should we handle this?
    torch_gcu::GCUCachingAllocator::releasePool(capture_dev_, mempool_id_);
  }
  if (has_graph_) {
    C10_GCU_CHECK_WARN(topsGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    C10_GCU_CHECK_WARN(topsGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
}

// Returns an id another graph's capture_begin can use to share the same memory
// pool as this graph.
MempoolId_t GCUGraph::pool() {
  TORCH_CHECK(
      has_graph_exec_,
      "Called GCUGraph::pool() without a preceding successful capture.");
  return mempool_id_;
}

GCUGraph::~GCUGraph() {
  AT_GCU_CHECK(topsHostFree(host_current_seed));
  AT_GCU_CHECK(topsHostFree(host_offset_val));
  for (auto& [gen, data] : extra_generators_) {
    gen->unregister_graph(this);
    if (data.host_seed) {
      C10_GCU_CHECK_WARN(topsHostFree(data.host_seed));
    }
    if (data.host_offset) {
      C10_GCU_CHECK_WARN(topsHostFree(data.host_offset));
    }
  }
  reset();
}

}  // namespace torch_gcu
