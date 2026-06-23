/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */
#pragma once

#include <ATen/Tensor.h>
#include <ATen/core/Generator.h>
#include <c10/core/Device.h>

#include <unordered_map>

#include "gcu/gcu_graphs_c10_utils.h"
#include "gcu/gcu_macros.h"
#include "gcu/gcu_stream.h"

namespace torch_gcu {

struct GCUGeneratorImpl;

// Standalone way to get a unique mempool id usable as a pool=... argument
// to GCUGraph::capture_begin
MempoolId_t graph_pool_handle();

struct TORCH_GCU_API GCUGraph {
  GCUGraph();
  ~GCUGraph();

  static void inc_pending_event_queries();
  static void dec_pending_event_queries();
  static int num_pending_event_queries();
  void capture_begin(
      MempoolId_t pool = {0, 0},
      topsStreamCaptureMode capture_mode = topsStreamCaptureModeGlobal);
  void capture_end();
  void replay();
  void reset();
  MempoolId_t pool();
  void enable_debug_mode();
  void debug_dump(const std::string& debug_path);

  // Registers a non-default generator with this graph so its RNG state is
  // managed correctly across capture and replay (analogous to CUDA's
  // CUDAGraph::register_generator_state).
  void register_generator_state(const at::Generator& generator);

 protected:
  // Per-generator bookkeeping for non-default generators registered via
  // register_generator_state().
  struct ExtraGenData {
    at::Tensor seed_extragraph;
    at::Tensor offset_extragraph;
    uint64_t wholegraph_increment{0};
    void* host_seed{nullptr};
    void* host_offset{nullptr};
  };
  // Keyed by raw pointer; ownership of the generator lifetime is with the user.
  std::unordered_map<GCUGeneratorImpl*, ExtraGenData> extra_generators_;
  topsGraph_t graph_ = NULL;
  topsGraphExec_t graph_exec_ = NULL;

  static std::atomic<int> pending_event_queries;

  // internal states so reset() can do its best cleaning up
  // Set to true in capture_end if topsStreamEndCapture succeeded
  // Set back to false soon after, when graph_ is consumed by
  // topsGraphInstantiate to create graph_exec_, then graph_ is deleted
  bool has_graph_ = false;
  // Set to true in capture_end if topsGraphInstantiate succeeded
  bool has_graph_exec_ = false;

  // uuid of this instance's current capture, retrieved from Gcu
  CaptureId_t id_;

  // the ID assigned by gcu during graph capture,
  // used to identify when a stream is participating in capture
  CaptureId_t capture_id_ = -1;

  // uuid used to request a particular private mempool from
  // GCUCachingAllocator. By default, this will be set to {id_, 0}.
  //
  // If capture_begin is called with "pool=other_graph.pool()", this graph's
  // mempool_id_ will be set to the other graph's mempool_id_, and therefore
  // share a mempool with the other graph.
  //
  // If capture_begin is called with "pool=handle" where "handle" came from
  // graph_pool_handle(), it will share a mempool with any other captures that
  // used "pool=handle".
  //
  // Sharing a mempool across graphs saves memory, and it's safe if you
  // know you'll replay those graphs in the same order you captured them.
  MempoolId_t mempool_id_;

  // Stream on which capture began
  torch_gcu::GCUStream capture_stream_;

  // Default generator on device where capture began
  torch_gcu::GCUGeneratorImpl* capture_gen_;

  // Device where capture occurred. Right now, for simplicity, we require all
  // ops in a capture to run on the same device, but this is a limitation of
  // GCUGraph, not GCU itself.  We can straightforwardly modify GCUGraph to
  // support multi-device captures if needed.
  int capture_dev_;

  // RNG state trackers
  at::Tensor seed_extragraph_;
  at::Tensor offset_extragraph_;
  uint64_t wholegraph_increment_;

 private:
  // Use hostMalloc to avoid host pointer free before async copy is done.
  void* host_current_seed = nullptr;
  void* host_offset_val = nullptr;

 public:
  bool has_py_host_functions_ = false;
};

TORCH_GCU_API extern GCUGraph* current_gcu_graph_;
}  // namespace torch_gcu
