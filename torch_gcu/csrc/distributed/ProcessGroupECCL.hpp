#pragma once
#ifdef USE_C10D_ECCL
#include <cstdint>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "gcu/gcu_macros.h"
#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <ATen/Device.h>
#include <ATen/DynamicLibrary.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/core/Stream.h>
#include <c10/core/StreamGuard.h>
#include <torch/custom_class.h>
#include <torch_gcu/csrc/gcu/gcu_event.h>
#include <torch_gcu/csrc/gcu/gcu_stream.h>

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include "gcu/gcu_functions.h"
// #include <torch/csrc/distributed/c10d/intra_node_comm.hpp>
#include <torch_gcu/csrc/distributed/ECCLUtils.hpp>
#include <unordered_map>
namespace c10d_gcu {
// Control whether or not wait() is blocking or non-blocking.
static std::vector<std::string> TORCH_ECCL_BLOCKING_WAIT = {
    "TORCH_ECCL_BLOCKING_WAIT", "ECCL_BLOCKING_WAIT"};

// Control whether or not we perform Async Error Handling with ECCL.
static std::vector<std::string> TORCH_ECCL_ASYNC_ERROR_HANDLING = {
    "TORCH_ECCL_ASYNC_ERROR_HANDLING", "ECCL_ASYNC_ERROR_HANDLING"};

// Control whether dumping debug info on watchdog
// timeout is enabled. This variable must be set together with
// TORCH_ECCL_ENABLE_MONITORING=1 and TORCH_ECCL_TRACE_BUFFER_SIZE > 0.
static std::vector<std::string> TORCH_ECCL_DUMP_ON_TIMEOUT = {
    "TORCH_ECCL_DUMP_ON_TIMEOUT"};

// Control whether Desync Debug is enabled. This variable must be set
// together with TORCH_ECCL_ASYNC_ERROR_HANDLING.
static std::vector<std::string> TORCH_ECCL_DESYNC_DEBUG = {
    "TORCH_ECCL_DESYNC_DEBUG", "ECCL_DESYNC_DEBUG"};

// Enable recording start-events for all ProcessGroupECCL collectives, and
// compute accurate collective timing per-collective. (Note: end-events are
// recorded by default. Turn on this flag can increase chances of a watchdog
// hang due to performing a GCU event query which eventually calls
// topsEventElapsedTime() API.
static std::vector<std::string> TORCH_ECCL_ENABLE_TIMING = {
    "TORCH_ECCL_ENABLE_TIMING", "ECCL_ENABLE_TIMING"};

// Enable monitoring thread which aborts the process when the ProcessGroupECCL
// Watchdog thread gets stuck and no heartbeat is detected after
// TORCH_ECCL_HEARTBEAT_TIMEOUT_SEC. This can happen due to calling GCU/ECCL
// APIs that may hang. It is Useful to prevent jobs being stuck for a prolonged
// time than necessary tying up cluster resources.
static std::vector<std::string> TORCH_ECCL_ENABLE_MONITORING = {
    "TORCH_ECCL_ENABLE_MONITORING"};

// Control the watchdog heartbeat timeout period after which the monitoring
// thread will abort the process.
static std::vector<std::string> TORCH_ECCL_HEARTBEAT_TIMEOUT_SEC = {
    "TORCH_ECCL_HEARTBEAT_TIMEOUT_SEC"};

// The maximum number of events we store in the flight recorder's ring buffer.
// (One event could be the start or end of a collective, for example).
static std::vector<std::string> TORCH_ECCL_TRACE_BUFFER_SIZE = {
    "TORCH_ECCL_TRACE_BUFFER_SIZE"};

// Control how much extra time we will wait for dumping the debugging info
// before we exit and throws timeout exception.
static std::vector<std::string> TORCH_ECCL_WAIT_TIMEOUT_DUMP_MILSEC = {
    "TORCH_ECCL_WAIT_TIMEOUT_DUMP_MILSEC"};

// Control the interval inside the watchdog thread to check the coordinated
// signal from other ranks, e.g. to dump the debugging information.
static std::vector<std::string> TORCH_ECCL_COORD_CHECK_MILSEC = {
    "TORCH_ECCL_COORD_CHECK_MILSEC"};

// Whether to abort the communicators when users call destroy_process_group().
// If yes, communicators will be aborted when destroy_process_group is called,
// but not in destructor.
static std::vector<std::string> TORCH_ECCL_ABORT_IN_DESTROY_PG = {
    "TORCH_ECCL_ABORT_IN_DESTROY_PG"};

constexpr const char* ECCL_BACKEND_NAME = "eccl";

constexpr const char* TIMEOUT_DUMP = "timeout_dump";

constexpr const int kWorkStatusUpdatePeriodMs = 10 * 1000;  // 10 seconds

constexpr auto kProcessGroupECCLDefaultTimeout =
    std::chrono::milliseconds(10 * 60 * 1000);

// NoHandling: do not handle asynchronous ECCL errors
// TearDown: tear down process upon error, see `WorkECCL::handleException`
// CleanUpOnly: just clean up collectives and abort communicators without
// tearing down process SkipCleanUp: (this is a temporary option and can be
// removed in future) tear down process without cleaning up ECCL communicators.
// This should be used as a last resort in case `ecclCommAbort` itself is
// hanging
enum ErrorHandlingMode {
  NoHandling = 0,
  TearDown = 1,
  CleanUpOnly = 2,
  SkipCleanUp = 3
};

#define SHOULD_CLEAN_UP(a) (a != NoHandling && a != SkipCleanUp)

#define SHOULD_TEAR_DOWN(a) (a != NoHandling && a != CleanUpOnly)

#define PRINT_COLLECTIVE_HASH_SIGNATURE(phase, opType, numel, hashValue)      \
  LOG(WARNING) << logPrefix() << "Hash of " << phase << " to ECCL " << opType \
               << " with size " << numel << " is " << hashValue;

// If set, ProcessGroupECCL doesn't use recordStream calls to ensure
// caching allocator safety for tensors used on both user-facing and
// internal comm streams.
// Instead, it stashes live references to those tensors until after
// user-facing streams are synced with comm streams.
// See stashed_for_allocator_safety_ below.
static std::vector<std::string> TORCH_ECCL_AVOID_RECORD_STREAMS = {
    "TORCH_ECCL_AVOID_RECORD_STREAMS"};

// If set, ProcessGroupECCL registers postAlloc and preFree hooks to gcu cache
// allocator so that whenever a tensor is allocated or freed, ProcessGroupECCL
// can register/deregister the tensor on all available ECCL communicators.
static std::vector<std::string> TORCH_ECCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK =
    {"TORCH_ECCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK",
     "ECCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"};

#if defined(__linux__)
struct DumpPipe {
  DumpPipe(int rank) {
    std::string fileStem =
        c10d::getCvarString({"TORCH_ECCL_DEBUG_INFO_PIPE_FILE"}, "");
    if (fileStem.empty() ||
        c10d::getCvarInt({"TORCH_ECCL_TRACE_BUFFER_SIZE"}, 0) <= 0) {
      return;
    }
    TORCH_CHECK(!fileStem.empty(), "TORCH_ECCL_DEBUG_INFO_TEMP_FILE is empty");
    std::string filename = c10::str(fileStem, rank, ".pipe");
    TORCH_CHECK(unlink(filename.c_str()) != -1 || errno == ENOENT,
                "Error removing existing named pipe ", filename);
    TORCH_CHECK(mkfifo(filename.c_str(), 0666) != -1,
                "Error creating named pipe ", filename);
    fd_ = open(filename.c_str(), O_RDONLY | O_NONBLOCK);
    LOG(INFO) << "Pipe file " << filename
              << " has been opened, write to it to trigger ECCL Debug Dump.";
    TORCH_CHECK(fd_ != -1, "Error opening named pipe ", filename);
  }
  bool shouldDump() {
    if (fd_ == -1) {
      return false;
    }
    char buf[128];
    // non-blocking from O_NONBLOCK above.
    // Ignore EINTR because we already will poll this
    // again later.
    ssize_t bytesRead = read(fd_, &buf, 128);
    return bytesRead > 0;
  }
  ~DumpPipe() {
    if (fd_ != -1) {
      close(fd_);
    }
  }

 private:
  int fd_ = -1;
};
#else
struct DumpPipe {
  DumpPipe(int rank) {}
  bool shouldDump() { return false; }
};
#endif

// ProcessGroupECCL implements ECCL bindings for c10d.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group.  This is the only way that we
// can guarantee to match up the same calls among all processes.
//
// All ECCL functions provided by this class are asynchronous functions. More
// specifically, each ECCL call is scheduled on a separate GCU stream that is
// different from the current GCU stream. This is for the purpose of
// achieving potentially concurrency and better performance. As a result,
// it is the callers' responsibility to make sure that the GCU stream their
// code works on needs to wait for the ECCL operation from
// this class.
//
// This can be done by calling:
//
// either WorkECCL::wait() or WorkECCL::synchronize(), both achieves the same
// functionality and are synonyms.
//
// Also note that WorkECCL::finishedGCUExecution() is a helper function only
// provided by ProcessGroupECCL to check if the ECCL operation of WorkECCL has
// finished execution on the GCU (not just scheduled).
//
// Example on using the ECCL process group
//
//   ProcessGroupECCL pg(store, rank, size);
//   std::shared_ptr<WorkECCL> work = pg.allreduce(tensors);
//
//   // At this point, ECCL kernel has already by queued successfully
//   // Now, let current stream wait for the ECCL to finish, this function is
//   // async operation as well
//
//   work->wait()
//
//   // Now continue on other work in the current stream.
class TORCH_GCU_API ProcessGroupECCL : public c10d::Backend {
 public:
  class WorkECCL : public c10d::Work,
                   public std::enable_shared_from_this<WorkECCL> {
   public:
    friend struct c10d::WorkInfo;

    // Constructor takes a list of GCU devices
    WorkECCL(
        at::Device& device, int rank, c10d::OpType opType, uint64_t seq,
        const char* profilingTitle = nullptr,
        const c10::optional<std::vector<at::Tensor>>& inputs = c10::nullopt,
        bool desyncDebug = false, bool enableTiming = false,
        c10d::DebugLevel distDebugLevel = c10d::DebugLevel::Off);
    // Copy constructor doing partial copy without outputs_. Cleanup thread
    // monitors and removes finished works. However it will deadlock when
    // destructs outputs_ tensors who are view tensors in autograd graph.
    WorkECCL(const WorkECCL& w);

    ~WorkECCL() override;

    // Checks if the ECCL kernel has started to execute.
    bool isStarted();

    // Checks if request has completed. In this specific case of ECCL, it checks
    // if the ECCL operation has completed on the GCU in its own ECCL stream.
    // Non-blocking operation.
    bool isCompleted() override;

    bool isSuccess() const override;

    // Same as calling synchronize() for ECCL work.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    void abort() override;

    // Let current stream wait on the completing of the ECCL work
    // Throws on exceptions. Blocking operation, which will wait for work
    // completion.
    void synchronize() override;

    // Synchronize streams by blocking each on the ECCL stream
    void synchronizeStream();

    // Helper function to handle exception (throw if needed).
    void handleException(ErrorHandlingMode asyncErrorHandling);

    // Helper function that checks if the ECCL kernels have finished
    // execution on the GCUs
    bool finishedGCUExecution();

    // Get a Future object that will be marked as completed internally.
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    float getDuration() const override;

    uint64_t getSequencenumber() const override;

    const std::string& logPrefix() const;

    // Helper function that sets an exception_ptr on the WorkECCL object.
    void setException(std::exception_ptr exception_ptr);

    // Helper function that returns True if the WorkECCL object has timed out
    // and False otherwise.
    // In case of timeout, set exception on the WorkECCL object.
    bool checkTimeout(
        c10::optional<std::chrono::milliseconds> timeout = c10::nullopt);

    std::vector<at::Tensor> result() override;

   protected:
    // The process group unique id
    std::string pgUID_;

    // The process group description
    std::string pgDesc_;
    // The cached list of GCU devices to operate on
    at::Device device_;

    // The start GCU event of ECCL operator tracking this work item. These
    // start GCU events are needed by desync debugging if enabled.
    std::shared_ptr<torch_gcu::GCUEvent> ecclStartEvent_;

    // The end GCU event of ECCL operator tracking this work item.
    std::shared_ptr<torch_gcu::GCUEvent> ecclEndEvent_;

    // Note(torch_gcu): reuse GCUEvent will increase record time consumption, so
    // each time a new GCUEvent is created The GCU events used to sync ECCL
    // streams and we move ecclEvent_ to WorkECCL from ProcessGroupECCL
    torch_gcu::GCUEvent ecclEvent_;

    // The ECCL communicator used for this work item.
    std::shared_ptr<ECCLComm> ecclComm_;

    // Tensors used for barrier op
    at::Tensor barrierTensor_;

    // Clone of blockingWait_ from ProcessGroupECCL.
    bool blockingWait_ = false;

    // Clone of avoidRecordStreams_ from ProcessGroupECCL.
    bool avoidRecordStreams_ = false;

    // Clone of opTimeout_ from ProcessGroupECCL.
    std::chrono::milliseconds opTimeout_;

    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

    // Record the collective sequential number.
    uint64_t seq_;

    // Indicates if the eccl start event has been updated to the store trace.
    // This will be used by desync debug.
    bool startTraceUpdated_{false};
    const char* profiling_title_;

    // Record collective sizes for debug. We only record the size on the first
    // device as multi-device per process is deprecated
    size_t numelIn_ = -1;
    size_t numelOut_ = -1;

    // Wrapper method for the static checkForECCLErrors which can be overridden
    // for tests.
    virtual std::exception_ptr checkForECCLErrors();

    friend std::ostream& operator<<(std::ostream& output,
                                    const WorkECCL& workECCL);

   private:
    // Helper function for synchronize
    void synchronizeInternal(std::chrono::milliseconds timeout);

    // Checks for ECCL errors and sets an appropriate exception_ptr.
    void checkAndSetException();

    // Just checks whether GCU execution has started, without modifying
    // exception_ptr.
    bool startedGCUExecutionInternal() const;

    // Just checks whether GCU execution has completed, without modifying
    // exception_ptr.
    bool finishedGCUExecutionInternal() const;

    // Reference to the store so that we can write aborted communicators
    // to the store.
    c10::intrusive_ptr<c10d::Store> store_;

    // Store a reference to ECCL collective's outputs, used by result and to
    // give a more descriptive message when representing the c10d::Work as a
    // string.
    std::shared_ptr<std::vector<at::Tensor>> outputs_;

    // TORCH_ECCL_AVOID_RECORD_STREAMS implementation helper.
    // Stores references to participating non-output tensors (ie inputs,
    // flattened intermediates).
    // We'll clear this list in synchronizeStream, just after user-facing
    // stream(s) are synced with the eccl work stream(s).
    // By keeping these refs (as well as outputs_) alive until after the
    // collective's work rejoins the user-facing streams, we achieve
    // caching allocator safety without any recordStream calls.
    // For in-place collectives, some refs stashed here may alias outputs_,
    // but that doesn't do any harm.
    std::shared_ptr<std::vector<at::Tensor>> stashed_for_allocator_safety_;

    // The future returned by getFuture.
    c10::intrusive_ptr<at::ivalue::Future> future_;

    bool timingEnabled_;
    // unique id used to tell the trace buffer that this
    // work has completed
    c10::optional<uint64_t> trace_id_;
    c10d::DebugLevel distDebugLevel_;
    friend class ProcessGroupECCL;
  };

  struct Options : c10d::Backend::Options {
    // NOTE: timeout in ProcessGroupECCL::Options denote the timeout for
    // operations. This is only used when blockingWait_ is enabled.
    explicit Options(bool is_high_priority_stream = false);

    // return intrusive_ptr of the object
    static c10::intrusive_ptr<Options> create(
        bool is_high_priority_stream = false) {
      return c10::make_intrusive<Options>(is_high_priority_stream);
    }

    // Schedule ECCL operations on high priority GCU streams
    bool is_high_priority_stream;

#ifdef ECCL_HAS_COMM_NONBLOCKING
    // Configure ranks
    ecclConfig_t config = ECCL_CONFIG_INITIALIZER;
#endif

    // Optional "parent" backend and color to create communicators from
    // via `ecclCommSplit`
    std::shared_ptr<ProcessGroupECCL> split_from;
    int64_t split_color{0};
    std::vector<uint64_t> global_ranks_in_group;
  };

  // If you wish to create multiple process groups, each with a potentially
  // different rank and size, you can do so by passing a new store instance
  // to each one. If you have only a single store object, you can
  // use the `c10d::PrefixStore` to derive scoped instances.
  // This is also what the Python API in torch.distributed does.
  //
  // The process group instance keeps a reference to the store because
  // it may be used long after the constructor runs. In fact, the constructor
  // doesn't create any ECCL communicators. A single ECCL communicator can
  // only be used on a specific set of devices, and are therefore created
  // on-demand when a collective runs. If another collective is executed later,
  // against a different set of devices, the process group creates another ECCL
  // communicator. These ECCL communicators are cached and reused if possible.
  //
  ProcessGroupECCL(const c10::intrusive_ptr<c10d::Store>& store, int rank,
                   int size,
                   c10::intrusive_ptr<Options> options = Options::create());

  // This constructor includes the deprecated `groupName` argument.
  // If you have existing code that uses the `groupName`, you can replace
  // it by specifying a `c10d::PrefixStore(groupName, store)` for store.
  C10_DEPRECATED ProcessGroupECCL(
      const c10::intrusive_ptr<c10d::Store>& store, int rank, int size,
      const std::string& groupName,
      c10::intrusive_ptr<Options> options = Options::create())
      : ProcessGroupECCL(store, rank, size, options) {}

  ~ProcessGroupECCL() override;

  uint64_t getUid() { return static_cast<uint64_t>(uid_); }

  c10::intrusive_ptr<Options> getOptions() { return options_; }

  const std::string getBackendName() const override {
    return std::string(ECCL_BACKEND_NAME);
  }

  const std::vector<uint8_t> getUniqueID() const {
    TORCH_CHECK_WITH(
        DistBackendError, ecclUniqueId_.size() != 0,
        "must trigger a communication operation before call getUniqueId");
    return ecclUniqueId_;
  }
  bool supportsSplitting() const override { return true; }

  bool supportsCoalescing() const override { return true; }

  void startCoalescing() override;

  c10::intrusive_ptr<c10d::Work> endCoalescing() override;

  // For specifying a composite optype, such as ALLGATHER and REDUCE_SCATTER
  c10::intrusive_ptr<c10d::Work> endCoalescing(c10d::OpType optype);

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

  c10::intrusive_ptr<c10d::Work> _broadcast_oop(
      at::Tensor& outputTensors, at::Tensor& inputTensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions());

  c10::intrusive_ptr<c10d::Work> allreduce_sparse(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

  c10::intrusive_ptr<c10d::Work> allreduce_outplace(
      at::Tensor& outputTensor, at::Tensor& inputTensor,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions());

  c10::intrusive_ptr<c10d::Work> allgatherv(
      at::Tensor& outputTensor, at::Tensor& inputTensor,
      const std::vector<size_t>& recvCounts,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions());

  c10::intrusive_ptr<c10d::Work> reduce_scatterv(
      at::Tensor& outputTensor, at::Tensor& inputTensor,
      const std::vector<size_t>& recvCounts,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions());

  c10::intrusive_ptr<c10d::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts =
          c10d::AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;

  c10::intrusive_ptr<c10d::Work> _reduce_oop(
      at::Tensor& outputTensors, at::Tensor& inputTensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions());

  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> _allgather_base(
      at::Tensor& outputbuffer, at::Tensor& inputbuffer,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(
      at::Tensor& outputTensor, at::Tensor& inputTensor,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;

  c10::intrusive_ptr<c10d::Work> alltoall_base(
      at::Tensor& outputTensor, at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<c10d::Work> alltoallv_d(
      at::Tensor& outputTensor, at::Tensor& inputTensor,
      at::Tensor& outputSplitSizes, at::Tensor& inputSplitSizes, int32_t flag,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions());

  c10::intrusive_ptr<c10d::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<c10d::Work> send(std::vector<at::Tensor>& tensors,
                                      int dstRank, int tag) override;

  c10::intrusive_ptr<c10d::Work> recv(std::vector<at::Tensor>& tensors,
                                      int srcRank, int tag) override;

  void groupStart();

  void groupEnd();

  void groupEndNonblocking(std::shared_ptr<ECCLComm> comm);

  c10::intrusive_ptr<c10d::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override;

  // Unsupported Ops
  c10::intrusive_ptr<c10d::Work> recvAnysource(std::vector<at::Tensor>& tensors,
                                               int tag) override;

  // Agrees on an initial sequence number for the whole group by having rank 0
  // create it and broadcast it to other ranks using the store.
  void setSequenceNumberForGroup() override;

  // Retrieves the current sequence number for the whole group, which should be
  // in sync. If the returned number is not consistent across the group, it
  // may indicate that there is some sort of collective desynchronization.
  uint64_t getSequenceNumberForGroup() override;

  // Return the total number of splits the communicators held by this process
  // group have performed.
  uint64_t getCommSplitCounter() const;

  void registerOnCompletionHook(
      std::function<void(std::shared_ptr<c10d::WorkInfo>)>&& hook) override;
  void waitForPendingWorks() override;

  void enableCollectivesTiming() override;

  // Helper function for iteratively aborting communicators in the provided map
  void abortCommsFromMap(
      std::unordered_map<std::string, std::shared_ptr<ECCLComm>>& ecclCommsMap,
      c10::optional<std::string> abortReason);

  //   c10::intrusive_ptr<c10d::intra_node_comm::IntraNodeComm>
  //   initIntraNodeComm();

  // Provides an API to abort the ProcessGroup (similar to ecclCommAbort)
  // instead of relying on ProcessGroupECCL destructor.
  // return true if abort is successful, otherwise false
  bool abort(c10::optional<std::string> abortReason = c10::nullopt);

  void shutdown(c10::optional<std::string> reason = c10::nullopt);

  void eagerConnectSingleDevice(at::Device device) override;

  void performNocolorSplit(at::Device device);

  // Helper that either looks up the cached ECCL communicators or creates
  // a new set of ECCL communicators as a cache entry
  std::shared_ptr<ECCLComm> getECCLComm(const std::string& deviceKey,
                                        at::Device& device, c10d::OpType opType,
                                        int p2pRank = 0,
                                        bool isSendRecvSelf = false);

 protected:
  // Helper that broadcasts eccl unique ID to all ranks through the store
  void broadcastUniqueECCLID(ecclUniqueId* ecclID, bool isSingleP2POp,
                             const std::string& devicesKey, int p2pRank);

  // Wrapper method which can be overridden for tests.
  virtual std::exception_ptr checkForECCLErrors(
      std::shared_ptr<ECCLComm>& ecclComm);

  // Ensure thaht if record is True, the work obj will be enqueued via
  // workEnqueue
  virtual c10::intrusive_ptr<ProcessGroupECCL::WorkECCL> initWork(
      at::Device& device, int rank, c10d::OpType opType,
      const char* profilingTitle = nullptr,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {}, bool record = false);

  // In the timeout case and we will dump debug info such as the ECCL flight
  // recorder to storage. Down the road, if we have more complicated or blocking
  // operations, we might need to use a side thread to do it.
  bool dumpDebuggingInfo();

 private:
  int globalRankStart;
  int globalRankStride;

  // Helper that encapsulates work shared across all collective communication
  // primitives.  The callbacks have the following signatures:
  //
  //    ecclResult_t fn(at::Tensor& input, at::Tensor& output,
  //                    ecclComm_t, torch_gcu::GCUStream&);
  //    void {pre,post}(std::vector<torch_gcu::GCUStream&>);
  template <typename Fn>
  c10::intrusive_ptr<c10d::Work> collective(
      at::Tensor& input, at::Tensor& output, Fn fn, c10d::OpType opType,
      const char* profilingTitle = nullptr, bool avoidRecordStreams = false);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<c10d::Work> collective(
      at::Tensor& input, at::Tensor& output, Fn fn, PreProcess pre,
      PostProcess post, c10d::OpType opType,
      const char* profilingTitle = nullptr, bool avoidRecordStreams = false);

  template <typename Fn>
  c10::intrusive_ptr<c10d::Work> collectiveCoalesced(
      std::vector<at::Tensor>& input, std::vector<at::Tensor>& output, Fn fn,
      c10d::OpType opType, const char* profilingTitle = nullptr,
      bool avoidRecordStreams = false);

  // Helper that encapsulates work shared across point-to-point communication
  // primitives. It is the same structure as the helper used for collective
  // communication primitives.
  template <typename Fn>
  c10::intrusive_ptr<c10d::Work> pointToPoint(
      at::Tensor& tensor, Fn fn, int peer, c10d::OpType opType,
      const char* profilingTitle = nullptr);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<c10d::Work> pointToPoint(at::Tensor& tensor, Fn fn,
                                              int peer, c10d::OpType opType,
                                              PreProcess pre, PostProcess post,
                                              const char* profilingTitle);

  c10::intrusive_ptr<c10d::Work> allreduce_impl(
      at::Tensor& tensor,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions());

  // Checks for ECCL errors on each of the communicators and returns an
  // appropriate exception_ptr (nullptr if no errors).
  static std::exception_ptr checkForECCLErrorsInternal(
      std::shared_ptr<ECCLComm>& ecclComm);

  // Function that runs as part of a separate thread and checks for errors on
  // ECCL communicators. We need a separate thread to check for ECCL errors
  // since we can't rely on the user calling certain methods like wait(),
  // isCompleted() etc. to detect and remediate errors. In addition to this, we
  // need a mechanism to safely abort and remove ECCL communicators from our
  // cache. This can be done cleanly by having a thread for the ProcessGroupECCL
  // class. Attempting to modify the communicator cache from the WorkECCL class
  // might run into issues with object lifetime since the ProcessGroupECCL
  // object might get destroyed before the WorkECCL object.
  void ecclCommWatchdog();

  // Return the GCU device most likely associated with this backend.
  // If we aren't bound to a specific device, there is no strict
  // guarantee that this heuristic is the correct assignment of ranks
  // to GCUs that Python layers use, but in practice it tends to be.
  // Fortunately we don't rely on this for correctness of any tensor
  // operations, just for ancillary uses like barriers.
  at::Device guessDeviceForRank() const;

  // Destroys initialized ECCL communicators in devECCLComMap_ given by input
  // key. Throws if there are no communicators to destroy. Also removes
  // communicators from the cache and clears used device indices.
  void destroyECCLComms(const std::string& devECCLCommMapKey);

  // Watchdog's inside loop.
  // Takes care of cleaning up completed work, and aborting upon failure or
  // timeout.
  void watchdogHandler();

  void runHookLoop();

  // Desync debug helper
  void logWorkStart(WorkECCL& work);

  // Desync debug helper
  void logWorkEnd(WorkECCL& work);

  // Generates a prefix that is unique to this process group and rank, for
  // disambiguating logs
  std::string createLogPrefix() const;

  // Returns the unique prefix created in createLogPrefix
  const std::string& logPrefix() const;

  // Returns the global rank of the device. This function assumes that users
  // always create a default global process group(PG) which includes all
  // devices. It is called in the constructor of ProcessGroupECCL, so it always
  // return the rank_ of the the very first PG created, aka, default global PG.
  const int& globalRank() const;

  // Returns the global ranks of a PG.
  const std::vector<uint64_t>& groupRanks() const;

 protected:
  // Function that runs as part of a separate thread aside from watchdog
  // thread because we need to check the heartbeat from watchdog thread
  // so that when we get stuck in some ECCL/GCU calls,
  // we can dump the debugging information and abort the process.
  virtual void heartbeatMonitor();

  // Function that directly trigger std::abort so that the whole process
  // gets terminated.
  virtual void terminateProcess(std::string errMsg);

  // A helper function to wait for a future to complete or timeout.
  void waitForFutureOrTimeout(std::future<bool>& fut,
                              const std::chrono::milliseconds& timeOutMilSec,
                              const std::string& futDescription,
                              bool throwException = false);

  // When watchdog timeout, this function will be called and return debug info
  // for users. For now we only get information from retrieveDesyncReport.
  // We are working on enabling more useful debug information for watchdog
  // timeout.
  virtual std::string getECCLWatchdogDebugInfo();

  static const int64_t kWatchdogThreadSleepMillis;

  // The store is used to broadcast the ECCL unique ID of rank 0. This store
  // comes with prefix and it is different across ProcessGroup ECCL instances
  // (aka, different ProcessGroups).
  c10::intrusive_ptr<c10d::Store> store_;

  // Reference to the store without prefix so that keys are same across all
  // ProcessGroup ECCL instances and (key, value) pairs written to the store are
  // global.
  c10::intrusive_ptr<c10d::Store> globalStore_;

  bool storeError_{false};

  const c10::intrusive_ptr<Options> options_;

  // The number of ECCL communicators that have been created during
  // the lifetime of this process group. This sequence number is
  // used to scope keys used in the store.
  uint64_t ecclCommCounter_{0};

  // The store keys to trace the last ECCL collective kernel GCU events - start
  // event and end event respectively. These are used to do desync root cause
  // analysis.
  const std::string traceKeyStart_;
  const std::string traceKeyEnd_;
  std::vector<std::string> storeKeys_;
  // The ECCL communicator that the process group has cached.
  //
  // For collective operations:
  // The key is a list of GCU devices that an operation is operating on
  // The GCU devices are stored in a device sequence and the cache ECCL
  // communicator is associated with this GCU device sequence
  //
  // e.g. If the process group op only uses device 0, then the value of
  // the used device string stored (value of the hashmap) would be "0".
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 1, 2, 3, 4, 5, 6, 7 separately,
  //      then the value of the used device string (key) stored would be
  //      "0,1,2,3,4,5,6,7"
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 4, 5, 6, 7, 1, 2, 3 separately,
  //      then the value of the used device string stored would be
  //      "0,4,5,6,7,1,2,3"
  //
  //      Note that the order of the device for the tensor list matters.
  //
  // For point-to-point operations:
  // The key is a string of my current rank and the peer process rank.
  // e.g. If process 1 and process 2 are involved in a point-to-point
  // communication, the key will be "1:2" on both processes. Note: this is for
  // the scenario where there is only 1 GCU per process. When it comes to
  // multiple GCUs per process, this part may need to redesigned.
  std::unordered_map<std::string, std::shared_ptr<ECCLComm>> devECCLCommMap_;

  // The ECCL communicators currently in process of being initialized.
  std::unordered_map<std::string, std::shared_ptr<ECCLComm>>
      inInitializationCommMap_;

  // Map from ecclUniqueId to appropriate communicator.
  std::unordered_map<std::string, std::shared_ptr<ECCLComm>> ecclIdToCommMap_;

  // Mutex to guard maps like devECCLCommMap_ and ecclIdToCommMap_.
  std::mutex mutex_;

  // Heartbeat of watchdog thread.
  std::atomic_uint64_t heartbeat_;

  // The time interval used for deciding whether there is no watchdog heartbeat.
  int heartbeatTimeoutInSec_;

  // timeout for the dump to finish.
  int waitTimeoutDumpInMilSec_;

  // Interval of check coordinated signals in ProcessGroupECCL from other ranks
  // e.g., trigger the dump of the debugging info for timeout when notified.
  int coordCheckIntervalMilSec_;

  // Size of ring buffer where we store ECCL Traces for debugging.
  int ecclTraceBufferSize_;

  // We gate the heartbeat monitor thread so that we can roll it out gradually.
  std::atomic<bool> monitorThreadEnabled_;

  // Monitor thread which checks the heartbeat of Watchdog thread.
  // If the monitor thread finds there is no heartbeat, it will dump debug info
  // and then kill the watchdog thread to avoid hang.
  std::thread ecclHeartbeatMonitorThread_;

  // Watchdog thread which looks for errors on the cached ECCL communicators.
  std::thread ecclCommWatchdogThread_;

  std::thread onCompletionHookThread_;

  // Whether or not we should terminate the watchdog and workCleanup threads.
  std::atomic<bool> terminateProcessGroup_;

  // Whether or not we should terminate the heartbeat monitoring threads.
  std::atomic<bool> terminateHeartbeatMonitorThread_;

  // Whether we are in the shutdown mode when we are trying to get debug info,
  // such as desync report.
  std::atomic<bool> collectiveDebugInfoMode_;

  // Whether there are hooks pending to be fired
  std::atomic<bool> hasPendingHooks_;

  // This is the signal from watchdog threads to indicate whether the monitor
  // thread should dump. Making it static so that it is accessible from all the
  // PGs. With this flag, monitor thread would dump debug info under any one of
  // the 3 conditions: 1: this flag is set to true by the watchdog thread when
  // it detects a timeout. 2: timeout signal is received from
  // other ranks through tcpstore 3: no heartbeat of watchdog Note that only the
  // monitor thread from PG0 should dump the debug info and only once
  static std::atomic<bool> shouldDump_;

  // Mutex to Guard workMetaList_
  std::mutex workMetaListMutex_;

  // Mutex to Guard monitorWakeUpCV_
  std::mutex monitorMutex_;

  bool writeDebugInfo_ = false;

  // Condition Variable for watchdog thread sleep
  std::condition_variable workMetaListCV_;

  // Condition Variable for monitor thread to wake up early
  std::condition_variable monitorWakeUpCV_;

  // Vector to Store WorkECCL pointers
  std::list<ProcessGroupECCL::WorkECCL> workMetaList_;

  std::chrono::time_point<std::chrono::steady_clock> lastWorkListUpdateTime_;

  // Mutex to Guard workMetaList_
  std::mutex completedWorkListMutex_;

  // Condition Variable for watchdog thread sleep
  std::condition_variable completedWorkListCV_;

  std::list<ProcessGroupECCL::WorkECCL> completedWorkList_;

  // Add c10d::Work Pointer to workVector
  void workEnqueue(c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>);

  // The GCU streams used by ECCL kernels
  std::unordered_map<std::string, torch_gcu::GCUStream> ecclStreams_;

  // Note(torch_gcu): reuse GCUEvent will increase record time consumption, so
  // each time a new GCUEvent is created The GCU events used to sync ECCL
  // streams and we move ecclEvent_ to WorkECCL from ProcessGroupECCL
  //   std::unordered_map<std::string, torch_gcu::GCUEvent> ecclEvents_;

  // Device Indexes used for all collectives in this group
  std::set<int> usedDeviceIdxs_;

  // Flag to denote if a coalescing groupStart/groupEnd block is active
  int coalescing_state_ = 0;

  // Stores device indexes for all collectives run inside a coalescing block
  std::vector<at::Device> coalescedDevices_;

  // Stores communicators for all collectives run inside a coalescing block
  std::vector<std::shared_ptr<ECCLComm>> coalescedComms_;

  // map from the key: "group name + pg counter (ID)" to the
  // unique ECCL ID count. This needs to be group and pg specific
  //
  // For each process group, we need a uniform unique ECCL ID counter to ensure
  // that ECCL operation in this process group can be completed successfully.
  // Since each process group ID belongs to a group name, the key to this map
  // is a combination of group name and ProcessGroupECCL ID.
  static std::unordered_map<std::string, ssize_t> pgUniqueECCLIDCnt_;

  // map from group name to the pg counter (ID) within that group
  //
  // For each group with the "group name" (which is the key), we need to
  // keep track of a unique process group ID when creating a new
  // ProcessGroupECCL for this "group name". Therefore, the value of this
  // map keeps the unique ProcessGroupECCL's ID for a specific group with
  // the "group name". The reason we need a per-group process group ID counter
  // is that different group can have different ranks and we need ensure that
  // each group has its own uniform process group ID for all its ranks.
  static std::unordered_map<std::string, ssize_t> processGroupCounterMap_;

  // Whether or not wait() and synchronize() are blocking operations that wait
  // for the operation to complete.
  bool blockingWait_ = false;

  // Whether to abort the communicators when users call destroy_process_group().
  // If yes, communicators will be aborted when destroy_process_group is called,
  // but not in destructor.
  bool abortInDestroyProcessGroup_ = false;

  // Whether or not to hook the cache allocator to register all allocated
  // tensors
  bool useTensorRegisterAllocatorHook_ = false;

  // Whether or not the workCleanupThread is used to perform async error
  // handling.
  ErrorHandlingMode asyncErrorHandling_ = NoHandling;

  // Whether or not to enable timeout root cause analysis.
  bool desyncDebug_;

  // Whether or not to dump debug info on timeout
  bool dumpOnTimeout_;

  // Whether or not to create start topsEvent and enable timing for start
  // and end events. Note that enableTiming_ is always true if desyncDebug_
  // is set to true.
  std::atomic<bool> enableTiming_;

  // Flag to enable the print of hash value of input/output of collectives for
  // verification.
  std::atomic<bool> enableCollecticeHashDebug_;

  // Whether or not TORCH_ECCL_AVOID_RECORD_STREAMS was set
  bool avoidRecordStreams_ = false;

  // Set of communicators that this process group has aborted and their
  // ecclUniqueId has been written to the store. We don't need a lock
  // for this map since only the watchdog thread accesses this set. The
  // set contains the string representation of ecclUniqueId.
  std::unordered_set<std::string> abortedComms_;

  // The number of active ecclGroupStart() calls. This counter will be increased
  // by 1 when ecclGroupStart() is called and decreased by 1 when ecclGroupEnd()
  // is called.
  static thread_local uint64_t ecclActiveGroupCounter_;

  // Counting for the sequential number of ECCL collective call.
  // (specifically, how many actual kernels we launched, which differs from
  // op_id_ when coalescing is enabled)
  uint64_t seq_{0};

  // Incrementing counter for logical operations (collective or p2p) issued on
  // the ProcessGroup
  uint64_t op_id_{0};

  // the sequential number of the last collective enqueued into workMetaList_
  // This is useful for identifying a rank that has not join a collective
  uint64_t lastEnqueuedSeq_;

  // the sequential number of the last collective completed marked by
  // the watchdog thread
  uint64_t lastCompletedSeq_;

  std::exception_ptr watchDogException_ = nullptr;

  size_t uid_;

  std::string logPrefix_;
  std::vector<uint8_t> ecclUniqueId_;

  //   c10::intrusive_ptr<c10d::intra_node_comm::IntraNodeComm> intraNodeComm_;
};

TORCH_API std::string dump_eccl_trace();

// Gets a mutable reference to a global optional function.  Heartbeat Monitor
// will query this function and if available, call it to dump traces. Inside
// fbcode, we store a function here that uses an internal tool for process
// tracing
TORCH_API c10::optional<std::function<std::string()>>& get_cpp_trace_dumper();

// Similar to get_cpp_trace_dumper, this stores a function defined in
// torch-python layer that lets us check whether the GIL can be acquired,
// helpful for instrumenting in cases where a hang was observed.
typedef bool (*gil_checker_t)();

TORCH_API gil_checker_t& get_gil_checker();
}  // namespace c10d_gcu

#endif  // USE_C10D_ECCL