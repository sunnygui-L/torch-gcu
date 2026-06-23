
#ifdef USE_C10D_ECCL
#include "distributed/ProcessGroupECCL.hpp"

#include <c10/core/DeviceType.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <eccl.h>
#include <eccl_ext.h>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/torch.h>

#include <cstdint>
#include <exception>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "ATen/ops/empty_like.h"
#include "aten/aot_ops/topsaten_bridge_define.h"
#include "aten/op_debug_config.h"
#include "c10/util/logging_is_not_google_glog.h"
#include "distributed/ECCLUtils.hpp"
#include "gcu/gcu_allocator_config.h"
#include "gcu/gcu_caching_allocator.h"
#include "gcu/gcu_context.h"
#include "gcu/gcu_graph.h"
#include "gcu/gcu_graphs_c10_utils.h"
#include "gcu/gcu_guard.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_utils.h"
#include "gcu/sys_util.h"
#include "gcu/thread_util.h"
#include "gcu/trace.h"
#include "topsaten/topsaten_extensions.h"

namespace c10d_gcu {

constexpr const char* const kECCLAbortedCommStoreKey = "ECCLABORTEDCOMM";
using namespace torch_gcu;

namespace {

// ECCL op mapping
const std::map<c10d::ReduceOp, ecclRedOp_t> ecclOp = {
    {c10d::ReduceOp::MIN, ecclMin}, {c10d::ReduceOp::MAX, ecclMax},
    {c10d::ReduceOp::SUM, ecclSum}, {c10d::ReduceOp::PRODUCT, ecclProd},
    {c10d::ReduceOp::AVG, ecclAvg},
};

// ECCL type typing
std::map<at::ScalarType, ecclDataType_t> ecclDataType = {
    {at::kChar, ecclInt8},
    {at::kByte, ecclUint8},
    {at::kFloat, ecclFloat32},
    {at::kDouble, ecclFloat32},  // GCU do not support 64-bit
    {at::kInt, ecclInt32},
    {at::kLong, ecclInt32},  // GCU do not support 64-bit
    {at::kHalf, ecclHalf},
    {at::kBool, ecclUint8},
    {at::kBFloat16, ecclBfloat16},
};

// Helper function that gets the data type and issues error if not supported
ecclDataType_t getEcclDataType(at::ScalarType type, bool need_reduce = false) {
  auto fp8 = std::set<at::ScalarType>{
      at::ScalarType::Float8_e5m2, at::ScalarType::Float8_e4m3fn,
      at::ScalarType::Float8_e5m2fnuz, at::ScalarType::Float8_e4m3fnuz};
  if (!need_reduce) {
    if (fp8.find(type) != fp8.end()) {
      return ecclInt8;
    }
  }
  auto it = ecclDataType.find(type);
  TORCH_CHECK(
      it != ecclDataType.end(),
      "Input tensor data type is not supported for ECCL process group: ", type);
  return it->second;
}

constexpr const char* const kECCLAbortedCommStoreKey = "ECCLABORTEDCOMM";

#if defined(ECCL_MAJOR) && \
    ((ECCL_MAJOR > 2) || (ECCL_MAJOR == 2) && (ECCL_MINOR >= 10))
#define ECCL_HAS_AVG 1
#endif

bool complexViewAsRealAllowed(const c10d::ReduceOp reduceOp) {
  switch (reduceOp) {
    case c10d::ReduceOp::SUM:
      return true;
    case c10d::ReduceOp::AVG:
      return true;
    case c10d::ReduceOp::PREMUL_SUM:
      return true;
    case c10d::ReduceOp::UNUSED:
      return true;
    default:
      return false;
  }
  return false;
}

#ifdef ENABLE_ECCL_PREMUL_SUM_SUPPORT
template <typename T, ecclDataType_t dataType>
ecclRedOpRAII unpackPreMulSum(const c10d::ReduceOp& reduceOp,
                              const ecclComm_t& comm) {
  const auto* preMulSupplement =
      reinterpret_cast<::c10d::NCCLPreMulSumSupplement*>(
          reduceOp.supplement_.get());
  ecclRedOp_t preMulSum;
  bool has_tensor = preMulSupplement->tensor_factor.defined();
  auto residence = has_tensor ? ecclScalarDevice : ecclScalarHostImmediate;
  const T* ptr_factor =
      has_tensor ? preMulSupplement->tensor_factor.const_data_ptr<T>()
                 : nullptr;
  T scalar_factor = T(preMulSupplement->double_factor);
  ecclRedOpCreatePreMulSum(
      &preMulSum,
      // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html#ecclredopcreatepremulsum
      // tells us that the scalar input is strictly a multiplier.
      /*scalar=*/has_tensor ? const_cast<T*>(ptr_factor) : &scalar_factor,
      dataType, residence, comm);
  return ecclRedOpRAII(preMulSum, comm);
}
#endif

ecclRedOpRAII getEcclReduceOp(const c10d::ReduceOp reduceOp, at::Tensor& input,
                              const ecclDataType_t& dataType,
                              const ecclComm_t& comm) {
  try {
    if (input.scalar_type() == at::kBool) {
      if (reduceOp == c10d::ReduceOp::SUM) {
        // For bool tensors, map sum to max, which both represent a bitwise or.
        // This is to prevent overflow issues with sum, since we use uint8 to
        // represent a bool (see ecclDataType mapping).
        return ecclMax;
      }
#ifdef ECCL_HAS_AVG
      if (reduceOp == c10d::ReduceOp::AVG) {
        TORCH_CHECK(false, "Cannot use ReduceOp.AVG with boolean inputs");
      }
#endif
    }
    if (reduceOp == c10d::ReduceOp::PREMUL_SUM) {
#ifdef ENABLE_ECCL_PREMUL_SUM_SUPPORT
      switch (dataType) {
        case ecclHalf:
          return unpackPreMulSum<at::Half, ecclHalf>(reduceOp, comm);
        case ecclFloat:
          return unpackPreMulSum<float, ecclFloat>(reduceOp, comm);
        case ecclDouble:
          return unpackPreMulSum<double, ecclDouble>(reduceOp, comm);
        default:
          TORCH_CHECK(false,
                      "PreMulSum Data type must be half, float, or double");
          ecclRedOp_t unused;
          return unused;
      }
#else
      TORCH_CHECK(false, "ECCL not supprt PreMulSum");
#endif
    }
    return ecclOp.at(reduceOp);
  } catch (const std::out_of_range& e) {
    switch (reduceOp) {
      case c10d::ReduceOp::BAND:
        TORCH_CHECK(false, "Cannot use ReduceOp.BAND with ECCL");
        break;
      case c10d::ReduceOp::BOR:
        TORCH_CHECK(false, "Cannot use ReduceOp.BOR with ECCL");
        break;
      case c10d::ReduceOp::BXOR:
        TORCH_CHECK(false, "Cannot use ReduceOp.BXOR with ECCL");
        break;
      default:
        TORCH_CHECK(false, "Unhandled ReduceOp");
        break;
    }
  }
}

// Get a key string from device
inline std::string getKeyFromDevice(at::Device& device) {
  return std::to_string(device.index());
}

std::string getKeySendRecv(int myRank, int peer) {
  int lowRank = myRank < peer ? myRank : peer;
  int highRank = myRank < peer ? peer : myRank;
  std::string sendRecvPair =
      std::to_string(lowRank) + ":" + std::to_string(highRank);
  return sendRecvPair;
}

// Get device from tensor
inline at::Device getDevice(at::Tensor& tensor) { return tensor.device(); }

// [Sync Streams] Helper that lets the input ecclStreams to wait for the current
// stream. ECCL communications run on ecclStreams, but input tensors are
// allocated on different streams (i.e., current streams). Communications on
// ecclStreams cannot start before pending input tensor ops on current streams
// finish. Otherwise, ops on two streams might read/write same tensors
// concurrently.
//
// The synchronization above alone is not enough. We also need to make sure
// input tensors are not freed before their usages on ecclStreams finish. This
// can be achieved by calling torch_gcu::GCUCachingAllocator::recordStream,
// which remembers the usage stream (ecclStream), creates an event on the usage
// stream when GC attempts to free the input tensor, and delays GC until that
// event is done.
void syncStream(at::Device& device, torch_gcu::GCUEvent& ecclEvent,
                torch_gcu::GCUStream& ecclStream) {
  if (torch_gcu::OpDebugConfig::GetInstance().enableSyncMode()) {
    StreamSynchronize(torch_gcu::getCurrentGCUStream(device.index()));
  }

  ecclEvent.record(torch_gcu::getCurrentGCUStream(device.index()));
  ecclEvent.block(ecclStream);
}

// Given a ecclUniqueId, convert it to a string representation that can be put
// in the store.
std::string buildEcclUniqueIdStr(const ecclUniqueId& ecclID) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&ecclID);
  std::ostringstream oss;
  for (const auto i : c10::irange(ECCL_UNIQUE_ID_BYTES)) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

std::string getEcclAbortedCommStoreKey(const std::string ecclIdStr) {
  return std::string(kECCLAbortedCommStoreKey) + ":" + ecclIdStr;
}

// Returns exception's what() given an exception_ptr instance.
std::string getExceptionMsgFromExceptionPtr(
    const std::exception_ptr& exceptionPtr) {
  TORCH_CHECK(exceptionPtr != nullptr);
  try {
    std::rethrow_exception(exceptionPtr);
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "Unknown exception type";
  }
}

int64_t get_gcu_element_size(const at::Tensor& tensor) {
  if (torch_gcu::is_narrow_type(tensor.scalar_type())) {
    return 4;
  }
  return tensor.element_size();
}

inline void errorIfCapturingNonCapturableECCL(torch_gcu::CaptureStatus status) {
  // do nothing for gcu
}

}  // namespace

// Map from each communicator to its device index.
// This map is used when register/deregister cache segments from cache
// allocator. See design notes below:
// - Each segment should be registered only to the communicator on the
//   same device.
// - We cannot reuse devECCLCommMap_ in each ProcessGroup because the key may be
//   ranks rather than device in point-to-point case.
// - This map has also to be maintained as global variable since the register
//   hooks are called outside the scope of any PG, thus we need traverse
//   communicators in all PGs.
static std::unordered_map<std::shared_ptr<ECCLComm>, int> ecclCommDevIdxMap;
static std::mutex ecclCommDevIdxMapMutex;
static bool allocatorHooksAttached = false;

std::atomic<bool> ProcessGroupECCL::shouldDump_(false);

void cacheAllocatorRegisterHook(
    const torch_gcu::GCUCachingAllocator::TraceEntry& te) {
  // Register after SEGMENT_ALLOC
  if (te.action_ !=
      torch_gcu::GCUCachingAllocator::TraceEntry::Action::SEGMENT_ALLOC) {
    return;
  }

  std::lock_guard<std::mutex> lock(ecclCommDevIdxMapMutex);
  for (auto& it : ecclCommDevIdxMap) {
    auto& ecclComm = it.first;
    auto& devIdx = it.second;
    if (te.device_ == devIdx) {
      ecclComm->registerSegment(reinterpret_cast<void*>(te.addr_), te.size_);
    }
  }
}

void cacheAllocatorDeregisterHook(
    const torch_gcu::GCUCachingAllocator::TraceEntry& te) {
  // deregister before SEGMENT_FREE
  if (te.action_ !=
      torch_gcu::GCUCachingAllocator::TraceEntry::Action::SEGMENT_FREE) {
    return;
  }

  std::lock_guard<std::mutex> lock(ecclCommDevIdxMapMutex);
  for (auto& it : ecclCommDevIdxMap) {
    auto& ecclComm = it.first;
    auto& devIdx = it.second;
    if (te.device_ == devIdx) {
      ecclComm->deregisterSegment(reinterpret_cast<void*>(te.addr_));
    }
  }
}

#if defined(IS_ECCL_EXP) && defined(ECCL_COMM_DUMP)
std::string dump_eccl_trace() {
  std::unordered_map<
      std::string /* ecclUniqueID */,
      std::unordered_map<std::string, std::string> /* dump from this comm */>
      ecclDumpMap;
  // dump_eccl_trace is only called from the default PG (uid_=0), but we want to
  // dump from all comms so we need to iterate over ecclCommDevIdxMap, which
  // is static
  std::vector<std::shared_ptr<ECCLComm>> allECCLComms;
  // within the critical section, we don't want to dump while holding the lock
  // as dump might hang
  ecclCommDevIdxMapMutex.lock();
  for (auto& [ecclComm, _] : ecclCommDevIdxMap) {
    allECCLComms.push_back(ecclComm);
  }
  ecclCommDevIdxMapMutex.unlock();
  for (auto& ecclComm : allECCLComms) {
    std::string ecclUniqueIDStr = buildEcclUniqueIdStr(ecclComm->getEcclId());
    ecclDumpMap[ecclUniqueIDStr] = ecclComm->ecclCommDump();
  }
  // return ECCLTraceBuffer::get()->dump(ecclDumpMap);
  return "";
}
#else
std::string dump_eccl_trace() {
  return "";
  // return ECCLTraceBuffer::get()->dump(c10::nullopt);
}
#endif

c10::optional<std::function<std::string()>>& get_cpp_trace_dumper() {
  static c10::optional<std::function<std::string()>> dumper(c10::nullopt);
  return dumper;
}

gil_checker_t& get_gil_checker() {
  static gil_checker_t gil_checker = nullptr;
  return gil_checker;
}

std::future<bool> launchAsyncGilCheck() {
  std::promise<bool> resultPromise;
  std::future<bool> resultFuture = resultPromise.get_future();
  TORCH_CHECK(get_gil_checker(), "Can't check GIL with null GIL checker");
  std::thread workerThread([promise = std::move(resultPromise)]() mutable {
    torch_gcu::util::setThreadName("TorchDistAsyncG");
    try {
      auto& gil_checker = get_gil_checker();
      promise.set_value((*gil_checker)());
    } catch (...) {
      promise.set_exception(std::current_exception());
    }
  });

  // Detach the thread to allow it to run independently
  workerThread.detach();

  return resultFuture;
}

// Return GCU device with ordinal given by input rank.  If we aren't
// bound to a specific device, there is no strict guarantee that this
// heuristic is the correct assignment of ranks to GCUs that Python
// layers use, but in practice it tends to be.  Fortunately we don't
// rely on this for correctness of any tensor operations, just for
// ancillary uses like barriers.
at::Device ProcessGroupECCL::guessDeviceForRank() const {
  TORCH_CHECK_WITH(ValueError, rank_ >= 0, "Invalid rank ", rank_);
  if (c10d::Backend::getBoundDeviceId()) {
    return *c10d::Backend::getBoundDeviceId();
  } else {
    auto numGCUs = torch_gcu::device_count();
    int16_t deviceIdx = static_cast<int16_t>(rank_ % numGCUs);
    return at::Device(at::DeviceType::PrivateUse1, deviceIdx);
  }
}

const int64_t ProcessGroupECCL::kWatchdogThreadSleepMillis = 100;
constexpr int64_t kSynchronizeBusyWaitMillis = 10;
thread_local uint64_t ProcessGroupECCL::ecclActiveGroupCounter_ = 0;

std::ostream& operator<<(std::ostream& output,
                         const ProcessGroupECCL::WorkECCL& workECCL) {
  std::string workInfo;
  workInfo = c10::str("WorkECCL(", "SeqNum=", workECCL.seq_, ", c10d::OpType=",
                      workECCL.opType_ != c10d::OpType::UNKNOWN
                          ? opTypeToString(workECCL.opType_)
                          : workECCL.profiling_title_,
                      ", NumelIn=", workECCL.numelIn_,
                      ", NumelOut=", workECCL.numelOut_,
                      ", Timeout(ms)=", workECCL.opTimeout_.count(), ")");
  return output << workInfo;
}

ProcessGroupECCL::WorkECCL::WorkECCL(
    at::Device& device, int rank, c10d::OpType opType, uint64_t seq,
    const char* profilingTitle,
    const c10::optional<std::vector<at::Tensor>>& inputs, bool desyncDebug,
    bool enableTiming, c10d::DebugLevel distDebugLevel)
    : c10d::Work(rank, opType, profilingTitle, inputs),
      device_(device),
      workStartTime_(std::chrono::steady_clock::now()),
      seq_(seq),
      profiling_title_(profilingTitle),
      timingEnabled_(enableTiming),
      distDebugLevel_(distDebugLevel) {
  // Creates the GCU event wrappers
  // Note: The actual events are lazily created when first recorded to with
  // DEFAULT_FLAGS = topsEventDisableTiming.
  if (enableTiming) {
    ecclStartEvent_ = std::make_shared<torch_gcu::GCUEvent>(topsEventDefault);
  }
  ecclEndEvent_ = std::make_shared<torch_gcu::GCUEvent>(
      enableTiming ? topsEventDefault : topsEventDisableTiming);
}

ProcessGroupECCL::WorkECCL::WorkECCL(const WorkECCL& w)
    : c10d::Work(w.rank_, w.opType_),
      std::enable_shared_from_this<WorkECCL>(w),
      device_(w.device_),
      ecclStartEvent_(w.ecclStartEvent_),
      ecclEndEvent_(w.ecclEndEvent_),
      ecclComm_(w.ecclComm_),
      blockingWait_(w.blockingWait_),
      opTimeout_(w.opTimeout_),
      workStartTime_(w.workStartTime_),
      seq_(w.seq_),
      startTraceUpdated_(w.startTraceUpdated_),
      profiling_title_(w.profiling_title_),
      numelIn_(w.numelIn_),
      numelOut_(w.numelOut_),
      store_(w.store_),
      timingEnabled_(w.timingEnabled_),
      trace_id_(w.trace_id_),
      distDebugLevel_(w.distDebugLevel_) {
  exception_ = w.exception_;
}

ProcessGroupECCL::WorkECCL::~WorkECCL() = default;

bool ProcessGroupECCL::WorkECCL::isCompleted() {
  checkAndSetException();
  return exception() || finishedGCUExecutionInternal();
}

bool ProcessGroupECCL::WorkECCL::isStarted() {
  checkAndSetException();
  return exception() || startedGCUExecutionInternal();
}

bool ProcessGroupECCL::WorkECCL::isSuccess() const {
  C10_THROW_ERROR(NotImplementedError, "WorkECCL::isSuccess() is deprecated");
}

void ProcessGroupECCL::WorkECCL::checkAndSetException() {
  if (exception()) {
    // We already have an exception.
    return;
  }

  auto exception_ptr = checkForECCLErrors();
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
  if (exception_) {
    LOG(INFO) << logPrefix()
              << "found async exception when checking for ECCL errors: "
              << getExceptionMsgFromExceptionPtr(exception_);
  }
}

const std::string& ProcessGroupECCL::WorkECCL::logPrefix() const {
  static std::string prefix = c10::str("[Rank ", rank_, "] ");
  return prefix;
}

void ProcessGroupECCL::WorkECCL::setException(
    std::exception_ptr exception_ptr) {
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
}

// Helper that checks if the ECCL kernels are completed on the GCUs
bool ProcessGroupECCL::WorkECCL::finishedGCUExecution() {
  checkAndSetException();
  return finishedGCUExecutionInternal();
}

bool ProcessGroupECCL::WorkECCL::startedGCUExecutionInternal() const {
  // if timing is disabled we won't have allocated start events
  if (!timingEnabled_) {
    return false;
  }
  // Checking the work's corresponding GCU event's status
  if (!ecclStartEvent_->query()) {
    return false;
  }
  return true;
}

bool ProcessGroupECCL::WorkECCL::finishedGCUExecutionInternal() const {
  // Checking the work's corresponding GCU event's status
  if (!ecclEndEvent_->query()) {
    return false;
  }
  return true;
}

bool ProcessGroupECCL::WorkECCL::checkTimeout(
    c10::optional<std::chrono::milliseconds> timeout) {
  auto currentTimepoint = std::chrono::steady_clock::now();
  auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      currentTimepoint - workStartTime_);
  auto workTimeout = timeout ? *timeout : opTimeout_;

  if (timeElapsed < workTimeout) return false;

  // Timed out

  // There is already an error, we don't override it
  if (exception()) return true;

  std::string exceptionMsg = c10::str(
      logPrefix(), "Watchdog caught collective operation timeout: ", *this,
      " ran for ", timeElapsed.count(), " milliseconds before timing out.");

  LOG(ERROR) << exceptionMsg;
  std::exception_ptr exception_ptr =
      std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exceptionMsg));
  setException(exception_ptr);
  return true;
}

void ProcessGroupECCL::WorkECCL::handleException(
    ErrorHandlingMode errorHandling) {
  if (exception_) {
    auto exceptionMsg = c10::str(
        "Some ECCL operations have failed or timed out. Due to the ",
        "asynchronous nature of GCU kernels, subsequent GCU operations ",
        "might run on corrupted/incomplete data.");
    LOG(ERROR) << logPrefix() << exceptionMsg;
    C10_LOG_API_USAGE_ONCE("ProcessGroupECCL.WorkECCL.handleException");

    if (SHOULD_TEAR_DOWN(errorHandling)) {
      auto tearDownMsg = c10::str(
          "To avoid data inconsistency, we are taking the entire process "
          "down.");
      LOG(ERROR) << logPrefix() << tearDownMsg;
      std::rethrow_exception(exception_);
    }
  }
}

void ProcessGroupECCL::WorkECCL::synchronize() {
  // Call Synchronize without a timeout. We use this method to avoid adding a
  // timeout argument to the public synchronize API.
  synchronizeInternal(kNoTimeout);
}

void ProcessGroupECCL::WorkECCL::synchronizeStream() {
  auto currentStream = torch_gcu::getCurrentGCUStream(device_.index());
  // Block the current stream on the ECCL stream
  ecclEndEvent_->block(currentStream);

  if (avoidRecordStreams_) {
    stashed_for_allocator_safety_->clear();
  }
}

DEFINE_BRIDGE_TOPSATENOP_WITH_NAMESPACE(topsextsDispatchPostprocess, topsexts);
DEFINE_BRIDGE_TOPSEXATSATENOP_OUT2_WITH_NAMESPACE(topsextsCombinePreprocess,
                                                  topsexts);
// Waiting on the work's corresponding GCU events
void ProcessGroupECCL::WorkECCL::synchronizeInternal(
    std::chrono::milliseconds timeout) {
  synchronizeStream();
  // In case of blocking, wait for the operation to complete.
  if (blockingWait_) {
    while (!isCompleted()) {
      bool timedOut = checkTimeout(
          timeout == kNoTimeout ? c10::nullopt : c10::make_optional(timeout));
      // Explicitly abort ecclComms here before throwing this timed out
      // exception to users.
      // If throwing timed out excepiton without aborting eccl communicators
      // here, it was observed that GCU GCU will have 100% utilization and
      // can not run new events successfully.
      if (timedOut) {
        std::string exceptionMsg = c10::str(
            logPrefix(), "c10d::Work ", (*this),
            " timed out in blocking wait (TORCH_ECCL_BLOCKING_WAIT=1).");
        LOG(ERROR) << exceptionMsg;
        break;
      }
      // Yield
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
    // exception() includes timeout and error during blocking wait
    if (exception()) {
      // Abort ECCL communicators
      abort();
      // Throw exception (from main thread here)
      handleException(TearDown);
    }
  }

  // Device synchronize only after we've completed timeout checks.
  if (barrierTensor_.defined()) {
    // If we use the work to do barrier, we should block here
    // `dist.barrier()` only requires all CPU processes to enter this
    // function, hence we only need to make sure the dummy all-reduce has
    // completed. So we would only need to sync the **current stream** back to
    // host, and do not need to synchronize the entire device (which may have
    // kernels running on other streams).
    // Using `topsStreamSynchronize` instead of `topsDeviceSynchronize` can:
    // - lower chance of hang;
    // - CurrentGCUStream is usually the context of the next operation in
    // Python, thus blocking current stream would already block the next
    // compute kernel;
    // - achieve better barrier performance.
    auto currentStream = torch_gcu::getCurrentGCUStream(device_.index());
    StreamSynchronize(currentStream);
  }
}

// Same as calling synchronize().
bool ProcessGroupECCL::WorkECCL::wait(std::chrono::milliseconds timeout) {
  DIST_API_TRACE_FUNC();
  RECORD_PARAM_COMMS(static_cast<int>(this->seq_),      // seq
                     std::make_tuple(pgUID_, pgDesc_),  // PG name tuple
                     rank_,                             // rank
                     "wait",                            // colName
                     0,                                 // inNelems
                     0,                                 // outNelems
                     at::kByte,                         // dType
                     std::vector<int64_t>(),            // inSplitSizes
                     std::vector<int64_t>(),            // outSplitSizes
                     -1, -1,
                     static_cast<int>(1));  // number of device?
  synchronizeInternal(timeout);
  // TODO(kwen2501): this should be moved to c10d tests, to qualify a ECCL
  // upgrade. Once a ECCL version is qualified, this code should not be needed
  // at runtime.
#ifdef PGECCL_ENABLE_HASH
  if (distDebugLevel_ >= c10d::DebugLevel::Detail) {
    auto numel = getTensorsNumel(*outputs_);
    auto hashValue = hashTensors(*outputs_);
    PRINT_COLLECTIVE_HASH_SIGNATURE("output", opTypeToString(opType_), numel,
                                    hashValue);
  }
#endif
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupECCL::WorkECCL::abort() {
  // Abort all communicators of this work
  ecclComm_->ecclCommAbort();

  ecclCommDevIdxMapMutex.lock();
  ecclCommDevIdxMap.erase(ecclComm_);
  ecclCommDevIdxMapMutex.unlock();
}

static std::atomic<size_t> process_group_id = 0;

constexpr const char* MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple. You are probably using "
    "multiple "
    "devices under one thread. The support for such usage has been deprecated. "
    "For details, please refer to "
    "https://pytorch.org/docs/stable/"
    "distributed.html#multi-gcu-collective-functions. "
    "ProcessGroupECCL continues supporting multi-process and multi-thread "
    "modes.";

ProcessGroupECCL::ProcessGroupECCL(const c10::intrusive_ptr<c10d::Store>& store,
                                   int rank, int size,
                                   c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(store),
      options_(options),
      ecclCommCounter_(0),
      traceKeyStart_(c10d::getTraceStartKey("ECCL", rank)),
      traceKeyEnd_(c10d::getTraceEndKey("ECCL", rank)),
      terminateProcessGroup_(false),
      terminateHeartbeatMonitorThread_(false),
      collectiveDebugInfoMode_(false),
      // intraNodeComm_(initIntraNodeComm()),
      uid_(process_group_id++) {
  TORCH_CHECK_WITH(
      ValueError, torch_gcu::device_count() != 0,
      "ProcessGroupECCL is only supported with GCUs, no GCUs found!");
  logPrefix_ = createLogPrefix();
  blockingWait_ = c10d::getCvarBool(TORCH_ECCL_BLOCKING_WAIT, false);
  abortInDestroyProcessGroup_ =
      c10d::getCvarBool(TORCH_ECCL_ABORT_IN_DESTROY_PG, false);
  asyncErrorHandling_ = static_cast<ErrorHandlingMode>(
      c10d::getCvarInt(TORCH_ECCL_ASYNC_ERROR_HANDLING, 3 /*SkipCleanUp*/));
  desyncDebug_ = c10d::getCvarBool(TORCH_ECCL_DESYNC_DEBUG, false) ||
                 (dist_debug_level_ >= c10d::DebugLevel::Detail);
  dumpOnTimeout_ = c10d::getCvarBool(TORCH_ECCL_DUMP_ON_TIMEOUT, false) ||
                   (dist_debug_level_ >= c10d::DebugLevel::Detail);
  heartbeat_ = 1ULL;
  monitorThreadEnabled_.store(
      c10d::getCvarBool(TORCH_ECCL_ENABLE_MONITORING, true));
  heartbeatTimeoutInSec_ =
      c10d::getCvarInt(TORCH_ECCL_HEARTBEAT_TIMEOUT_SEC, 60 * 10 /*10 Mins*/);
  waitTimeoutDumpInMilSec_ = c10d::getCvarInt(
      TORCH_ECCL_WAIT_TIMEOUT_DUMP_MILSEC, 60 * 1000 /*60 Sec*/);
  coordCheckIntervalMilSec_ =
      c10d::getCvarInt(TORCH_ECCL_COORD_CHECK_MILSEC, 1000);
  ecclTraceBufferSize_ = c10d::getCvarInt(TORCH_ECCL_TRACE_BUFFER_SIZE, 0);
  // ECCLTraceBuffer::get()->record_pg_ranks(uid_, groupRanks());
  enableCollecticeHashDebug_ = (dist_debug_level_ >= c10d::DebugLevel::Detail);
  // store_ usually is wrapped with PrefixStore and the prefix is different
  // across different ProcessGroupECCL(PG) instances. We need to get the
  // underlying non-PrefixStore for sharing global information shared across
  // different PGs.
  c10d::PrefixStore* prefixStore =
      dynamic_cast<c10d::PrefixStore*>(store_.get());
  globalStore_ =
      prefixStore ? prefixStore->getUnderlyingNonPrefixStore() : store_;
#ifdef ENABLE_ECCL_ERROR_CHECKING
  enableTiming_.store(c10d::getCvarBool(TORCH_ECCL_ENABLE_TIMING, false) ||
                      desyncDebug_);
#endif
  avoidRecordStreams_ =
      c10d::getCvarBool(TORCH_ECCL_AVOID_RECORD_STREAMS, false);
#ifdef ECCL_HAS_COMM_REGISTER
  useTensorRegisterAllocatorHook_ =
      c10d::getCvarBool(TORCH_ECCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK, false);
  if (torch_gcu::GCUCachingAllocator::GCUAllocatorConfig::
          expandable_segments()) {
    useTensorRegisterAllocatorHook_ = false;
    LOG(INFO)
        << logPrefix()
        << "disables TORCH_ECCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK because it "
           "is not compatible with GCU allocator expandable segments mode.";
  }
#endif

  if (blockingWait_) {
    if (asyncErrorHandling_ != NoHandling || desyncDebug_) {
      LOG(INFO)
          << logPrefix() << "TORCH_ECCL_BLOCKING_WAIT and "
          << "TORCH_ECCL_ASYNC_ERROR_HANDLING|TORCH_ECCL_DESYNC_DEBUG"
          << "should not both be enabled. "
          << "Only TORCH_ECCL_BLOCKING_WAIT is being used in this process.";
      asyncErrorHandling_ = NoHandling;
      desyncDebug_ = false;
    }
  } else {
    if (desyncDebug_ && asyncErrorHandling_ == NoHandling) {
      LOG(INFO)
          << logPrefix()
          << "TORCH_ECCL_DESYNC_DEBUG and TORCH_ECCL_ASYNC_ERROR_HANDLING "
          << "must both be enabled. "
          << "Enabling TORCH_ECCL_ASYNC_ERROR_HANDLING.";
      asyncErrorHandling_ = SkipCleanUp;
    }
  }

#ifdef ENABLE_ECCL_ERROR_CHECKING
  ecclCommWatchdogThread_ =
      std::thread(&ProcessGroupECCL::ecclCommWatchdog, this);
#endif

  init();
  const std::string OFF = "OFF";
  std::string torch_distributed_debug =
      c10d::getCvarString({"TORCH_DISTRIBUTED_DEBUG"}, OFF.c_str());
  LOG(INFO) << logPrefix() << "ProcessGroupECCL initialization options: "
            << "ECCL version: " << getEcclVersion() << ", size: " << size
            << ", global rank: " << globalRank()
            << ", TORCH_ECCL_ASYNC_ERROR_HANDLING: " << asyncErrorHandling_
            << ", TORCH_ECCL_DUMP_ON_TIMEOUT: " << dumpOnTimeout_
            << ", TORCH_ECCL_WAIT_TIMEOUT_DUMP_MILSEC: "
            << waitTimeoutDumpInMilSec_
            << ", TORCH_ECCL_DESYNC_DEBUG: " << desyncDebug_
            << ", TORCH_ECCL_ENABLE_TIMING: " << enableTiming_.load()
            << ", TORCH_ECCL_BLOCKING_WAIT: " << blockingWait_
            << ", TIMEOUT(ms): " << options_->timeout.count()
            << ", USE_HIGH_PRIORITY_STREAM: "
            << options_->is_high_priority_stream
            << ", SPLIT_FROM: " << options_->split_from
            << ", SPLIT_COLOR: " << options_->split_color
            << ", TORCH_DISTRIBUTED_DEBUG: " << torch_distributed_debug
#ifdef ECCL_HAS_COMM_REGISTER
            << ", TORCH_ECCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: "
            << useTensorRegisterAllocatorHook_
#endif
            << ", TORCH_ECCL_ENABLE_MONITORING: "
            << monitorThreadEnabled_.load()
            << ", TORCH_ECCL_HEARTBEAT_TIMEOUT_SEC: " << heartbeatTimeoutInSec_
            << ", TORCH_ECCL_TRACE_BUFFER_SIZE: " << ecclTraceBufferSize_
            << ", TORCH_ECCL_COORD_CHECK_MILSEC: " << coordCheckIntervalMilSec_
            << ", ID=" << this->getID();

  if (options_->global_ranks_in_group.empty()) {
    this->globalRankStart = 0;
  } else {
    this->globalRankStart = options_->global_ranks_in_group[0];
  }

  if (options_->global_ranks_in_group.empty()) {
    this->globalRankStride = 1;
  } else if (options_->global_ranks_in_group.size() == 1) {
    this->globalRankStride = 0;
  } else {
    bool ranksAreStrided = true;
    int startRank = options_->global_ranks_in_group[0];
    int stride =
        options_->global_ranks_in_group[1] - options_->global_ranks_in_group[0];
    for (std::vector<uint64_t>::size_type i = 0;
         i < options_->global_ranks_in_group.size(); i++) {
      if (options_->global_ranks_in_group[i] != startRank + i * stride) {
        ranksAreStrided = false;
        break;
      }
    }

    if (ranksAreStrided) {
      this->globalRankStride = options_->global_ranks_in_group[1] -
                               options_->global_ranks_in_group[0];
    } else {
      this->globalRankStride = -1;
    }
  }

  RECORD_PARAM_COMMS(0,                                   // seq
                     std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                     rank,                                // rank
                     "init",                              // colName
                     0,                                   // inNelems
                     0,                                   // outNelems
                     at::kByte,                           // dType
                     std::vector<int64_t>(),              // inSplitSizes
                     std::vector<int64_t>(),              // outSplitSizes
                     globalRankStart,                     // globalRankStart
                     globalRankStride,                    // globalRankStride
                     size_);                              // worldSize

  // Attach hooks to cache allocator to trigger the hooks whenever a traced
  // action is called. In the following hooks, we register a newly allocated
  // segment when SEGMENT_ALLOC action occurs, and deregister a segment when
  // SEGMENT_FREE action occurs.
  // We attach hooks only once at the first PG creation.
  // Attaching hooks fails if GCUCachingAllocator is not initialized, so
  // lazyInitGCU is called (and is a no-op if GCU is already initialized).
  if (useTensorRegisterAllocatorHook_ && !allocatorHooksAttached) {
    at::globalContext().lazyInitDevice(at::kPrivateUse1);
    torch_gcu::GCUCachingAllocator::attachAllocatorTraceTracker(
        &cacheAllocatorRegisterHook);
    torch_gcu::GCUCachingAllocator::attachAllocatorTraceTracker(
        &cacheAllocatorDeregisterHook);
    allocatorHooksAttached = true;
  }
}

void ProcessGroupECCL::eagerConnectSingleDevice(at::Device device) {
  const auto key = getKeyFromDevice(device);
  LOG(INFO) << logPrefix() << "Eagerly connecting eccl backend with device "
            << device;
  getECCLComm(key, device, c10d::OpType::ALLREDUCE);
}

void ProcessGroupECCL::performNocolorSplit(at::Device device) {
  // If our backend doesn't support splitting, this is a no-op for
  // ranks not in the new subgroup (and ranks that would be in it will
  // just use a new communicator rather than split).
#ifdef ECCL_HAS_COMM_SPLIT
  const auto key = getKeyFromDevice(device);
  LOG(INFO) << logPrefix() << "Performing nocolor split on backend device "
            << device << ", key " << key << ", i am " << this;
  auto comm = getECCLComm(key, device, c10d::OpType::ALLREDUCE);
  ECCLComm::split(comm.get(), ECCL_SPLIT_NOCOLOR, rank_, options_->config);
#endif
}

// TODO(torch_gcu): support intra node communication
// c10::intrusive_ptr<c10d::intra_node_comm::IntraNodeComm>
// ProcessGroupECCL::initIntraNodeComm() {
//   return c10d::intra_node_comm::IntraNodeComm::rendezvous(
//       store_, std::to_string(uid_), rank_, size_);
// }

void ProcessGroupECCL::setSequenceNumberForGroup() {
}  // ECCL just starts sequence numbers at 0.

uint64_t ProcessGroupECCL::getSequenceNumberForGroup() { return seq_; }

void ProcessGroupECCL::registerOnCompletionHook(
    std::function<void(std::shared_ptr<c10d::WorkInfo>)>&& hook) {
  TORCH_WARN_ONCE(
      "ProcessGroupECCL OnCompletion hook will be deprecated in favor of "
      "Flight Recorder. "
      "Please check out FlightRecorder.hpp for information that is recorded at "
      "work completion. "
      "You can file an issue if you want additional information to be "
      "recorded. "
      "You can also file an RFC if you want Flight Recorder to accept plugins "
      "that customize the recording.");

  TORCH_CHECK_WITH(DistBackendError, onCompletionHook_ == nullptr,
                   "ProcessGroupECCL OnCompletion hook already registered");

  TORCH_CHECK_WITH(
      ValueError, enableTiming_.load(),
      "ProcessGroupECCL OnCompletion hook requires recording start and end "
      "events which require setting TORCH_ECCL_ENABLE_TIMING environment "
      "variable. "
      "This is only available for ECCL version >= 2.4.");
  onCompletionHook_ = std::move(hook);
  onCompletionHookThread_ = std::thread(&ProcessGroupECCL::runHookLoop, this);
}

// must release GIL when calling this method
void ProcessGroupECCL::waitForPendingWorks() {
  // Reasoning about hook completion:
  // 1. waitForPendingWorks should be called after user code has finished
  // calling
  //    all collectives. This means, when we got here, all of the collectives
  //    are either in workMetaList_ or has been erased from workMetaList_.
  // 2. The watchdog thread grabs both locks to move c10d::Work object from the
  //    workMetaList_ to the completedWorkList_, and the hook thread only erases
  //    a c10d::Work object after the hook is returned. Therefore, after user
  //    code calls a collective, its c10d::Work object is either in
  //    workMetaList_ or in completedWorkList_ before it finishes.
  // 3. We have three threads and two locks.
  //      a. main thread (this function) grabs two locks atomically
  //      b. watchdog thread (watchdogHandler function) always grabs
  //      workMetaListMutex_
  //         first and then grabs completedWorkListMutex_.
  //      c. hook thread (runHookLoop function) only grabs
  //      completedWorkListMutex_. Therefore, locks are always acquired in the
  //      same order and hence no deadlocks.
  while (true) {
    {
      std::lock(workMetaListMutex_, completedWorkListMutex_);
      std::lock_guard<std::mutex> lockWork(workMetaListMutex_, std::adopt_lock);
      std::lock_guard<std::mutex> lockHook(completedWorkListMutex_,
                                           std::adopt_lock);

      if (workMetaList_.empty() && completedWorkList_.empty()) {
        return;
      }
    }

    std::this_thread::sleep_for(
        std::chrono::milliseconds(kWatchdogThreadSleepMillis));
  }
}

void ProcessGroupECCL::enableCollectivesTiming() { enableTiming_.store(true); }

void ProcessGroupECCL::waitForFutureOrTimeout(
    std::future<bool>& fut, const std::chrono::milliseconds& timeOutMilSec,
    const std::string& futDescription, bool throwException) {
  std::string errorMsg;
  TORCH_CHECK(fut.valid(), "Expected a valid future");
  std::future_status status = fut.wait_for(timeOutMilSec);
  if (status == std::future_status::ready) {
    // Calling .get() will re-raise any exception from the future, and we don't
    // care about the retval
    try {
      bool result = fut.get();
      if (result) {
        LOG(INFO) << logPrefix()
                  << "future is successfully executed for: " << futDescription;
      }
    } catch (const std::exception& e) {
      errorMsg =
          c10::str(logPrefix(), "Exception thrown when waitng for future ",
                   futDescription, ": ", e.what());
      LOG(ERROR) << errorMsg;
    } catch (...) {
      errorMsg = c10::str(logPrefix(),
                          "Unknown exception thrown when waitng for future ",
                          futDescription);
      LOG(ERROR) << errorMsg;
    }
  } else {
    errorMsg = c10::str(logPrefix(), "Future for ", futDescription,
                        " timed out after ", timeOutMilSec.count(), " ms");
    LOG(ERROR) << errorMsg;
  }
  if (throwException && !errorMsg.empty()) {
    C10_THROW_ERROR(DistBackendError, errorMsg);
  }
}

void ProcessGroupECCL::abortCommsFromMap(
    std::unordered_map<std::string, std::shared_ptr<ECCLComm>>& ecclCommsMap,
    c10::optional<std::string> abortReason) {
  // The process may control multiple devices, loop through the communicators on
  // each device
  for (auto& it : ecclCommsMap) {
    auto& devName = it.first;
    auto& ecclComm = it.second;

    LOG(INFO) << logPrefix() << "ProcessGroupECCL destroying ecclComm_ "
              << ecclComm->ecclComm_ << " on GCU device: " << devName;
    ecclComm->ecclCommAbort(abortReason);
    // Note that we don't remove the aborted communicators from the
    // cache. The reason is that if we do remove the communicator
    // from the cache, it is possible that a new collective operation
    // calls `ecclCommInitRank` to create a new communicator whereas
    // other ranks might have failed/timed out and didn't enter
    // `ecclCommInitRank`. As a result, when there is a failure on
    // a communicator the application receives an exception and its
    // their responsibility to destroy the process group and recreate
    // it to recover from errors.

    c10::StreamId streamId = -1;
    if (ecclStreams_.find(devName) != ecclStreams_.end()) {
      auto stream = ecclStreams_.at(devName);
      streamId = stream.id();
    }

    LOG(INFO) << logPrefix() << "ProcessGroupECCL destroyed "
              << " communicator on GCU device: " << devName
              << " with stream: " << streamId;
  }
}

// Abort all communicators on this rank
bool ProcessGroupECCL::abort(c10::optional<std::string> abortReason) {
  // Remove record from global ecclCommDevIdxMapMutex before aborted,
  // so that a new cache segment would not register to already aborted
  // communicators. Note that ecclCommDevIdxMap is a global container which may
  // contain other PG's communicators, thus we need to only erase communicators
  // for the current PG.
  ecclCommDevIdxMapMutex.lock();
  for (auto& it : devECCLCommMap_) {
    auto& ecclComm = it.second;
    ecclCommDevIdxMap.erase(ecclComm);
  }
  ecclCommDevIdxMapMutex.unlock();

  std::lock_guard<std::mutex> lock(mutex_);
  abortCommsFromMap(devECCLCommMap_, abortReason);
  abortCommsFromMap(inInitializationCommMap_, abortReason);
  return true;
}

void ProcessGroupECCL::shutdown(c10::optional<std::string> reason) {
  // Don't join threads here since the purpose of this method is to abort all
  // communicators and signal the threads to exit. Joining on the threads could
  // potentially block and hence avoid it in this method.
  terminateProcessGroup_.store(true);
  workMetaListCV_.notify_one();

  // launch abort asynchronously and wait for it to complete or timeout
  LOG(INFO) << logPrefix()
            << "Launching ProcessGroupECCL abort asynchronously.";
  std::future<bool> fut = std::async(
      std::launch::async, [this, &reason]() { return this->abort(reason); });

  waitForFutureOrTimeout(fut, options_->timeout, "ProcessGroup abort", true);
  LOG(INFO) << logPrefix() << "ProcessGroupECCL aborts successfully.";

  // We need to wait for abort to finish before we can safely shut down
  // heartbeat monitoring thread.
  terminateHeartbeatMonitorThread_.store(true);
  monitorWakeUpCV_.notify_one();
}

ProcessGroupECCL::~ProcessGroupECCL() {
  LOG(INFO) << logPrefix() << "ProcessGroupECCL destructor entered.";
  for (const auto& key : storeKeys_) {
    store_->deleteKey(key);
  }
  if (!terminateProcessGroup_.load()) {
    // Only if TORCH_ECCL_ABORT_IN_DESTROY_PG is enabled, terminateProcessGroup_
    // will be set to true through destroy_process_group
    if (abortInDestroyProcessGroup_) {
      LOG(WARNING) << c10::str(
          "WARNING: process group has NOT been destroyed before it is being "
          "destructed. ",
          "On normal program exit, the application should call "
          "destroy_process_group to ",
          "ensure that any pending ECCL data transfers have finished in this "
          "process. "
          "In rare cases this process can exit before this point and block the "
          "progress of "
          "another member of the process group. This constraint has always "
          "been present, "
          " but this warning has only been added since PyTorch 2.3");
    }
    // If user haven't explicitly destroy/shutdown process group, destructor
    // needs to do so
    shutdown();
  }

  // Wait for all threads to finish before returning
#ifdef ENABLE_ECCL_ERROR_CHECKING
  if (ecclCommWatchdogThread_.joinable()) {
    ecclCommWatchdogThread_.join();
    LOG(INFO) << logPrefix() << "ProcessGroupECCL watchdog thread joined.";
  }
  if (ecclHeartbeatMonitorThread_.joinable()) {
    ecclHeartbeatMonitorThread_.join();
    LOG(INFO) << logPrefix()
              << "ProcessGroupECCL heart beat monitor thread joined.";
  }
#endif
  if (onCompletionHookThread_.joinable()) {
    onCompletionHookThread_.join();
    LOG(INFO) << logPrefix()
              << "ProcessGroupECCL onCompletionHookThread thread joined.";
  }
}

bool ProcessGroupECCL::dumpDebuggingInfo() {
  // Serialize all calls to this function to avoid corrupting data, but allow
  // multiple calls in one runtime. User is responsible for preserving the
  // output file from an earlier call before a later call overwrites it.
  static std::mutex writeDebugInfoMutex;
  std::lock_guard<std::mutex> lock(writeDebugInfoMutex);
  LOG(ERROR) << logPrefix() << "ProcessGroupECCL preparing to dump debug info.";
  if (ecclTraceBufferSize_ > 0) {
    // We dump eccl trace into local disk by default and users can register
    // their customized writer by inheriting `DebugInfoWriter` via
    // `registerDebugInfoWriter`.
    // auto ecclTrace = dump_eccl_trace();
    // c10d::DebugInfoWriter& writer =
    // c10d::DebugInfoWriter::getWriter(globalRank()); writer.write(ecclTrace);
    return true;
  }
  return false;
}

void ProcessGroupECCL::terminateProcess(std::string errMsg) {
  // Logging with `FATAL`, after errMsg printed, it calls `std::abort()`
  // to terminate the program execution.
  LOG(FATAL) << logPrefix() << errMsg;
}

int computeDeltaMS(std::chrono::time_point<std::chrono::steady_clock> start,
                   std::chrono::time_point<std::chrono::steady_clock> end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

void ProcessGroupECCL::heartbeatMonitor() {
  torch_gcu::util::setThreadName("TorchDistHeBeMo");
  DIST_TRACE_START(heartbeatMonitor);
  uint64_t heartBeatCounter = 0ULL;
  std::string errorMsg;
  std::string exitMsg;
  bool checkTimeoutSignal = (dumpOnTimeout_ && uid_ == 0);
  int monitorPollInterval = checkTimeoutSignal ? coordCheckIntervalMilSec_
                                               : heartbeatTimeoutInSec_ * 1000;
  auto lastTimePollStore = std::chrono::steady_clock::now();
  auto lastTimeHeartBeatCheck = std::chrono::steady_clock::now();
  c10::optional<DumpPipe> dumpPipe = c10::nullopt;
  if (uid_ == 0) {
    // DumpPipe is one per-trainer process, and its convenient to name them
    // after 'global' ranks in the system, So we assume processgroup (uid)==0 is
    // the global PG and has globally unique rank ids across trainers.
    dumpPipe.emplace(rank_);
  }
  while (true) {
    // This won't have any lock since this lock is only used here.
    // Please be aware that mutex `monitorMutex_` should not be used
    // somewhere else to avoid the deadlock.
    std::unique_lock<std::mutex> lock(monitorMutex_);
    if (monitorWakeUpCV_.wait_for(
            lock, std::chrono::milliseconds(monitorPollInterval),
            [&] { return terminateHeartbeatMonitorThread_.load(); })) {
      // For the normal complete or user interception, monitorWakeUpCV_
      // will get notified, we early return and exit heartbeatMonitor.
      return;
    }
    auto currentTime = std::chrono::steady_clock::now();

    // We put extra functionality in the thread for the default PG (aka, uid_=0)
    // because the signal is same across different PGs. We only need to run
    // once per process to avoid duplicate things performed in too many separate
    // threads. For example, we check a global flag on the TCPStore periodically
    // to see if any PG on any rank observed a timeout and signaled peers to
    // dump debugging info, and we avoid hammering the TCPStore from all PGs on
    // the same rank.
    if (checkTimeoutSignal) {
      // There are two scenarios where monitor thread will dump on timeout:
      // 1. The local rank is the first to observe a timeout.shouldDump_ will be
      // set to true.
      // 2. other ranks detected the timeout and signal the local rank to dump
      // In addition, monitor threads will dump if watchdog threads has no
      // heartbeat or dumpPipe is not empty.
      if (shouldDump_.load()) {
        errorMsg =
            c10::str(logPrefix(),
                     "Received a timeout signal from this local rank and will ",
                     "start to dump the debug info. ",
                     "Last enqueued ECCL work: ", lastEnqueuedSeq_,
                     ", last completed ECCL work: ", lastCompletedSeq_, ".");
        exitMsg = c10::str(
            "ProcessGroupECCL's watchdog detected a collective timeout from "
            "the local rank. ",
            "This is most likely caused by incorrect usages of collectives, "
            "e.g., wrong ",
            "sizes used across ranks, the order of collectives is not same for "
            "all ranks ",
            "or the scheduled collective, for some reason, didn't run. "
            "Additionally, ",
            "this can be caused by GIL deadlock or other reasons such as "
            "network errors or ",
            "bugs in the communications library (e.g. ECCL), etc. We tried our "
            "best to ",
            "dump the debug info into the storage to help you debug the "
            "issue.");
        break;
      }
      // We poll store to see if some ranks have flagged a timeout when
      // we haven't polled for `heartbeat_timeout` seconds and there haven't
      // any work added or removed for `watchdog_timeout` seconds.
      if (computeDeltaMS(lastWorkListUpdateTime_, currentTime) >=
              kWatchdogThreadSleepMillis &&
          computeDeltaMS(lastTimePollStore, currentTime) >=
              coordCheckIntervalMilSec_) {
        lastTimePollStore = currentTime;
        if (globalStore_->check({std::string(TIMEOUT_DUMP)})) {
          errorMsg =
              c10::str(logPrefix(),
                       "Received a global timeout from another rank and will ",
                       "start to dump the debug info. ",
                       "Last enqueued ECCL work: ", lastEnqueuedSeq_,
                       ", last completed ECCL work: ", lastCompletedSeq_, ".");
          exitMsg = c10::str(
              "ProcessGroupECCL's watchdog detected a collective timeout on "
              "some other rank and notified current rank. ",
              "This is most likely caused by incorrect usages of collectives, "
              "e.g., wrong ",
              "sizes used across ranks, the order of collectives is not same "
              "for all ranks ",
              "or the scheduled collective, for some reason, didn't run. "
              "Additionally, ",
              "this can be caused by GIL deadlock or other reasons such as "
              "network errors or ",
              "bugs in the communications library (e.g. ECCL), etc. We tried "
              "our best to ",
              "dump the debug info into the storage to help you debug the "
              "issue.");
          break;
        }
      }
    }

    if (computeDeltaMS(lastTimeHeartBeatCheck, currentTime) >=
        heartbeatTimeoutInSec_ * 1000) {
      // Check the heart beat of watchdog thread.
      lastTimeHeartBeatCheck = currentTime;
      auto heartbeat = heartbeat_.load();
      if (heartbeat != heartBeatCounter) {
        heartBeatCounter = heartbeat;
      } else {
        // No heartbeat increase detected and timeout.
        errorMsg = c10::str(logPrefix(),
                            "Heartbeat monitor timed out! Process will be "
                            "terminated after dumping debug info.",
                            " workMetaList_.size()=", workMetaList_.size());
        exitMsg = c10::str(
            "ProcessGroupECCL's watchdog got stuck for ",
            heartbeatTimeoutInSec_,
            " seconds without making progress in monitoring enqueued "
            "collectives. ",
            "This typically indicates a ECCL/GCU API hang blocking the "
            "watchdog, ",
            "and could be triggered by another thread holding the GIL inside "
            "a ",
            "GCU api, or other deadlock-prone behaviors.",
            "If you suspect the watchdog is not actually stuck and a longer "
            "timeout would help, ",
            "you can either increase the timeout "
            "(TORCH_ECCL_HEARTBEAT_TIMEOUT_SEC) to a larger value "
            "or disable the heartbeat monitor (TORCH_ECCL_ENABLE_MONITORING=0)."
            "If either of aforementioned helps, feel free to file an issue to "
            "PyTorch about the short timeout "
            "or false positive abort; otherwise, please attempt to debug the "
            "hang. "
            "workMetaList_.size() = ",
            workMetaList_.size(), "");
        break;
      }
    }
    // process a request to dump the trace. only PG uid 0 will respond to dump
    // requests, but this is fine since all PG's feed into the same flight
    // recorder and dump. After dump, the training should continue.
    if (dumpPipe.has_value() && dumpPipe->shouldDump()) {
      // best effort dump, not waiting for the dump here
      std::future<bool> fut = std::async(
          std::launch::async, [this]() { return this->dumpDebuggingInfo(); });
    }
  }
  LOG(ERROR) << errorMsg;

  auto& cpp_dumper = get_cpp_trace_dumper();
  if (cpp_dumper.has_value()) {
    LOG(INFO) << "Dumping c++ stacktraces: " << cpp_dumper.value()();
  }

  // Store debug info to storage if no other thread does it. (By default to
  // local disk)
  std::future<bool> asyncDebugDump = std::async(
      std::launch::async, [this]() { return this->dumpDebuggingInfo(); });

  // wait for the dump until timeout
  waitForFutureOrTimeout(asyncDebugDump,
                         std::chrono::milliseconds(waitTimeoutDumpInMilSec_),
                         "Flight recorder dump in heartbeatMonitor");

  if (get_gil_checker() != nullptr) {
    auto fut = launchAsyncGilCheck();
    auto kGilCheckTimeout = std::chrono::milliseconds(300);
    auto futStatus = fut.wait_for(kGilCheckTimeout);
    if (futStatus != std::future_status::ready) {
      TORCH_CHECK(futStatus != std::future_status::deferred,
                  "Expected the future to have been launched eagerly.");
      LOG(ERROR) << "Could not acquire GIL within 300 ms on exit, possible GIL "
                    "induced hang";
    }
    LOG(INFO) << "Could acquire GIL on exit";
  } else {
    LOG(INFO)
        << "GIL checker was not registered, perhaps this is a no-python build?";
  }

  // There are two possible cases for the watchdog thread exit:
  // Case one: desync report runs quickly, and it follows the step:
  // collective timeout -> desync -> exception handling -> destructors
  // -> set terminateHeartbeatMonitorThread_ -> notify monitorWakeUpCV_.
  // So the code either early returns above or will skip the sleep below.
  // Case two: desync might be slow or get stuck. Or we get stuck in
  // destructors, we will sleep for some time before calling std::abort() to
  // kill the whole process.
  if ((terminateProcessGroup_.load() || collectiveDebugInfoMode_.load()) &&
      !terminateHeartbeatMonitorThread_.load()) {
    // Leave another two mins for desync report generation or process group
    // destroy.
    std::this_thread::sleep_for(std::chrono::seconds(heartbeatTimeoutInSec_));
  }

  // At this point, we either already sleep for another `heartbeatTimeoutInSec_`
  // or the thread has finished. Because we don't want to block the monitor
  // thread, so We mark the thread detach and the dump of debug info becomes
  // "best effort". If the process exit normally, marking it detach also makes
  // sense because we don't really care about dumping the debug info.

  // We already log completion inside the thread, so it may not be necessary to
  // check the return value here.  We mainly use a future so we can exit early
  // if done.

  if (!terminateHeartbeatMonitorThread_.load()) {
    // Create a error message reported from MonitorThread, so
    // we throw exception and make the whole process to be killed.
    // TODO(fduwjj): After having a hang debug wiki, we need to update the wiki
    // url here.
    const auto finalExitMsg = c10::str(logPrefix(), exitMsg);
    if (monitorThreadEnabled_.load()) {
      terminateProcess(finalExitMsg);
    } else {
      LOG(ERROR) << "PGECCL Monitor Thread is disabled, but would have killed "
                    "this job:\n"
                 << finalExitMsg;
    }
  }
  DIST_TRACE_END(heartbeatMonitor);
}

void ProcessGroupECCL::ecclCommWatchdog() {
  torch_gcu::util::setThreadName("TorchDistEcclWD");
  DIST_TRACE_START(ecclCommWatchdog);
  try {
    VLOG(2) << logPrefix() << "Process group watchdog thread started!";
    ecclHeartbeatMonitorThread_ =
        std::thread(&ProcessGroupECCL::heartbeatMonitor, this);
    watchdogHandler();
    VLOG(2) << logPrefix()
            << "Process group watchdog thread terminated normally";
  } catch (std::exception& e) {
    if (std::string(e.what()).find("driver shutting down") !=
        std::string::npos) {
      LOG(INFO) << logPrefix()
                << "main process destroyed gcu before watchdog loop exited, "
                   "terminating watchdog."
                << " (Watchdog caught exception: " << e.what();

    } else {
      // Append error message reported from watchdogHandler
      const auto exitMsg =
          c10::str(logPrefix(),
                   "Process group watchdog thread terminated with exception: ",
                   e.what());
      LOG(ERROR) << exitMsg;
      // TODO(whc) clean up the rethrow - why is it stored in a class var and
      // rethrown?
      watchDogException_ =
          std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
      std::rethrow_exception(watchDogException_);
    }
  } catch (...) {
    const auto exitMsg = c10::str(
        logPrefix(),
        "Process group watchdog thread terminated with exception: unknown");
    LOG(ERROR) << exitMsg;
    watchDogException_ =
        std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
    std::rethrow_exception(watchDogException_);
  }
  DIST_TRACE_END(ecclCommWatchdog);
}

void ProcessGroupECCL::logWorkStart(WorkECCL& work) {
  if (work.startTraceUpdated_) return;

  if (terminateProcessGroup_.load() || storeError_) return;

  work.startTraceUpdated_ = true;
  storeError_ = !c10d::traceUpdate(store_, traceKeyStart_, work.seq_,
                                   opTypeToString(work.opType_));
}

void ProcessGroupECCL::logWorkEnd(WorkECCL& work) {
  if (terminateProcessGroup_.load() || storeError_) return;

  // In case the start of the work hasn't been logged
  if (!work.startTraceUpdated_) {
    logWorkStart(work);
  }

  storeError_ = !c10d::traceUpdate(store_, traceKeyEnd_, work.seq_,
                                   opTypeToString(work.opType_));
}

std::string ProcessGroupECCL::getECCLWatchdogDebugInfo() {
  return retrieveDesyncReport(store_, "ECCL", rank_, size_);
}

std::string ProcessGroupECCL::createLogPrefix() const {
  return c10::str("[PG ", uid_, " Rank ", rank_, "] ");
}

const std::string& ProcessGroupECCL::logPrefix() const { return logPrefix_; }

const int& ProcessGroupECCL::globalRank() const {
  static int globalRank = rank_;
  return globalRank;
}

const std::vector<uint64_t>& ProcessGroupECCL::groupRanks() const {
  if (options_->global_ranks_in_group.empty() && uid_ == 0) {
    static std::vector<uint64_t> globalRanks(size_);
    std::iota(globalRanks.begin(), globalRanks.end(), 0);
    return globalRanks;
  }
  return options_->global_ranks_in_group;
}

void ProcessGroupECCL::watchdogHandler() {
  DIST_TRACE_START(watchdogHandler);
  bool done = false;
  lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
  std::list<ProcessGroupECCL::WorkECCL> completedWorkList;

  while (!done || !terminateProcessGroup_.load()) {
    std::unique_lock<std::mutex> lock(workMetaListMutex_);
    // We busy-poll the work vector every kWatchdogThreadSleepMillis
    // milliseconds as long as the atomic is True.
    workMetaListCV_.wait_for(
        lock, std::chrono::milliseconds(kWatchdogThreadSleepMillis),
        [&]() -> bool { return terminateProcessGroup_.load(); });
    // Bump up heart beat by one.
    heartbeat_++;

// Some versions of GLOG support less-spammy version of LOG_EVERY_MS
// in which case we don't want to spam the logs.
#ifdef LOG_EVERY_MS
    // Log the progress of this PG periodically
    C10_LOG_EVERY_MS(INFO, kWorkStatusUpdatePeriodMs)
        << c10::str(logPrefix(), "ECCL c10d::Work update periodically: ",
                    "last enqueued ECCL work: ", lastEnqueuedSeq_,
                    ", last completed ECCL work: ", lastCompletedSeq_, ".");
#endif

    for (auto it = workMetaList_.begin(); it != workMetaList_.end();
         /* no increment */) {
      auto& work = *it;
      // When terminateProcessGroup_ is true, communicators have already been
      // aborted, So cannot check exception based on them. But watchdog needs to
      // finish the check for the works that have already been enqueued to
      // workMetaList_
      if (!terminateProcessGroup_.load()) {
        work.checkAndSetException();
      }
      bool timedOut = work.checkTimeout();

      // If work hits an exception (either an error or timeout)
      if (work.exception()) {
        if (SHOULD_CLEAN_UP(asyncErrorHandling_)) {
          // Abort work and corresponding communicators
          work.abort();
          // PG level abort, which would abort all other communicators on this
          // rank
          abort();
        }

        // Report desync state in case of timeout
        if (timedOut) {
          LOG(ERROR) << c10::str(
              logPrefix(), "Timeout at ECCL work: ", work.seq_,
              ", last enqueued ECCL work: ", lastEnqueuedSeq_,
              ", last completed ECCL work: ", lastCompletedSeq_, ".");
          try {
            if (desyncDebug_ || dumpOnTimeout_) {
              // Set shutdown mode, so the heartbeat monitor thread will not
              // abort process immediately.
              collectiveDebugInfoMode_.store(true);
              std::vector<uint8_t> vec(1);
              globalStore_->set(std::string(TIMEOUT_DUMP), vec);
            }

            if (dumpOnTimeout_) {
              // signal the monitor thread to start dumping
              shouldDump_.store(true);
              // This sleep is used to give time for dumping before throwing
              // exception
              std::this_thread::sleep_for(
                  std::chrono::seconds(heartbeatTimeoutInSec_));
            }

            if (desyncDebug_) {
              auto desyncMsg = getECCLWatchdogDebugInfo();
              LOG(ERROR) << logPrefix() << desyncMsg;
            }
          } catch (const std::exception& e) {
            LOG(ERROR) << logPrefix()
                       << "Failed to retrieve TORCH_ECCL_DESYNC_DEBUG report. "
                       << " Please file an issue. Error: " << e.what();
          } catch (...) {
            LOG(ERROR) << logPrefix()
                       << "Failed to rerieve TORCH_ECCL_DESYNC_DEBUG report "
                          "with unknown error."
                       << " Please file an issue.";
          }
        }
        // Throw exception
        work.handleException(asyncErrorHandling_);
      }

      // c10d::Work status logging for desync debug
      if (desyncDebug_) {
        if (work.isStarted()) {
          logWorkStart(work);
        }
        if (work.isCompleted()) {
          logWorkEnd(work);
        }
      }

      // Clean up completed work
      if (work.isCompleted()) {
        lastCompletedSeq_ = work.seq_;
        // ECCLTraceBuffer::get()->retire_id(work.trace_id_, true);
        if (onCompletionHook_) {
          // Move c10d::Work object to completedWorkList_ to be consumed by the
          // hook thread
          {
            const std::lock_guard<std::mutex> lock(completedWorkListMutex_);
            completedWorkList_.splice(completedWorkList_.end(), workMetaList_,
                                      it++);
          }
          completedWorkListCV_.notify_one();
        } else {
          it = workMetaList_.erase(it);
          lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
        }
        torch_gcu::GCUGraph::dec_pending_event_queries();
      } else {
        // Increment the iterator if the current WorkECCL object is not
        // completed.
        ++it;
      }
      // Increment heartbeat after each work processed,
      // in case processing is slowed down (but not hung) by gcuapi contention
      heartbeat_++;
    }
    done = workMetaList_.empty();
  }
  DIST_TRACE_END(watchdogHandler);
}

void ProcessGroupECCL::runHookLoop() {
  torch_gcu::util::setThreadName("TorchDistRHL");
  bool done = false;
  while (!done || !terminateProcessGroup_.load()) {
    std::unique_lock<std::mutex> lock(completedWorkListMutex_);
    // We busy-poll the work vector every kWatchdogThreadSleepMillis
    // milliseconds as long as the atomic is True.
    completedWorkListCV_.wait_for(
        lock, std::chrono::milliseconds(kWatchdogThreadSleepMillis),
        [&]() -> bool {
          return !completedWorkList_.empty() || terminateProcessGroup_.load();
        });
    try {
      for (auto it = completedWorkList_.begin(); it != completedWorkList_.end();
           /* no increment */) {
        const WorkECCL& work = *it;
        // Hook might grab GIL, unlock first to prevent deadlock
        lock.unlock();
        auto timeFinished = std::chrono::steady_clock::now();
        auto timeStarted =
            timeFinished +
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                work.workStartTime_ - std::chrono::steady_clock::now());
        onCompletionHook_(std::make_shared<c10d::WorkInfo>(
            work.retrieveOpType(),     // c10d::OpType
            work.getSequencenumber(),  // seq
            timeStarted,               // timeStarted
            timeFinished,              // timeFinished
            std::chrono::duration<float, std::milli>(
                work.getDuration())  // activeDuration
            ));
        lock.lock();
        it = completedWorkList_.erase(it);
      }
    } catch (std::exception& e) {
      if (std::string(e.what()).find("driver shutting down") !=
          std::string::npos) {
        LOG(INFO) << logPrefix()
                  << "main process destroyed gcu before runHookLoop exited, "
                     "terminating runHookLoop."
                  << " (runHookLoop caught exception: " << e.what();

      } else {
        // PythonOnCompletionHook has already extracted Python exception message
        // and wrapped it with a cpp one. So we no longer need to acquire GIL
        // here.
        const auto errorStr = c10::str(
            "Caught exception on rank ", rank_,
            " while running onCompletion hook for ProcessGroupECCL: ", e.what(),
            ". Aborting all communicators.");

        // No need to call abort() on WorkECCL here as that collective has
        // already finished successfully at this point. We just need to abort
        // the process Abort all ECCL Communicators on this ProcessGroupECCL
        // instance.
        abort(errorStr);
      }
    }

    // Lock is still acquired at this point
    done = completedWorkList_.empty();
  }
}

std::exception_ptr ProcessGroupECCL::WorkECCL::checkForECCLErrors() {
  return checkForECCLErrorsInternal(ecclComm_);
}

std::exception_ptr ProcessGroupECCL::checkForECCLErrors(
    std::shared_ptr<ECCLComm>& ecclComm) {
  return checkForECCLErrorsInternal(ecclComm);
}

std::exception_ptr ProcessGroupECCL::checkForECCLErrorsInternal(
    std::shared_ptr<ECCLComm>& ecclComm) {
  // Prioritize commFailureReason over checkForEcclError() result if
  // commFailureReason is set.
  auto commFailureReason = ecclComm->getEcclCommFailureReason();
  if (commFailureReason != c10::nullopt) {
    return std::make_exception_ptr(C10_BUILD_ERROR(
        DistBackendError,
        c10::str(
            "ECCL communicator encountered error set by ProcessGroupECCL: ",
            *commFailureReason)));
  }
  ecclResult_t ecclAsyncErr = ecclComm->checkForEcclError();
  // When nonblocking mode is enabled by TORCH_ECCL_USE_COMM_NONBLOCKING,
  // ecclInProgress could be returned when there are pending ECCL calls.
  // In this case, no exception should be thrown
#ifdef ECCL_HAS_COMM_NONBLOCKING
  // ecclInProgress is defined only if ECCL_HAS_COMM_NONBLOCKING is defined
  if (ecclAsyncErr != ecclSuccess && ecclAsyncErr != ecclInProgress) {
#else
  if (ecclAsyncErr != ecclSuccess) {
#endif
    return std::make_exception_ptr(C10_BUILD_ERROR(
        DistBackendError,
        "ECCL error: " + ecclGetErrorWithVersion(ecclAsyncErr)));
  }

  return nullptr;
}

void ProcessGroupECCL::broadcastUniqueECCLID(ecclUniqueId* ecclID,
                                             bool isSingleP2POp,
                                             const std::string& p2pKey,
                                             int p2pRank) {
  // For collective operations:
  // For every ECCL communicator that we create we need to broadcast
  // a unique ID from rank 0 to all other ranks. This broadcast is
  // done by rank 0 setting a key in the store and all other ranks
  // retrieving the contents of that key. A single process group
  // may create multiple ECCL communicators, so we use a sequence
  // number to differentiate between them.
  // For single point-to-point operations:
  // The sequence number will only be increased on 2 out of all the
  // processes in a Process Group. So all following collective
  // operations will see different sequence numbers which will cause
  // runtime errors. To avoid that, use the src:target pair instead
  // of sequence number for p2p communications.

  std::string storeKey;
  if (!isSingleP2POp) {
    storeKey = std::to_string(ecclCommCounter_++);
  } else {
    storeKey = p2pKey;
  }

  if (rank_ == 0 || (isSingleP2POp && p2pRank == 0)) {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(ecclID),
        reinterpret_cast<uint8_t*>(ecclID) + ECCL_UNIQUE_ID_BYTES);
    store_->set(storeKey, vec);
    storeKeys_.emplace_back(storeKey);
  } else {
    try {
      auto vec = store_->get(storeKey);
      TORCH_CHECK_WITH(DistBackendError, vec.size() == ECCL_UNIQUE_ID_BYTES,
                       "Invalid size for ecclUniqueId");
      std::memcpy(ecclID, vec.data(), vec.size());
    } catch (const std::exception& e) {
      std::string exceptionMsg = c10::str(
          "[", rank_,
          "] is setting up ECCL communicator and "
          "retrieving ecclUniqueId from [0] via c10d key-value store by key '",
          storeKey, "', but store->get('", storeKey, "') got error: ");
      C10_THROW_ERROR(DistBackendError,
                      exceptionMsg + e.what() +
                          ". This may indicate a possible application crash on "
                          "rank 0 or a network set up issue.");
    } catch (...) {
      C10_THROW_ERROR(DistBackendError,
                      c10::str("Unknown exception while [", rank_,
                               "] is setting up ECCL communicator and "
                               "retrieving ecclUniqueId from [0] via c10d "
                               "key-value store by key '",
                               storeKey, "'",
                               ". This may indicate a possible application "
                               "crash on rank 0 or a network set up issue."));
    }
  }
}

void ProcessGroupECCL::destroyECCLComms(const std::string& devECCLCommMapKey) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (devECCLCommMap_.find(devECCLCommMapKey) == devECCLCommMap_.end()) {
    TORCH_INTERNAL_ASSERT(false, "Expected to find key ", devECCLCommMapKey,
                          " in ECCL communicator map.");
  }
  std::shared_ptr<ECCLComm>& ecclComm = devECCLCommMap_[devECCLCommMapKey];
  // ecclCommDestroy(comm->getEcclComm()) results in segfault when PG is being
  // destroyed, so using ecclCommAbort here.
  ecclComm->ecclCommAbort();
  // Remove communicators from the cache.
  devECCLCommMap_.erase(devECCLCommMapKey);
  // Clear used device indices.
  usedDeviceIdxs_.clear();

  ecclCommDevIdxMapMutex.lock();
  ecclCommDevIdxMap.erase(ecclComm);
  ecclCommDevIdxMapMutex.unlock();
}

std::shared_ptr<ECCLComm> ProcessGroupECCL::getECCLComm(
    const std::string& deviceKey, at::Device& device, c10d::OpType opType,
    int p2pRank, bool isSendRecvSelf) {
  // Sanity check
  if (deviceKey.empty()) {
    C10_THROW_ERROR(DistBackendError,
                    "Not able to create/get the ECCL Communicator since "
                    "the GCU devices are not known");
  }
  if (bound_device_id_) {
    if (*bound_device_id_ != device) {
      LOG(ERROR) << logPrefix() << "Tensor found on device " << device
                 << " but backend constrained to " << *bound_device_id_;
      C10_THROW_ERROR(DistBackendError,
                      "Attempt to perform collective on tensor not on device "
                      "passed to init_process_group");
    }
  }

  usedDeviceIdxs_.insert(device.index());

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (devECCLCommMap_.find(deviceKey) != devECCLCommMap_.end()) {
      // Reuse the cached communicator if there is one.
      return devECCLCommMap_[deviceKey];
    }
  }

  // ECCL communicator not cached, create a new entry
  std::shared_ptr<ECCLComm> ecclComm;

  // Create the unique ECCL ID and broadcast it
  ecclUniqueId ecclID;

  // For batch_isend_irecv, ecclGroupStart() would be called upfront
  bool batchP2P = ecclActiveGroupCounter_ > 0;
  bool singleP2POp = isP2POp(opType, batchP2P);
  // For point-to-point communication, lower rank of the two will get unique id.
  if (rank_ == 0 || (singleP2POp && p2pRank == 0)) {
    C10D_ECCL_CHECK(ecclGetUniqueId(&ecclID), c10::nullopt);
  }

  // For point-to-point communication on the same process, don't need broadcast.
  if (!isSendRecvSelf) {
    // Broadcast so that each process can have a unique ECCL ID
    broadcastUniqueECCLID(&ecclID, singleP2POp, deviceKey, p2pRank);
  }

  ecclUniqueId_.assign(
      reinterpret_cast<uint8_t*>(&ecclID),
      reinterpret_cast<uint8_t*>(&ecclID) + ECCL_UNIQUE_ID_BYTES);
  torch_gcu::OptionalGCUGuard gcuGuard;

  // [Group Start/End Note] This is used to ensure that eccl communicator will
  // be created before communication primitives are called. Let's look at this
  // example: Using the batch_isend_irecv to send a tensor to a target process.
  // On the sender side, the corresponding underlying ECCL calls will look like
  //   ecclGroupStart() // This is in batch_isend_irecv
  //   ecclGroupStart() // This is [Note 1]
  //   ecclCommInitRank() // Inside ECCLComm::create
  //   ecclSend()
  //   ecclGroupEnd() // This is [Note 2]
  //   ecclGroupEnd() // This is in batch_isend_irecv
  // With this pattern, the eccl communicator will be created in the last
  // ecclGroupEnd which means when ecclSend is processed, the passed
  // communicator argument is NULL which will lead to runtime error. So we need
  // to "close" all active eccl groups to ensure eccl communicator is actually
  // created before encountering any communication calls. This is why we need
  // the following for loop.
  for (const auto i : c10::irange(ecclActiveGroupCounter_)) {
    (void)i;
    // comms have not been initiated yet, so can only check in blocking-way
    C10D_ECCL_CHECK(ecclGroupEnd(), c10::nullopt);
  }

  // [Note 1] Create the ECCL communicators for each GCU
  C10D_ECCL_CHECK(ecclGroupStart(), c10::nullopt);

  // GCU world size and GCU rank
  int numRanks, rank;

  if (!singleP2POp) {
    // Collective, all-to-all, or batch P2P
    numRanks = getSize();
    rank = getRank();
  } else if (isSendRecvSelf) {
    // Same process send and recv.
    numRanks = 1;
    rank = 0;
  } else {
    // For single point-to-point operation, there are only 2 processes
    // involved so the GCU rank is either 0 or 1.
    numRanks = 2;
    rank = p2pRank;
  }
  // Get the device index
  auto deviceIndex = device.index();
  gcuGuard.set_index(deviceIndex);
#ifdef ECCL_HAS_COMM_SPLIT
  if (options_->split_from) {
    TORCH_CHECK(options_->split_color != 0,
                "Must specify a non-zero color when splitting");
    // Find a valid, healthy communicator to split from if possible.
    std::lock_guard<std::mutex> lock(options_->split_from->mutex_);
    auto& other_comms = options_->split_from->devECCLCommMap_;
    auto dit = other_comms.find(deviceKey);
    if (dit != other_comms.end()) {
      auto& parentComm = dit->second;
      if (parentComm != nullptr && !parentComm->isAborted()) {
        ecclComm = ECCLComm::split(parentComm.get(), options_->split_color,
                                   rank, options_->config);
      }
    }
  }
#endif

  // To simplify conditioonal nesting, just create the ecclComms[i]
  // entry if it hasn't been yet rather than untangling the
  // conditions that might have resulted in a split above.
  if (!ecclComm) {
#ifdef ECCL_HAS_COMM_NONBLOCKING
    ecclComm = ECCLComm::create(numRanks, rank, ecclID, options_->config);
#else
    ecclComm = ECCLComm::create(numRanks, rank, ecclID);
#endif
  }

  // Creates the ECCL streams
  auto streamVal =
      torch_gcu::getStreamFromPool(options_->is_high_priority_stream);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    inInitializationCommMap_.emplace(deviceKey, ecclComm);
  }

  // [Note 2 ]
#ifndef ECCL_HAS_COMM_NONBLOCKING
  C10D_ECCL_CHECK(ecclGroupEnd(), c10::nullopt);
#else
  if (eccl_use_nonblocking()) {
    // If we use nonblocking mode, allow communicators to be
    // uninitialized/ecclInProgress until the first communication
    C10D_ECCL_CHECK_NONBLOCKING(ecclGroupEnd(), c10::nullopt);
  } else {
    C10D_ECCL_CHECK(ecclGroupEnd(), c10::nullopt);
  }
#endif

  LOG(INFO) << logPrefix() << "ProcessGroupECCL created ecclComm_ "
            << ecclComm->ecclComm_ << " on GCU device: " << deviceIndex;

  // At this point ECCL should have been initialized, hence we can accurately
  // get the env value even if ECCL sets it by reading from eccl.conf file
  LOG(INFO) << logPrefix()
            << "ECCL_DEBUG: " << c10d::getCvarString({"ECCL_DEBUG"}, "N/A");

  // See [Group Start/End Note]
  for (const auto i : c10::irange(ecclActiveGroupCounter_)) {
    (void)i;
    C10D_ECCL_CHECK(ecclGroupStart(), c10::nullopt);
  }

  ecclStreams_.emplace(deviceKey, std::move(streamVal));

  // Note(torch_gcu):reuse GCUEvent will increase record time consumption, so
  // each time a new GCUEvent is created
  // Note: these events are created with the
  // (default) topsEventDisableTiming flag This flag provides the best
  // performance when used with topsStreamWaitEvent() and topsEventQuery().
  // Since we here don't measure the performance using topsEvent, this should be
  // set.
  // TODO(kwen2501): is ecclEvents_ used anywhere else?
  // ecclEvents_.emplace(deviceKey,
  // torch_gcu::GCUEvent(topsEventDisableTiming));

  // Record the communicators based on ecclUniqueId.
  ecclIdToCommMap_.emplace(buildEcclUniqueIdStr(ecclID), ecclComm);

  // Move the ECCL resource to cache
  auto it = inInitializationCommMap_.find(deviceKey);
  // A previous thread could've already removed devicesKey from
  // inInitializationCommMap_ and added it to devECCLCommMap_
  if (it != inInitializationCommMap_.end()) {
    devECCLCommMap_.emplace(deviceKey, std::move(it->second));
    inInitializationCommMap_.erase(deviceKey);

    // Now ecclComms are fully initialized.
    // Register all active GCU memory segments in cache allocator to
    // the new ECCL communicators
    if (useTensorRegisterAllocatorHook_) {
      auto snapshot = torch_gcu::GCUCachingAllocator::snapshot();
      // Register the segment to a new ECCL communicator if on the same device
      for (const auto& segmentInfo : snapshot.segments) {
        TORCH_INTERNAL_ASSERT(
            segmentInfo.device == device.index(),
            "Mismatch between GCU memory segment device and current device");
        // ecclComm->registerSegment(reinterpret_cast<void*>(segmentInfo.address),
        //                           segmentInfo.total_size);
      }

      // Record the mapping between ecclComm and device index so that later
      // register hook can register a newly allocated segment to communicators
      // on the same device.
      // NOTE: we need remove the communicator from this map when it is
      // destroyed, otherwise may register onto an invalid communicator.
      ecclCommDevIdxMapMutex.lock();
      ecclCommDevIdxMap.emplace(ecclComm, device.index());
      ecclCommDevIdxMapMutex.unlock();
    }
  }

  it = devECCLCommMap_.find(deviceKey);
  TORCH_INTERNAL_ASSERT(it != devECCLCommMap_.end(),
                        "Communicators not populated in cache!");

  return it->second;
}

uint64_t ProcessGroupECCL::getCommSplitCounter() const {
  uint64_t ret = 0;
  for (const auto& i : ecclIdToCommMap_) {
    auto& ecclComm = i.second;
    // ret += ecclComm->getCommSplitCounter();
  }
  return ret;
}

namespace {

// Check validity of tensor
void check_gcu_single_tensor(
    const at::Tensor& tensor,
    const bool p2p = false  // whether operation is a P2P operation
) {
  if (!torch_gcu::is_gcu(tensor) || tensor.is_sparse()) {
    C10_THROW_ERROR(ValueError, "Tensors must be GCU and dense");
  }
  // Skip the following requirements for P2P operations
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    if (p2p) {
      TORCH_WARN_ONCE(
          "Detected non-contiguous tensor in P2P operations. It is user "
          "responsibility to guarantee that source and destination tensors "
          "have "
          "the same contiguity format.");
    } else {
      C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
    }
  }
}

// Checks that all `tensors' have the same type and shape and reside on the same
// GCU.
// TODO: test_c10d_eccl.py should consider adding tests for the error conditions
// here, ie, that deliberately pass invalid tensors and check the right
// exception is thrown. The "Expected list of tensors on the same device"
// condition may be a challenge because the test would need to pass tensors on
// different devices in the same process.
int64_t check_gcu_tensors_same_device(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    C10_THROW_ERROR(ValueError, "Tensor list must be nonempty");
  }

  const auto& first = tensors.front();

  int64_t total_numel = 0;
  for (const auto& t : tensors) {
    if (!torch_gcu::is_gcu(t) || t.is_sparse()) {
      C10_THROW_ERROR(ValueError, "Tensors must be GCU and dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      C10_THROW_ERROR(TypeError, "Tensors must have identical type");
    }
    if (!t.is_non_overlapping_and_dense()) {
      C10_THROW_ERROR(ValueError, "Tensors must be non-overlapping and dense");
    }
    // If we're in this function, the user called a _coalesced collective
    // on a set of tensors with potentially different sizes and strides.
    // Therefore, we don't check for matching sizes and strides,
    // but we do double-check tensors are on the same device.
    TORCH_CHECK_WITH(ValueError, t.get_device() == tensors[0].get_device(),
                     "Expected list of tensors on the same device");
    total_numel += t.numel();
  }

  return total_numel;
}

bool check_same_size(const std::vector<at::Tensor>& input_tensors) {
  for (const auto& input_tensor : input_tensors) {
    if (!input_tensors[0].is_same_size(input_tensor)) {
      return false;
    }
  }
  return true;
}

}  // namespace

c10::intrusive_ptr<ProcessGroupECCL::WorkECCL> ProcessGroupECCL::initWork(
    at::Device& device, int rank, c10d::OpType opType,
    const char* profilingTitle, const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs,  // TODO(kwen2501): necessary?
    bool record) {
  auto r = c10::make_intrusive<ProcessGroupECCL::WorkECCL>(
      device, rank, opType, seq_, profilingTitle,
      profilingTitle != nullptr ? c10::optional<std::vector<at::Tensor>>(inputs)
                                : c10::nullopt,
      desyncDebug_, enableTiming_.load(), dist_debug_level_);
  if (record) {
    // Ideally record every work that we enqueue, rather than every work we
    // create.
    // - at the time of this PR we do not currently enqueue every created work
    // - but it is unsafe to steal refs to start/end gcuevents from Works that
    //   may go out of scope before flight recorder has retired them,
    //   so we must ensure that any work that is initialized via initWork will
    //   be enqueued
    // - initially, moved record() into workEnqueue(), but found that makes it
    //   hard to get access to profilingTitle,
    //   inputs, and outputs for metadata recording, and we don't want to attach
    //   these objects to the c10d::Work because it has implications for keeping
    //   those tensors alive longer and adds overhead when copying c10d::Work
    //   objects between threads
    // r->trace_id_ = ECCLTraceBuffer::get()->record(
    //     uid_, seq_, op_id_, profilingTitle ? profilingTitle : "", inputs,
    //     outputs, r->ecclStartEvent_.get(), r->ecclEndEvent_.get());
  }
  return r;
}

// TODO(kwen2501): deprecate
std::vector<at::Tensor> ProcessGroupECCL::WorkECCL::result() {
  return *outputs_;
}

c10::intrusive_ptr<c10::ivalue::Future>
ProcessGroupECCL::WorkECCL::getFuture() {
  return future_;
}

float ProcessGroupECCL::WorkECCL::getDuration() const {
  TORCH_CHECK(timingEnabled_, "getDuration only works if timing was enabled");
  TORCH_CHECK(ecclStartEvent_,
              "getDuration only works if ecclStartEvents_ is populated, true "
              "if timing enabled");
  TORCH_CHECK(ecclEndEvent_,
              "getDuration only works if ecclEndEvents_ is populated, which "
              "should always be true");
  return ecclStartEvent_->elapsed_time(*ecclEndEvent_);
}

uint64_t ProcessGroupECCL::WorkECCL::getSequencenumber() const { return seq_; }

void ProcessGroupECCL::workEnqueue(
    c10::intrusive_ptr<ProcessGroupECCL::WorkECCL> work) {
  if (!terminateProcessGroup_.load()) {
    std::lock_guard<std::mutex> lock(workMetaListMutex_);
    // Avoid view tensors to be processed in cleanup thread.
    // View tensors' destruction invokes autograd_meta, which
    // needs to be destructed in user thread. Otherwise will
    // get deadlock. Here we enqueue work without outputs_.
    workMetaList_.emplace_back(*work);
    lastEnqueuedSeq_ = work->seq_;
    lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
  }
}

ProcessGroupECCL::Options::Options(bool is_high_priority_stream)
    : Backend::Options(ECCL_BACKEND_NAME, kProcessGroupECCLDefaultTimeout),
      is_high_priority_stream(is_high_priority_stream) {}

static constexpr int CoalActive = 0x01, CoalColl = 0x02, CoalP2P = 0x04;

void ProcessGroupECCL::startCoalescing() {
  coalescedDevices_.clear();
  coalescedComms_.clear();
  coalescing_state_ |= CoalActive;
  groupStart();
  // Other collective ops bump seq_ before creating a work. Thus, if coalesced
  // ops bump seq_ only after initing a work they will collide with (reuse) the
  // seq_ of the last non-coalesced collective.  Previously, seq_ was bumped
  // inside endCoalescing, but before initWork. Since we now record individual
  // ops from a coalesce group into the flight recorder, we want to have the
  // same seq_ for those ops and its 'endCoalescing' op. Hence we bump during
  // start, which has one minor downside- we burn a seq_ if someone ever does a
  // 'start' and 'end' coalescing region without doing an operation inbetween.
  seq_++;

  // Don't bump op_id_ here, because startCoalescing isn't a logical operation.
  // Bump it for each logical op inside the coalescing group.
}

// `optype` is for specifying a composite optype, such as ALLGATHER and
// REDUCE_SCATTER
c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::endCoalescing(
    c10d::OpType optype) {
  if (coalescedComms_.size() == 0) {
    // There is no actual work being coalesced, return here
    groupEnd();
    coalescing_state_ = 0;
    return nullptr;
  }

  // `coalescedComms_` should have same set of comms across collectives
  auto comm = coalescedComms_[0];
  // `coalescedDevices_` should have same set of devices across collectives
  auto device = coalescedDevices_[0];

  // `getKeyFromDevice` is how we get keys for both collectives and batch P2P
  const auto key = getKeyFromDevice(device);
  auto ecclStream = ecclStreams_.at(key);

  // Create c10d::Work object
  torch_gcu::CaptureStatus capture_status =
      torch_gcu::currentStreamCaptureStatusMayInitCtx();
  bool enqueue =
      (coalescing_state_) && capture_status == torch_gcu::CaptureStatus::None;
  auto work =
      initWork(device, rank_, optype, "eccl:coalesced", {}, {}, enqueue);
  work->ecclComm_ = comm;
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams_;
  work->opTimeout_ = options_->timeout;
  work->store_ = store_;

  // Record start before ecclGroupEnd
  if (work->timingEnabled_) {
    work->ecclStartEvent_->record(ecclStream);
  }

  if (eccl_use_nonblocking()) {
    groupEndNonblocking(comm);
  } else {
    groupEnd();
  }

  // Record end after ecclGroupEnd
  // TODO(eqy): is this still necessary if avoidRecordStreams_ is set?
  work->ecclEndEvent_->record(ecclStream);

  if (avoidRecordStreams_) {
    // other functions expect an initialized ptr if avoidRecordStreams_ is set
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>();
  }

  // Notify graphs before we check the capture status preemptively
  torch_gcu::GCUGraph::inc_pending_event_queries();

  if (enqueue) {
    workEnqueue(work);
  } else {
    torch_gcu::GCUGraph::dec_pending_event_queries();
  }

  coalescing_state_ = 0;
  if (torch_gcu::OpDebugConfig::GetInstance().enableSyncMode()) {
    StreamSynchronize(ecclStream);
  }
  return work;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::endCoalescing() {
  // Default c10d::OpType to COALESCED if not specified
  return endCoalescing(c10d::OpType::COALESCED);
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::collective(
    at::Tensor& input, at::Tensor& output, Fn fn, PreProcess pre,
    PostProcess post, c10d::OpType opType, const char* profilingTitle,
    bool avoidRecordStreams) {
  // Environment setting by the user may add onto collective call's option
  avoidRecordStreams |= avoidRecordStreams_;
  torch_gcu::CaptureStatus capture_status =
      torch_gcu::currentStreamCaptureStatusMayInitCtx();
  errorIfCapturingNonCapturableECCL(capture_status);

  // Bump collective counter
  seq_++;
  op_id_++;

  auto device = getDevice(input);
  const auto key = getKeyFromDevice(device);
  auto ecclComm = getECCLComm(key, device, opType);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalColl;
    coalescedDevices_.push_back(device);
    coalescedComms_.push_back(ecclComm);
  }

  // Used many times below, so we stash the unordered_map lookup
  auto ecclStream = ecclStreams_.at(key);
  auto sip_num = torch_gcu::util::GetEnvInt("TORCH_GCU_ECCL_SIP_NUM", 12);
  if (sip_num != 12) {
    C10_GCU_CHECK(topsStreamSetLaunchLimit(ecclStream, 2, sip_num));
  }

  // Note(torch_gcu):reuse GCUEvent will increase record time consumption, so
  // each time a new GCUEvent is created
  // First let ECCL streams wait for input tensors allocation streams
  // syncStream(device, ecclEvents_[key], ecclStream);
  torch_gcu::GCUEvent ecclEvent;
  syncStream(device, ecclEvent, ecclStream);

  std::vector<at::Tensor> inputs{input};
  std::vector<at::Tensor> outputs{output};

  bool enqueue =
      !coalescing_state_ && capture_status == torch_gcu::CaptureStatus::None;
  auto work =
      initWork(device, rank_, opType, profilingTitle, inputs, outputs, enqueue);

  // Store references to outputs to be used by WorkECCL::result and operator<<.
  work->outputs_ =
      std::make_shared<std::vector<at::Tensor>>(std::move(outputs));
  work->ecclEvent_ = std::move(ecclEvent);

  if (avoidRecordStreams) {
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>();
    work->stashed_for_allocator_safety_->push_back(input);
  }

  torch_gcu::OptionalGCUGuard gcuGuard;

  // Start event should only be recorded before the ecclGroupStart()
  if (work->timingEnabled_) {
    work->ecclStartEvent_->record(ecclStream);
  }

  pre(ecclStream, work);

  ecclComm_t comm = ecclComm->getEcclComm();

  // Both `inputs' and `outputs' are created on a worker stream and used in
  // different ecclStreams.  Hence, both must record the ecclStream to
  // prevent being freed before the collective finishes.
  //
  // We only record `inputs' here, and leave recording `outputs' to `fn' for
  // operations where `inputs' and `outputs' are not the same.
  //
  // See [Sync Streams].
  if (!avoidRecordStreams) {
    if (!input.is_sparse()) {
      torch_gcu::GCUCachingAllocator::recordStream(input.storage().data_ptr(),
                                                   ecclStream);
    } else {
      // for sparse input case record streams on both index and value
      // tensors
      torch_gcu::GCUCachingAllocator::recordStream(
          input.values().storage().data_ptr(), ecclStream);
      torch_gcu::GCUCachingAllocator::recordStream(
          input.indices().storage().data_ptr(), ecclStream);
    }
  }
#ifndef ECCL_HAS_COMM_NONBLOCKING
  C10D_ECCL_CHECK(fn(input, output, comm, ecclStream),
                  ecclComm->getEcclCommFailureReason());
#else
  C10D_ECCL_CHECK_TIMEOUT(fn(input, output, comm, ecclStream), comm,
                          ecclComm->getEcclCommFailureReason());
#endif

  post(ecclStream, work);

  // End event should only be recorded after the ecclGroupEnd()
  if (!coalescing_state_) {
    work->ecclEndEvent_->record(ecclStream);
  }
  work->ecclComm_ = ecclComm;

  {
    torch_gcu::GCUMultiStreamGuard streamGuard(ecclStream);
    std::vector<at::Device> devices{device};
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // Add a callback that runs profiling end callbacks. wrapCallback() in GCU
    // future blocks the stream this callback runs on the corresponding
    // ecclEndEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false allows us to skip synchronization in
          // ivalue::Future, but is only valid as long as the lambda doesn't use
          // the "Future" argument.
          /*uses_future=*/false);
    }
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Set appropriate work parameters.
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams;
  work->opTimeout_ = options_->timeout;
  work->store_ = store_;
  // Record size info for debug. We only record the size on the first device as
  // multi-device per process is deprecated
  work->numelIn_ = input.numel();
  work->numelOut_ = output.numel();

  // Notify graphs before we check the capture status preemptively
  torch_gcu::GCUGraph::inc_pending_event_queries();
  if (enqueue) {
    workEnqueue(work);
  } else {
    torch_gcu::GCUGraph::dec_pending_event_queries();
  }
  if (torch_gcu::OpDebugConfig::GetInstance().enableSyncMode()) {
    StreamSynchronize(ecclStream);
  }
  return work;
}

template <typename Fn>
c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::collectiveCoalesced(
    std::vector<at::Tensor>& inputs, std::vector<at::Tensor>& outputs, Fn fn,
    c10d::OpType opType, const char* profilingTitle, bool avoidRecordStreams) {
  // Environment setting by the user may add onto collective call's option
  avoidRecordStreams |= avoidRecordStreams_;
  torch_gcu::CaptureStatus capture_status =
      torch_gcu::currentStreamCaptureStatusMayInitCtx();
  errorIfCapturingNonCapturableECCL(capture_status);

  // Bump collective counter
  seq_++;
  // For coalescingManager collectives, there is no individual c++ call per
  // collective so there is no flight record and we increment seq_ and op_id_
  // together. Compare this to startCoalesing/endCoalescing flow where we
  // increment seq_ once per group and increment op_id_ once per individual
  // operation within the group
  op_id_++;

  // Currently, the API permits one scenario where inputs.size() and
  // outputs.size() are > 0.
  // 1. If the call was a _coalesced call, all inputs must be on the same
  // device.
  //    The group of eccl calls applies the collective separately to each input,
  //    but the group as a whole should be efficient, and might even execute as
  //    a single fused kernel.
  auto device = getDevice(inputs[0]);
  const auto key = getKeyFromDevice(device);
  auto ecclComm = getECCLComm(key, device, opType);

  // Used many times below, so we stash the unordered_map lookup
  auto ecclStream = ecclStreams_.at(key);
  auto sip_num = torch_gcu::util::GetEnvInt("TORCH_GCU_ECCL_SIP_NUM", 12);
  if (sip_num != 12) {
    C10_GCU_CHECK(topsStreamSetLaunchLimit(ecclStream, 2, sip_num));
  }

  // Note(torch_gcu):reuse GCUEvent will increase record time consumption, so
  // each time a new GCUEvent is created
  // First let ECCL streams wait for input tensors allocation streams
  torch_gcu::GCUEvent ecclEvent;
  syncStream(device, ecclEvent, ecclStream);

  auto work = initWork(device, rank_, opType, nullptr, inputs, outputs,
                       /*record=*/true);

  // Store references to outputs to be used by WorkECCL::result and operator<<.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);
  work->ecclEvent_ = std::move(ecclEvent);

  if (avoidRecordStreams) {
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>(inputs);
  }

  torch_gcu::OptionalGCUGuard gcuGuard;

  // Start event should only be recorded before the ecclGroupStart()
  if (work->timingEnabled_) {
    work->ecclStartEvent_->record(ecclStream);
  }

  ecclComm_t comm = ecclComm->getEcclComm();

// TODO(kwen2501): this should be moved to c10d tests, to qualify a ECCL
// upgrade. Once a ECCL version is qualified, this code should not be needed at
// runtime.
#ifdef PGECCL_ENABLE_HASH
  if (enableCollecticeHashDebug_.load()) {
    auto numel = getTensorsNumel(inputs);
    auto hashValue = hashTensors(inputs);
    PRINT_COLLECTIVE_HASH_SIGNATURE("input", opTypeToString(opType), numel,
                                    hashValue);
  }
#endif

  {
    AutoEcclGroup eccl_group_guard(comm, eccl_use_nonblocking());
    for (const auto i : c10::irange(inputs.size())) {
      // Both `inputs' and `outputs' are created on a worker stream and used in
      // different ecclStreams.  Hence, both must record the ecclStream to
      // prevent being freed before the collective finishes.
      //
      // We only record `inputs' here, and leave recording `outputs' to `fn' for
      // operations where `inputs' and `outputs' are not the same.
      //
      // See [Sync Streams].
      if (!avoidRecordStreams) {
        if (!inputs[i].is_sparse()) {
          torch_gcu::GCUCachingAllocator::recordStream(
              inputs[i].storage().data_ptr(), ecclStream);
        } else {
          // for sparse input case record streams on both index and value
          // tensors
          torch_gcu::GCUCachingAllocator::recordStream(
              inputs[i].values().storage().data_ptr(), ecclStream);
          torch_gcu::GCUCachingAllocator::recordStream(
              inputs[i].indices().storage().data_ptr(), ecclStream);
        }
      }
#ifndef ECCL_HAS_COMM_NONBLOCKING
      C10D_ECCL_CHECK(fn(inputs[i], outputs[i], comm, ecclStream),
                      ecclComm->getEcclCommFailureReason());
#else
      C10D_ECCL_CHECK_TIMEOUT(fn(inputs[i], outputs[i], comm, ecclStream), comm,
                              ecclComm->getEcclCommFailureReason());
#endif
    }
  }

  // End event should only be recorded after the ecclGroupEnd()
  if (!coalescing_state_) {
    work->ecclEndEvent_->record(ecclStream);
  }
  work->ecclComm_ = ecclComm;

  {
    torch_gcu::GCUMultiStreamGuard streamGuard(ecclStream);
    std::vector<at::Device> devices{device};
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // Add a callback that runs profiling end callbacks. wrapCallback() in GCU
    // future blocks the stream this callback runs on the corresponding
    // ecclEndEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false allows us to skip synchronization in
          // ivalue::Future, but is only valid as long as the lambda doesn't use
          // the "Future" argument.
          /*uses_future=*/false);
    }
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Set appropriate work parameters.
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams;
  work->opTimeout_ = options_->timeout;
  work->store_ = store_;
  // Record size info for debug. We only record the size on the first device as
  // multi-device per process is deprecated
  work->numelIn_ = inputs[0].numel();
  work->numelOut_ = outputs[0].numel();

  // Notify graphs before we check the capture status preemptively
  torch_gcu::GCUGraph::inc_pending_event_queries();

  if (!coalescing_state_ && capture_status == torch_gcu::CaptureStatus::None) {
    workEnqueue(work);
  } else {
    torch_gcu::GCUGraph::dec_pending_event_queries();
  }
  if (torch_gcu::OpDebugConfig::GetInstance().enableSyncMode()) {
    StreamSynchronize(ecclStream);
  }
  return work;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::pointToPoint(
    at::Tensor& tensor, Fn fn, int peer, c10d::OpType opType, PreProcess pre,
    PostProcess post, const char* profilingTitle) {
  // avoidRecordStreams_ note:
  // send, recv, and irecv should be ok with avoidRecordStreams,
  // However, for isend, I don't think the API requires the user
  // to wait() on the returned handle, so ProcessGroupECCL can't know
  // when it's safe to release the input back to the allocator,
  // and the present call has no way to know it's not an isend.
  // Therefore, we warn and fall back to the typical recordStream logic:
  if (avoidRecordStreams_) {
    TORCH_WARN_ONCE(
        "TORCH_ECCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point "
        "collectives.");
  }

  auto device = getDevice(tensor);
  std::string key;
  int p2pRank = 0, p2pTargetRank = 0;
  bool isSendRecvSelf = false;
  // For batch_isend_irecv, ecclGroupStart() would be called upfront
  bool batchP2P = ecclActiveGroupCounter_ > 0;
  if (batchP2P) {
    // For batch P2P, we need to treat it like a collective when selecting
    // communicator, because other ranks can call into this batch other than my
    // rank and my peer
    key = getKeyFromDevice(device);
    p2pRank = rank_;
    p2pTargetRank = peer;
  } else {
    // For single P2P, preserve the old two-rank behavior (to avoid perf diff)
    key = getKeySendRecv(rank_, peer);
    p2pRank = rank_ <= peer ? 0 : 1;
    isSendRecvSelf = rank_ == peer;
    p2pTargetRank = isSendRecvSelf ? 0 : 1 - p2pRank;

    if (!coalescing_state_) {
      // Bump sequence number. Don't do so if it's a batch P2P, it will be
      // bumped in `endCoalescing`.
      seq_++;
    }
  }

  // Bump the logical operation counter regardless of whether this op is
  // coalesced or individual
  op_id_++;

  auto ecclComm = getECCLComm(key, device, opType, p2pRank, isSendRecvSelf);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalP2P;
    coalescedDevices_.push_back(device);
    coalescedComms_.push_back(ecclComm);
  }

  // Used many times below, so we stash the unordered_map lookup
  auto ecclStream = ecclStreams_.at(key);
  auto sip_num = torch_gcu::util::GetEnvInt("TORCH_GCU_ECCL_SIP_NUM", 12);
  if (sip_num != 12) {
    C10_GCU_CHECK(topsStreamSetLaunchLimit(ecclStream, 2, sip_num));
  }
  // First let ECCL streams wait for input tensors allocation streams
  // Note(torch_gcu):reuse GCUEvent will increase record time consumption, so
  // each time a new GCUEvent is created
  // syncStream(device, ecclEvents_[key], ecclStream);
  torch_gcu::GCUEvent ecclEvent;
  syncStream(device, ecclEvent, ecclStream);

  // c10d::Work itself will create the GCU events on all GCUs of tensors
  c10::intrusive_ptr<ProcessGroupECCL::WorkECCL> work;
  if (coalescing_state_) {
    // When coalescing, we record events per op that lack timing/state
    // information because there is no 'work' associated with them, and then
    // later in endCoalescing we record a 'coalesced' c10d::Work which has
    // timing/state updates via watchdog thread, but lacks op metadata such as
    // input/output sizes and profilingTitle per-op in the group.
    auto trace_id = 0;
    // auto trace_id =
    //     ECCLTraceBuffer::get()->record(uid_, seq_, op_id_, profilingTitle,
    //                                    {tensor}, {tensor}, nullptr, nullptr);
    // TODO(whc) if we want to make the per-p2p-op flightrecorder entries get
    // their timings/states updated by proxy when the c10d::Work obj
    // representing the coalesce group gets its update, we could accumulate
    // these trace_ids together and ask FlightRecorder to take the update from
    // one c10d::Work and apply it to multiple entries
    (void)trace_id;
  } else {
    // Store references to outputs to be used by WorkECCL::result and
    // operator<<. Note that these outputs are only valid for recv(), as send()
    // does not modify the inputs but we still create these outputs for use
    // cases such as profiling.

    work = initWork(device, rank_, opType, profilingTitle, {tensor}, {},
                    /*record=*/false);
    // This bypasses something in c10d::Work() that crashes if {tensor} is given
    // as output, not sure what
    work->outputs_ = std::make_shared<std::vector<at::Tensor>>();
    work->outputs_->push_back(tensor);
    work->ecclEvent_ = std::move(ecclEvent);
    // TODO(whc) because we don't pass output {tensor} to initWork, we tell
    // initWork to not record, and then we manually call record passing all the
    // information it wants.
    // work->trace_id_ = ECCLTraceBuffer::get()->record(
    //     uid_, seq_, op_id_, profilingTitle, {tensor}, {tensor},
    //     work->ecclStartEvent_.get(), work->ecclEndEvent_.get());
  }

  // is gcuGuard needed for the if block below, or can i swap them
  torch_gcu::OptionalGCUGuard gcuGuard;

  if (!coalescing_state_) {
    // Start event should only be recorded before the ecclGroupStart()
    if (work->timingEnabled_) {
      work->ecclStartEvent_->record(ecclStream);
    }

    pre(ecclStream, work);
  }

  // Both send tensor and recv tensor are created on a worker stream and used
  // in different ecclStreams.  Hence, both must record the ecclStream to
  // prevent being freed before the collective finishes.
  //
  // See [Sync Streams].
  torch_gcu::GCUCachingAllocator::recordStream(tensor.storage().data_ptr(),
                                               ecclStream);

  // This part seems common to both p2p and coalesced-p2p usage?
  ecclComm_t comm_ = ecclComm->getEcclComm();

#ifndef ECCL_HAS_COMM_NONBLOCKING
  C10D_ECCL_CHECK(fn(tensor, comm_, ecclStream, p2pTargetRank),
                  ecclComm->getEcclCommFailureReason());
#else
  C10D_ECCL_CHECK_TIMEOUT(fn(tensor, comm_, ecclStream, p2pTargetRank),
                          ecclComm->getEcclComm(),
                          ecclComm->getEcclCommFailureReason());
#endif

  if (!coalescing_state_) {
    post(ecclStream);

    // End event should only be recorded after the ecclGroupEnd()
    work->ecclEndEvent_->record(ecclStream);
    work->ecclComm_ = ecclComm;
    work->blockingWait_ = blockingWait_;
    work->opTimeout_ = options_->timeout;
    work->store_ = store_;
    // Record size info for debug. We only record the size on the first device
    // as multi-device per process is deprecated
    work->numelIn_ = work->numelOut_ = tensor.numel();

    // Future only needs to be created and marked completed with outputs for
    // recv(), but still create future for use cases such as profiling even for
    // send().
    {
      torch_gcu::GCUMultiStreamGuard streamGuard(ecclStream);
      std::vector<at::Device> devices{device};
      work->future_ = c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()), devices);
      work->future_->markCompleted(at::IValue(*work->outputs_));
    }

    // Add a callback that runs profiling end callbacks. wrapCallback() in GCU
    // future blocks the stream this callback runs on the corresponding
    // ecclEndEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false allows us to skip synchronization in
          // ivalue::Future, but is only valid as long as the lambda doesn't use
          // the "Future" argument.
          /*uses_future=*/false);
    }
  }

  // Enqueue P2P op so that it can be cancelled by ECCL watchdog
  torch_gcu::CaptureStatus capture_status =
      torch_gcu::currentStreamCaptureStatusMayInitCtx();

  // Notify graphs before we check the capture status preemptively
  torch_gcu::GCUGraph::inc_pending_event_queries();

  if (!coalescing_state_ && capture_status == torch_gcu::CaptureStatus::None) {
    workEnqueue(work);
    if (torch_gcu::OpDebugConfig::GetInstance().enableSyncMode()) {
      StreamSynchronize(ecclStream);
    }
    return work;
  } else {
    torch_gcu::GCUGraph::dec_pending_event_queries();
    return nullptr;
  }
}

template <typename Fn>
c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::collective(
    at::Tensor& input, at::Tensor& output, Fn fn, c10d::OpType opType,
    const char* profilingTitle, bool avoidRecordStreams) {
  return collective(
      input, output, fn,
      [](torch_gcu::GCUStream&,
         c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>& work) {},
      [](torch_gcu::GCUStream&,
         c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>& work) {},
      opType, profilingTitle, avoidRecordStreams);
}

template <typename Fn>
c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::pointToPoint(
    at::Tensor& tensor, Fn fn, int peer, c10d::OpType opType,
    const char* profilingTitle) {
  return pointToPoint(
      tensor, fn, peer, opType,
      [](torch_gcu::GCUStream&,
         c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>& work) {},
      [](torch_gcu::GCUStream&) {}, profilingTitle);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::allreduce_sparse(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
  DIST_API_TRACE_FUNC();
  TORCH_CHECK(false, "ProcessGroupeCCL not support allreduce_sparse.");
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
#ifdef IS_ECCL_EXP
  tensor = tensor.coalesce();
  at::Tensor outputTensor =
      torch::zeros(tensor.sizes(), tensor.options().layout(torch::kStrided));
  auto work = collective(
      tensor, outputTensor,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        auto ecclDataType = getEcclDataType(input.scalar_type(), true);
        auto ecclReduceOp =
            getEcclReduceOp(opts.reduceOp, input, ecclDataType, comm);

        size_t num_elements = output.numel();
        auto indices = input.indices();
        auto sizes = input.sizes();
        int colSize = sizes[1];
        auto rows = indices[0];
        size_t blockCount = rows.sizes()[0];
        auto recvIndices = indices[0] * colSize;

        // prevent output and recvIndices from being freed
        torch_gcu::GCUCachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        torch_gcu::GCUCachingAllocator::recordStream(
            recvIndices.storage().data_ptr(), stream);
        auto result = ecclAllReduceSparseBlock(
            input._values().data_ptr(),       // sendbuff
            recvIndices.data_ptr<int64_t>(),  // recv_indices
            blockCount,                       // block_count
            colSize,                          // block_length
            torch_gcu::gcu_data_ptr(output),  // recvbuff
            output.numel(),                   // recv_count
            ecclDataType, ecclReduceOp, comm, stream.stream());
        return result;
      },
      [](torch_gcu::GCUStream& ecclStream,
         c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>& work) {},
      [&](torch_gcu::GCUStream& ecclStream,
          c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>& work) {
        // Convert output tensors to sparse and back into tensors.
        torch_gcu::GCUStreamGuard guard(ecclStream);
        if (opts.sparseIndices.has_value()) {
          tensor = at::sparse_coo_tensor(opts.sparseIndices.scale(),
                                         outputTensor, tensor.sizes());
        } else {
          tensor = outputTensor.to_sparse();
        }
      },
      c10d::OpType::_ALLREDUCE_SPARSE, "eccl:all_reduce_sparse");
  return work;
#else
  // If the eccl branch is not "exp" then we just error
  C10_THROW_ERROR(
      Error,
      "allreduce_sparse is only available in the ECCL experimental branch.");
#endif
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::allreduce_impl(
    at::Tensor& tensor, const c10d::AllreduceOptions& opts) {
  return collective(
      tensor, tensor,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        auto ecclDataType = getEcclDataType(input.scalar_type(), true);
        auto ecclReduceOp =
            getEcclReduceOp(opts.reduceOp, input, ecclDataType, comm);
        return ecclAllReduce(torch_gcu::gcu_data_ptr(input),
                             torch_gcu::gcu_data_ptr(output), input.numel(),
                             ecclDataType, ecclReduceOp, comm, stream.stream());
      },
      c10d::OpType::ALLREDUCE, "eccl:all_reduce");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::allreduce(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
  DIST_API_TRACE_FUNC();
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    TORCH_CHECK(complexViewAsRealAllowed(opts.reduceOp),
                "all_reduce does not support", opts.reduceOp,
                "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  check_gcu_single_tensor(tensor);

  // if (intraNodeComm_ != nullptr && opts.reduceOp == c10d::ReduceOp::SUM) {
  //   using namespace c10d::intra_node_comm;
  //   auto algo = intraNodeComm_->selectAllReduceAlgo(tensor);
  //   if (algo != c10d::intra_node_comm::AllReduceAlgo::NONE) {
  //     intraNodeComm_->allReduce(tensor, algo);
  //     return c10::make_intrusive<IntraNodeCommWork>();
  //   }
  // }

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          tensors,                             // inputTensors
                          tensors,                             // outputTensors
                          rank_,                               // rank
                          "allreduce",                         // colName
                          tensor.numel(),                      // inNelems
                          tensor.numel(),                      // outNelems
                          tensor.scalar_type(),                // dType
                          std::vector<int64_t>(),              // inSplitSizes
                          std::vector<int64_t>(),              // outSplitSizes
                          globalRankStart,   // globalRankStart
                          globalRankStride,  // globalRankStride
                          this->getSize());  // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return allreduce_impl(tensor, opts);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::allreduce_outplace(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    const c10d::AllreduceOptions& opts) {
  DIST_API_TRACE_FUNC();
  if (inputTensor.is_complex()) {
    TORCH_CHECK(complexViewAsRealAllowed(opts.reduceOp),
                "all_reduce does not support", opts.reduceOp,
                "on complex tensors");
    inputTensor = at::view_as_real(inputTensor);
  }

  if (outputTensor.is_complex()) {
    TORCH_CHECK(complexViewAsRealAllowed(opts.reduceOp),
                "all_reduce does not support", opts.reduceOp,
                "on complex tensors");
    outputTensor = at::view_as_real(outputTensor);
  }
  check_gcu_single_tensor(inputTensor);
  check_gcu_single_tensor(outputTensor);

  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          inputTensor,                         // inputTensors
                          outputTensor,                        // outputTensors
                          rank_,                               // rank
                          "allreduce",                         // colName
                          inputTensor.numel(),                 // inNelems
                          outputTensor.numel(),                // outNelems
                          inputTensor.scalar_type(),           // dType
                          std::vector<int64_t>(),              // inSplitSizes
                          std::vector<int64_t>(),              // outSplitSizes
                          globalRankStart,   // globalRankStart
                          globalRankStride,  // globalRankStride
                          this->getSize());  // worldSize

  return collective(
      inputTensor, outputTensor,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        auto ecclDataType = getEcclDataType(input.scalar_type(), true);
        auto ecclReduceOp =
            getEcclReduceOp(opts.reduceOp, input, ecclDataType, comm);
        return ecclAllReduce(torch_gcu::gcu_data_ptr(input),
                             torch_gcu::gcu_data_ptr(output), input.numel(),
                             ecclDataType, ecclReduceOp, comm, stream.stream());
      },
      c10d::OpType::ALLREDUCE, "eccl:all_reduce");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceCoalescedOptions& opts) {
  DIST_API_TRACE_FUNC();
  auto total_numel = check_gcu_tensors_same_device(tensors);

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          tensors,                             // inputTensors
                          tensors,                             // outputTensors
                          rank_,                               // rank
                          "allreduce_coalesced",               // colName
                          total_numel,                         // inNelems
                          total_numel,                         // outNelems
                          tensors[0].scalar_type(),            // dType
                          // I'm not sure what in,outSplitSizes mean here.
                          std::vector<int64_t>(),  // inSplitSizes
                          std::vector<int64_t>(),  // outSplitSizes
                          globalRankStart,         // globalRankStart
                          globalRankStride,        // globalRankStride
                          this->getSize());        // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return collectiveCoalesced(
      tensors, tensors,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        auto ecclDataType = getEcclDataType(input.scalar_type(), true);
        auto ecclReduceOp =
            getEcclReduceOp(opts.reduceOp, input, ecclDataType, comm);
        return ecclAllReduce(torch_gcu::gcu_data_ptr(input),
                             torch_gcu::gcu_data_ptr(output), input.numel(),
                             ecclDataType, ecclReduceOp, comm, stream.stream());
      },
      c10d::OpType::COALESCED, "eccl:allreduce_coalesced");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::broadcast(
    std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {
  DIST_API_TRACE_FUNC();
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    tensor = at::view_as_real(tensor);
  }
  check_gcu_single_tensor(tensor);

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          tensors,                             // inputTensors
                          tensors,                             // outputTensors
                          opts.rootRank,                       // root rank
                          "broadcast",                         // colName
                          tensor.numel(),                      // inNelems
                          tensor.numel(),                      // outNelems
                          tensor.scalar_type(),                // dType
                          std::vector<int64_t>(),              // inSplitSizes
                          std::vector<int64_t>(),              // outSplitSizes
                          globalRankStart,   // globalRankStart
                          globalRankStride,  // globalRankStride
                          this->getSize());  // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  return collective(
      tensor, tensor,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        const auto root = opts.rootRank + opts.rootTensor;
        return ecclBroadcast(torch_gcu::gcu_data_ptr(input),
                             torch_gcu::gcu_data_ptr(input), input.numel(),
                             getEcclDataType(input.scalar_type()), root, comm,
                             stream.stream());
      },
      c10d::OpType::BROADCAST, "eccl:broadcast", avoidRecordStreams);
}

// _broadcast_oop adds an out-of-place broadcast in PGECCL
// Custom collectives may be implemented by coalescing broadcast operations
// One use-case is implementing a vector all_gather (all_gather_v)
// where unevenly sized inputs are gathered among participating ranks
// Since all_gather provides an out-of-place API, an all_gather_v
// semantic implemented inside pg_eccl.all_gather also needs to support
// out-of-place, for which an out-of-place broadcast is required to be added
c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::_broadcast_oop(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    const c10d::BroadcastOptions& opts) {
  DIST_API_TRACE_FUNC();
  if (outputTensor.numel() != inputTensor.numel()) {
    C10_THROW_ERROR(ValueError,
                    "Tensor input and output of _broadcast_oop must have the "
                    "same number of elements ");
  }

  return collective(
      inputTensor, outputTensor,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        const auto root = opts.rootRank + opts.rootTensor;
        return ecclBroadcast(torch_gcu::gcu_data_ptr(input),
                             torch_gcu::gcu_data_ptr(output), input.numel(),
                             getEcclDataType(input.scalar_type()), root, comm,
                             stream.stream());
      },
      c10d::OpType::BROADCAST, "eccl:_broadcast_oop");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::reduce(
    std::vector<at::Tensor>& tensors, const c10d::ReduceOptions& opts) {
  DIST_API_TRACE_FUNC();
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    TORCH_CHECK(complexViewAsRealAllowed(opts.reduceOp),
                "reduce does not support", opts.reduceOp, "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  check_gcu_single_tensor(tensor);
  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          tensors,                             // inputTensors
                          tensors,                             // outputTensors
                          opts.rootRank,                       // root rank
                          "reduce",                            // colName
                          tensor.numel(),                      // inNelems
                          tensor.numel(),                      // outNelems
                          tensor.scalar_type(),                // dType
                          std::vector<int64_t>(),              // inSplitSizes
                          std::vector<int64_t>(),              // outSplitSizes
                          globalRankStart,   // globalRankStart
                          globalRankStride,  // globalRankStride
                          this->getSize());  // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return collective(
      tensor, tensor,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        const auto root = opts.rootRank + opts.rootTensor;
        auto ecclDataType = getEcclDataType(input.scalar_type(), true);
        auto ecclReduceOp =
            getEcclReduceOp(opts.reduceOp, input, ecclDataType, comm);
        return ecclReduce(torch_gcu::gcu_data_ptr(input),
                          torch_gcu::gcu_data_ptr(output), input.numel(),
                          ecclDataType, ecclReduceOp, root, comm,
                          stream.stream());
      },
      c10d::OpType::REDUCE, "eccl:reduce");
}

// _reduce_oop exposes an out-of-place reduce from PGECCL
// Custom collectives may be implemented by coalescing reduce operations
// One use-case is implementing a vector reduce_scatter (reduce_scatter_v)
// where inputs are reduced and scattered unevenly among participating ranks
// Since reduce_scatter provides an out-of-place API, a reduce_scatter_v
// semantic implemented inside pg_eccl.reduce_scatter also needs to support
// out-of-place, for which an out-of-place reduce is required to be added
c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::_reduce_oop(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    const c10d::ReduceOptions& opts) {
  DIST_API_TRACE_FUNC();
  if (outputTensor.numel() != inputTensor.numel()) {
    C10_THROW_ERROR(ValueError,
                    "Tensor input and output of _reduce_oop must have the same "
                    "number of elements ");
  }

  return collective(
      inputTensor, outputTensor,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        const auto root = opts.rootRank + opts.rootTensor;
        const auto ecclDataType = getEcclDataType(input.scalar_type(), true);
        const auto ecclReduceOp =
            getEcclReduceOp(opts.reduceOp, input, ecclDataType, comm);
        return ecclReduce(torch_gcu::gcu_data_ptr(input),
                          torch_gcu::gcu_data_ptr(output), input.numel(),
                          ecclDataType, ecclReduceOp, (int)root, comm,
                          stream.stream());
      },
      c10d::OpType::REDUCE, "eccl:_reduce_oop");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const c10d::AllgatherOptions& opts) {
  DIST_API_TRACE_FUNC();
  TORCH_CHECK(inputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto inputTensor = inputTensors.back();
  check_gcu_single_tensor(inputTensor);
  // @lint-ignore CLANGTIDY
  auto outputTensors_ = outputTensors.back();

  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          inputTensors,                        // inputTensors
                          outputTensors,                       // outputTensors
                          rank_,                               // rank
                          "all_gather",                        // colName
                          inputTensor.numel(),                 // inNelems
                          inputTensor.numel() *                // outNelems
                              this->getSize(),
                          inputTensor.scalar_type(),  // dType
                          std::vector<int64_t>(),     // inSplitSizes
                          std::vector<int64_t>(),     // outSplitSize
                          globalRankStart,            // globalRankStart
                          globalRankStride,           // globalRankStride
                          this->getSize());           // worldSize

  bool same_size = check_same_size(outputTensors_);
  if (same_size) {
    // Flatten a vector of tensors into a single, stacked tensor.
    at::Tensor outputFlattened = c10d::newLikeFlat(outputTensors_);

    return collective(
        inputTensor, outputFlattened,
        [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
            torch_gcu::GCUStream& stream) {
          if (!avoidRecordStreams_) {
            torch_gcu::GCUCachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          return ecclAllGather(torch_gcu::gcu_data_ptr(input),
                               torch_gcu::gcu_data_ptr(output), input.numel(),
                               getEcclDataType(input.scalar_type()), comm,
                               stream.stream());
        },
        [](torch_gcu::GCUStream& ecclStream,
           c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>& work) {
          // avoidRecordStreams_ note: We actually don't need to stash anything
          // here.
          //  - inputTensors is stashed onto work->stashed_for_allocator_safety_
          //    in collective().
          //  - outputFlattened is stashed onto work->outputs_ in collective().
          //  - User-facing outputTensors should be held by the user until after
          //    waiting on work_, or the call makes no sense.
          // So all participating tensors are accounted for, and won't be
          // released back to their allocation streams until after work_ is
          // waited on.
        },
        [&](torch_gcu::GCUStream& ecclStream,
            c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>& work) {
          // Copy the flattened output tensors to the outputs.
          torch_gcu::GCUStreamGuard guard(ecclStream);
          for (const auto j : c10::irange(outputTensors_.size())) {
            // See [Sync Streams].
            if (!avoidRecordStreams_) {
              torch_gcu::GCUCachingAllocator::recordStream(
                  outputTensors_[j].storage().data_ptr(), ecclStream);
            }
            outputTensors_[j].copy_(outputFlattened[j], true);
          }
        },
        c10d::OpType::ALLGATHER, "eccl:all_gather");
  } else {
    const auto num_reduces = outputTensors_.size();
    // startCoalescing();
    c10::intrusive_ptr<c10d::Work> work;
    for (const int i : c10::irange(num_reduces)) {
      auto& output = outputTensors_[i];
      auto& input = (i == rank_) ? inputTensor : output;
      auto broadcastOpts = c10d::BroadcastOptions{
          static_cast<int64_t>(i), static_cast<int64_t>(0), opts.timeout};
      work = _broadcast_oop(output, input, broadcastOpts);
    }
    // auto work = endCoalescing(c10d::OpType::ALLGATHER);
    return work;
  }
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const c10d::AllgatherOptions& /* unused */) {
  C10_THROW_ERROR(NotImplementedError,
                  "ProcessGroupECCL does not support allgather_coalesced");
}

c10::intrusive_ptr<c10d::Work>
ProcessGroupECCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
    const c10d::AllgatherOptions& opts) {
  DIST_API_TRACE_FUNC();
  return collectiveCoalesced(
      inputs, outputs,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        return ecclAllGather(torch_gcu::gcu_data_ptr(input),
                             torch_gcu::gcu_data_ptr(output), input.numel(),
                             getEcclDataType(input.scalar_type()), comm,
                             stream.stream());
      },
      c10d::OpType::COALESCED, "eccl:all_gather_into_tensor_coalesced");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ReduceScatterOptions& opts) {
  DIST_API_TRACE_FUNC();
  TORCH_CHECK(outputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto outputTensor = outputTensors.back();
  check_gcu_single_tensor(outputTensor);
  // @lint-ignore CLANGTIDY
  auto inputTensors_ = inputTensors.back();

  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          inputTensors,                        // inputTensors
                          outputTensors,                       // outputTensors
                          rank_,                               // rank
                          "reduce_scatter",                    // colName
                          outputTensor.numel() * this->getSize(),  // inNelems
                          outputTensor.numel(),                    // outNelems
                          outputTensor.scalar_type(),              // dType
                          std::vector<int64_t>(),  // inSplitSizes
                          std::vector<int64_t>(),  // outSplitSizes
                          globalRankStart,         // globalRankStart
                          globalRankStride,        // globalRankStride
                          this->getSize());        // worldSize

  bool same_size = check_same_size(inputTensors_);
  if (same_size) {
    // Flatten a vector of tensors into a single, stacked tensor.
    at::Tensor inputFlattened = c10d::newLikeFlat(inputTensors_);

    return collective(
        inputFlattened, outputTensor,
        [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
            torch_gcu::GCUStream& stream) {
          if (!avoidRecordStreams_) {
            torch_gcu::GCUCachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          const auto ecclDataType = getEcclDataType(input.scalar_type(), true);
          const auto ecclReduceOp =
              getEcclReduceOp(opts.reduceOp, input, ecclDataType, comm);
          return ecclReduceScatter(torch_gcu::gcu_data_ptr(input),
                                   torch_gcu::gcu_data_ptr(output),
                                   output.numel(), ecclDataType, ecclReduceOp,
                                   comm, stream.stream());
        },
        [&](torch_gcu::GCUStream& ecclStream,
            c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>& work) {
          if (avoidRecordStreams_) {
            // We only need to stash inputTensors.
            //  - inputFlattened is stashed onto
            //  work->stashed_for_allocator_safety_
            //    in collective().
            //  - User-facing outputTensors is stashed onto work->outputs_ in
            //  collective(),
            //    and should also be held by the user until after waiting on
            //    work_.
            auto& v = work->stashed_for_allocator_safety_;
            v->insert(v->end(), inputTensors_.begin(), inputTensors_.end());
          }

          // Copy the input tensors to the flattened inputs.
          torch_gcu::GCUStreamGuard guard(ecclStream);
          for (const auto j : c10::irange(inputTensors_.size())) {
            // See [Sync Streams].
            if (!avoidRecordStreams_) {
              torch_gcu::GCUCachingAllocator::recordStream(
                  inputTensors_[j].storage().data_ptr(), ecclStream);
            }
            inputFlattened[j].copy_(inputTensors_[j], true);
          }
        },
        [&](torch_gcu::GCUStream&,
            c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>& work) {},
        c10d::OpType::REDUCE_SCATTER, "eccl:reduce_scatter");
  } else {
    const auto num_reduces = inputTensors_.size();
    // startCoalescing();
    c10::intrusive_ptr<c10d::Work> work;
    for (const int i : c10::irange(num_reduces)) {
      auto& input = inputTensors_[i];
      auto& output = (i == rank_) ? outputTensor : input;
      auto reduceOpts =
          c10d::ReduceOptions{opts.reduceOp, static_cast<int64_t>(i),
                              static_cast<int64_t>(0), opts.timeout};
      work = _reduce_oop(output, input, reduceOpts);
    }
    // auto work = endCoalescing(c10d::OpType::REDUCE_SCATTER);
    return work;
  }
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::_reduce_scatter_base(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    const c10d::ReduceScatterOptions& opts) {
  DIST_API_TRACE_FUNC();
  if (inputTensor.dtype() != outputTensor.dtype()) {
    C10_THROW_ERROR(TypeError,
                    "input tensor must be the same type as the output tensor.");
  }

  if (inputTensor.numel() != outputTensor.numel() * size_) {
    C10_THROW_ERROR(
        ValueError,
        "input tensor must be the same size as output size times world size");
  }

  // @lint-ignore CLANGTIDY
  const auto& tensor = outputTensor;
  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          inputTensor,                         // inputTensor
                          outputTensor,                        // outputTensor
                          rank_,                               // rank
                          "_reduce_scatter_base",              // colName
                          inputTensor.numel(),                 // inNelems
                          tensor.numel(),                      // outNelems
                          tensor.scalar_type(),                // dtype
                          std::vector<int64_t>(),              // inSplitSizes
                          std::vector<int64_t>(),              // outSplitSizes
                          globalRankStart,   // globalRankStart
                          globalRankStride,  // globalRankStride
                          this->getSize());  // worldSize

  // avoidRecordStreams_ note: collective() will stash inputs and outputs.
  // Note 2: for asyncOp = false, we don't want to record streams because we
  // know that the ECCL stream will join back to the "current" stream right
  // after this op. So we might just as well keep the stream ownership of the
  // input/output tensors unchanged. The benefit would be that the
  // allocation/free of the tensors would look deterministic to the "current"
  // stream so that the caching allocator can reuse memory pool for this stream
  // in a clever way. This setting is added for libraries like FSDP which uses
  // `reduce_scatter_tensor`.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  return collective(
      inputTensor, outputTensor,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        if (!avoidRecordStreams) {
          torch_gcu::GCUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        auto ecclDataType = getEcclDataType(input.scalar_type(), true);
        auto ecclReduceOp =
            getEcclReduceOp(opts.reduceOp, input, ecclDataType, comm);
        return ecclReduceScatter(
            torch_gcu::gcu_data_ptr(input), torch_gcu::gcu_data_ptr(output),
            output.numel(), ecclDataType, ecclReduceOp, comm, stream.stream());
      },
      c10d::OpType::_REDUCE_SCATTER_BASE, "eccl:_reduce_scatter_base",
      avoidRecordStreams);
}

c10::intrusive_ptr<c10d::Work>
ProcessGroupECCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
    const c10d::ReduceScatterOptions& opts) {
  return collectiveCoalesced(
      inputs, outputs,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        if (!avoidRecordStreams_) {
          torch_gcu::GCUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        auto ecclDataType = getEcclDataType(input.scalar_type(), true);
        auto ecclReduceOp =
            getEcclReduceOp(opts.reduceOp, input, ecclDataType, comm);
        return ecclReduceScatter(
            torch_gcu::gcu_data_ptr(input), torch_gcu::gcu_data_ptr(output),
            output.numel(), ecclDataType, ecclReduceOp, comm, stream.stream());
      },
      c10d::OpType::COALESCED, "eccl:reduce_scatter_tensor_coalesced");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::barrier(
    const c10d::BarrierOptions& opts) {
  DIST_API_TRACE_FUNC();
  RECORD_PARAM_COMMS(static_cast<int>(this->getSequenceNumberForGroup() +
                                      1),  // seq + 1 to match collective
                     std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                     rank_,                               // rank
                     "barrier",                           // colName
                     0,                                   // inNelems
                     0,                                   // outNelems
                     at::kByte,                           // dType
                     std::vector<int64_t>(),              // inSplitSizes
                     std::vector<int64_t>(),              // outSplitSizes
                     globalRankStart,                     // globalRankStart
                     globalRankStride,                    // globalRankStride
                     this->getSize());                    // worldSize

  std::vector<at::Device> devices;

  // Use user defined GCU device ids if provided
  if (!opts.device_ids.empty()) {
    for (auto device : opts.device_ids) {
      devices.emplace_back(at::DeviceType::PrivateUse1, device);
    }
  } else if (usedDeviceIdxs_.empty()) {
    // This means there is not yet a ECCL collective being called
    // Here we have to use the best guesses and will use a single GCU to call
    // allreduce to achieve barrier.
    // In case the multiple processes fall into the same node, we use rank to
    // ensure that each process is on a different GCU
    auto numGCUs = torch_gcu::device_count();
    int16_t deviceIdx = static_cast<int16_t>(rank_ % numGCUs);
    LOG(INFO) << logPrefix()
              << c10::str(" using GCU ", deviceIdx,
                          " to perform barrier as devices used by this process "
                          "are currently unknown. ",
                          "This can potentially cause a hang if this rank to "
                          "GCU mapping is incorrect.",
                          "Specify device_ids in barrier() to force use of a "
                          "particular device.");
    devices.emplace_back(guessDeviceForRank());
  } else {
    for (auto usedDeviceIdx : usedDeviceIdxs_) {
      devices.emplace_back(at::DeviceType::PrivateUse1, usedDeviceIdx);
    }
  }

  // Use one device only
  auto device = devices.back();
  at::Tensor barrierTensor =
      at::empty({1}, at::TensorOptions().device(device).dtype(at::kByte));
  // All reduce to achieve the barrier
  auto work = allreduce_impl(barrierTensor);

  // c10d::Work will take over barrierTensors
  auto ecclWork = dynamic_cast<ProcessGroupECCL::WorkECCL*>(work.get());
  TORCH_CHECK(ecclWork);
  ecclWork->barrierTensor_ = std::move(barrierTensor);
  return work;
}

void all2all_single_equal_split(at::Tensor& input, at::Tensor& output, int size,
                                ecclComm_t comm, torch_gcu::GCUStream& stream) {
  int numranks;
  auto type = getEcclDataType(input.scalar_type());
  size_t count = input.numel() / size;
  // size_t rankdiff = input.nbytes() / size;
  size_t rankdiff = input.numel() * get_gcu_element_size(input) / size;
  const auto* sendbuff =
      reinterpret_cast<char*>(torch_gcu::gcu_data_ptr(input));
  auto* recvbuff = reinterpret_cast<char*>(torch_gcu::gcu_data_ptr(output));

  C10D_ECCL_CHECK(ecclCommCount(comm, &numranks), c10::nullopt);
  C10D_ECCL_CHECK(ecclGroupStart(), c10::nullopt);
  for (const auto r : c10::irange(numranks)) {
    C10D_ECCL_CHECK(
        ecclSend(sendbuff + r * rankdiff, count, type, r, comm, stream),
        c10::nullopt);
    C10D_ECCL_CHECK(
        ecclRecv(recvbuff + r * rankdiff, count, type, r, comm, stream),
        c10::nullopt);
  }
#ifndef ECCL_HAS_COMM_NONBLOCKING
  C10D_ECCL_CHECK(ecclGroupEnd(), c10::nullopt);
#else
  C10D_ECCL_CHECK_TIMEOUT(ecclGroupEnd(), comm, c10::nullopt);
#endif
}

void all2all_single_unequal_split(void* sendbuff, const size_t* sendcounts,
                                  const size_t* senddispls, void* recvbuff,
                                  const size_t* recvcounts,
                                  const size_t* recvdispls, size_t size,
                                  c10::ScalarType _type, ecclComm_t comm,
                                  torch_gcu::GCUStream& stream) {
  auto type = getEcclDataType(_type);
  int numranks;
  C10D_ECCL_CHECK(ecclCommCount(comm, &numranks), c10::nullopt);
  C10D_ECCL_CHECK(ecclGroupStart(), c10::nullopt);
  for (const auto r : c10::irange(numranks)) {
    C10D_ECCL_CHECK(ecclSend(((char*)sendbuff) + senddispls[r] * size,
                             sendcounts[r], type, r, comm, stream),
                    c10::nullopt);

    C10D_ECCL_CHECK(ecclRecv(((char*)recvbuff) + recvdispls[r] * size,
                             recvcounts[r], type, r, comm, stream),
                    c10::nullopt);
  }

#ifndef ECCL_HAS_COMM_NONBLOCKING
  C10D_ECCL_CHECK(ecclGroupEnd(), c10::nullopt);
#else
  C10D_ECCL_CHECK_TIMEOUT(ecclGroupEnd(), comm, c10::nullopt);
#endif
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::alltoall_base(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const c10d::AllToAllOptions& /* unused */) {
  DIST_API_TRACE_FUNC();
  check_gcu_single_tensor(outputTensor, true);
  check_gcu_single_tensor(inputTensor, true);

  c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
  c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);

  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          inputTensor,                         // inputTensor
                          outputTensor,                        // outputTensor
                          rank_,                               // rank
                          "all_to_all",                        // colName
                          inputTensor.numel(),                 // inNelems
                          outputTensor.numel(),                // outNelems
                          inputTensor.scalar_type(),           // dType
                          inputSplitSizes,                     // inSplitSizes
                          outputSplitSizes,                    // outSplitSizes
                          globalRankStart,   // globalRankStart
                          globalRankStride,  // globalRankStride
                          this->getSize());  // worldSize

  // avoidRecordStreams_ note: collective() will stash inputTensors and
  // outputTensors.
  return collective(
      inputTensor, outputTensor,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        std::vector<size_t> send_lengths(size_);
        std::vector<size_t> recv_lengths(size_);
        std::vector<size_t> send_offsets(size_);
        std::vector<size_t> recv_offsets(size_);
        c10d::computeLengthsAndOffsets(inputSplitSizes, input, &send_lengths,
                                       &send_offsets);
        c10d::computeLengthsAndOffsets(outputSplitSizes, output, &recv_lengths,
                                       &recv_offsets);
        // See [Sync Streams].
        if (!avoidRecordStreams_) {
          torch_gcu::GCUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }

        return ecclAlltoAllv(
            torch_gcu::gcu_data_ptr(input), send_lengths.data(),
            send_offsets.data(), getEcclDataType(input.scalar_type()),
            torch_gcu::gcu_data_ptr(output), recv_lengths.data(),
            recv_offsets.data(), getEcclDataType(output.scalar_type()), comm,
            stream);
      },
      c10d::OpType::ALLTOALL_BASE, "eccl:all_to_all");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::allgatherv(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    const std::vector<size_t>& recvCounts, const c10d::AllgatherOptions& opts) {
  DIST_API_TRACE_FUNC();
  check_gcu_single_tensor(inputTensor);
  check_gcu_single_tensor(outputTensor);

  if (inputTensor.dtype() != outputTensor.dtype()) {
    C10_THROW_ERROR(TypeError,
                    "output tensor must have the same type as input tensor");
  }

  size_t scale = 1;
  if (inputTensor.dim() > 0) {
    for (int i = 1; i < inputTensor.dim(); ++i) {
      scale *= inputTensor.size(i);
    }
  }
  std::vector<size_t> recv_displs, recv_counts_scale;
  recv_counts_scale.reserve(recvCounts.size());
  recv_displs.reserve(recvCounts.size());
  recv_displs.push_back(0);
  for (size_t i = 0; i < recvCounts.size(); ++i) {
    recv_counts_scale.push_back(recvCounts[i] * scale);
    recv_displs.push_back(recv_displs.back() + recv_counts_scale.back());
  }

  TORCH_CHECK_WITH(
      ValueError,
      std::accumulate(recv_counts_scale.begin(), recv_counts_scale.end(), 0) <=
          outputTensor.numel(),
      "Output tensor size less then the sum of recvCounts.");

  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          inputTensor,                         // inputTensors
                          outputTensor,                        // outputTensors
                          rank_,                               // rank
                          "allgatherv",                        // colName
                          inputTensor.numel(),                 // inNelems
                          outputTensor.numel(),                // outNelems
                          inputTensor.scalar_type(),           // dType
                          std::vector<int64_t>(),              // inSplitSizes
                          std::vector<int64_t>(),              // outSplitSizes
                          globalRankStart,   // globalRankStart
                          globalRankStride,  // globalRankStride
                          this->getSize());  // worldSize

  return collective(
      inputTensor, outputTensor,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        auto ecclDataType = getEcclDataType(input.scalar_type(), false);
        return ecclAllGatherV(
            torch_gcu::gcu_data_ptr(input), recv_counts_scale[rank_],
            torch_gcu::gcu_data_ptr(output), recv_counts_scale.data(),
            recv_displs.data(), ecclDataType, comm, stream.stream());
        return ecclSuccess;
      },
      c10d::OpType::UNKNOWN, "eccl:allgatherv");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::reduce_scatterv(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    const std::vector<size_t>& recvCounts,
    const c10d::ReduceScatterOptions& opts) {
  DIST_API_TRACE_FUNC();
  check_gcu_single_tensor(inputTensor);
  check_gcu_single_tensor(outputTensor);

  if (inputTensor.dtype() != outputTensor.dtype()) {
    C10_THROW_ERROR(TypeError,
                    "output tensor must have the same type as input tensor");
  }

  // Calculate scale factor for multi-dimensional tensors
  size_t scale = 1;
  if (inputTensor.dim() > 0) {
    for (int i = 1; i < inputTensor.dim(); ++i) {
      scale *= inputTensor.size(i);
    }
  }

  // Calculate send counts and displacements for each rank
  // This implements the variable-length reduce-scatter where each rank
  // can receive different amounts of data
  std::vector<size_t> send_displs, send_counts_scale;
  send_counts_scale.reserve(recvCounts.size());
  send_displs.reserve(recvCounts.size());
  send_displs.push_back(0);
  for (size_t i = 0; i < recvCounts.size(); ++i) {
    send_counts_scale.push_back(recvCounts[i] * scale);
    send_displs.push_back(send_displs.back() + send_counts_scale.back());
  }

  // Validate input tensor size
  TORCH_CHECK_WITH(
      ValueError,
      std::accumulate(send_counts_scale.begin(), send_counts_scale.end(), 0) <=
          inputTensor.numel(),
      "Input tensor size less than the sum of recvCounts.");

  // Validate output tensor size
  TORCH_CHECK_WITH(
      ValueError, outputTensor.numel() >= send_counts_scale[rank_],
      "Output tensor size less than expected receive count for this rank.");

  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          inputTensor,                         // inputTensors
                          outputTensor,                        // outputTensors
                          rank_,                               // rank
                          "reduce_scatterv",                   // colName
                          inputTensor.numel(),                 // inNelems
                          outputTensor.numel(),                // outNelems
                          inputTensor.scalar_type(),           // dType
                          std::vector<int64_t>(),              // inSplitSizes
                          std::vector<int64_t>(),              // outSplitSizes
                          globalRankStart,   // globalRankStart
                          globalRankStride,  // globalRankStride
                          this->getSize());  // worldSize

  // Use native ECCL ecclReduceScatterV function
  return collective(
      inputTensor, outputTensor,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        auto ecclDataType = getEcclDataType(input.scalar_type(), true);
        auto ecclReduceOp =
            getEcclReduceOp(opts.reduceOp, input, ecclDataType, comm);
        if (ecclAvg == ecclReduceOp) {
          ecclReduceOp.op_ = ecclSum;
        }
        // Use the native ecclReduceScatterV function
        return ecclReduceScatterV(torch_gcu::gcu_data_ptr(input),  // sendbuff
                                  send_counts_scale.data(),        // sendcounts
                                  send_displs.data(),              // sdispls
                                  torch_gcu::gcu_data_ptr(output),  // recvbuff
                                  send_counts_scale[rank_],         // recvcount
                                  ecclDataType,                     // datatype
                                  ecclReduceOp,                     // op
                                  comm,                             // comm
                                  stream.stream()                   // stream
        );
      },
      [](torch_gcu::GCUStream& ecclStream,
         c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>& work) {},
      [&](torch_gcu::GCUStream& ecclStream,
          c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>& work) {
        // Post-processing: handle average reduction if needed
        if (opts.reduceOp == c10d::ReduceOp::AVG) {
          torch_gcu::GCUStreamGuard guard(ecclStream);
          outputTensor.div_(getSize());
        }
      },
      c10d::OpType::REDUCE_SCATTER, "eccl:reduce_scatterv");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::alltoallv_d(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    at::Tensor& outputSplitSizes, at::Tensor& inputSplitSizes, int32_t flag,
    const c10d::AllToAllOptions& /* unused */) {
  DIST_API_TRACE_FUNC();
  check_gcu_single_tensor(outputTensor, true);
  check_gcu_single_tensor(inputTensor, true);
  check_gcu_single_tensor(outputSplitSizes, true);
  check_gcu_single_tensor(inputSplitSizes, true);
  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          inputTensor,                         // inputTensor
                          outputTensor,                        // outputTensor
                          rank_,                               // rank
                          "alltoallv_d",                       // colName
                          inputTensor.numel(),                 // inNelems
                          outputTensor.numel(),                // outNelems
                          inputTensor.scalar_type(),           // dType
                          std::vector<int64_t>(),              // inSplitSizes
                          std::vector<int64_t>(),              // outSplitSizes
                          globalRankStart,   // globalRankStart
                          globalRankStride,  // globalRankStride
                          this->getSize());  // worldSize

  int scale = 1;
  if (inputTensor.dim() > 0) {
    for (int i = 1; i < inputTensor.dim(); ++i) {
      scale *= inputTensor.size(i);
    }
  }
  at::Tensor eccl_input_split = torch::empty_like(inputSplitSizes);
  at::Tensor eccl_output_split = torch::empty_like(outputSplitSizes);
  auto prob = torch_gcu::getCurrentDeviceProperties();
  // avoidRecordStreams_ note: collective() will stash inputTensors and
  // outputTensors.
  auto work = collective(
      inputTensor, outputTensor,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        // See [Sync Streams].
        if (!avoidRecordStreams_) {
          torch_gcu::GCUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
          torch_gcu::GCUCachingAllocator::recordStream(
              eccl_input_split.storage().data_ptr(), stream);
          torch_gcu::GCUCachingAllocator::recordStream(
              eccl_output_split.storage().data_ptr(), stream);
        }
        if (prob->major == 3) {
          return ecclAlltoAllv_2(torch_gcu::gcu_data_ptr(input),
                                 torch_gcu::gcu_data_ptr(eccl_input_split),
                                 getEcclDataType(input.scalar_type()),
                                 torch_gcu::gcu_data_ptr(output),
                                 torch_gcu::gcu_data_ptr(eccl_output_split),
                                 getEcclDataType(output.scalar_type()), comm,
                                 stream.stream(), flag);

        } else {
          return ecclAlltoAllvd(torch_gcu::gcu_data_ptr(input),
                                torch_gcu::gcu_data_ptr(inputSplitSizes),
                                getEcclDataType(input.scalar_type()),
                                torch_gcu::gcu_data_ptr(output),
                                torch_gcu::gcu_data_ptr(outputSplitSizes),
                                getEcclDataType(output.scalar_type()), scale,
                                comm, stream.stream(), flag);
        }
      },
      [&](torch_gcu::GCUStream& ecclStream,
          c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>& work) {
        if (prob->major == 3) {
          torch_gcu::GCUStreamGuard guard(ecclStream);
          bridge_topsextsCombinePreprocess_out2(
              eccl_output_split, eccl_input_split, outputSplitSizes,
              inputSplitSizes, scale, scale);
        }
      },
      [&](torch_gcu::GCUStream& ecclStream,
          c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>& work) {
        if (avoidRecordStreams_) {
          auto& v = work->stashed_for_allocator_safety_;
          v->emplace_back(eccl_input_split);
          v->emplace_back(eccl_output_split);
        }
        if (flag == 1) {
          if (prob->major == 3) {
            torch_gcu::GCUStreamGuard guard(ecclStream);
            bridge_topsextsDispatchPostprocess_out1(outputSplitSizes,
                                                    eccl_output_split, scale);
          }
        }
      },
      c10d::OpType::UNKNOWN, "eccl:alltoallvd");

  return work;
}

void all2all(std::vector<at::Tensor>& outputTensors,
             std::vector<at::Tensor>& inputTensors, ecclComm_t comm,
             ::torch_gcu::GCUStream& stream) {
  C10D_ECCL_CHECK(ecclGroupStart(), c10d::nullopt);
  for (const auto r : c10::irange(outputTensors.size())) {
    at::Tensor& input = inputTensors[r];
    at::Tensor& output = outputTensors[r];
    C10D_ECCL_CHECK(ecclSend(::torch_gcu::gcu_data_ptr(input), input.numel(),
                             getEcclDataType(input.scalar_type()), r, comm,
                             stream.stream()),
                    c10::nullopt);
    C10D_ECCL_CHECK(ecclRecv(::torch_gcu::gcu_data_ptr(output), output.numel(),
                             getEcclDataType(output.scalar_type()), r, comm,
                             stream.stream()),
                    c10::nullopt);
  }

#ifndef ECCL_HAS_COMM_NONBLOCKING
  C10D_ECCL_CHECK(ecclGroupEnd(), c10::nullopt);
#else
  C10D_ECCL_CHECK_TIMEOUT(ecclGroupEnd(), comm, c10d::nullopt);
#endif
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllToAllOptions& /* unused */) {
  DIST_API_TRACE_FUNC();
  std::vector<int64_t> inSplitSizes;
  std::vector<int64_t> outSplitSizes;
  int64_t total_numel = 0;

  auto device = outputTensors[0].device();
  for (const auto r : c10::irange(outputTensors.size())) {
    check_gcu_single_tensor(outputTensors[r], true);
    check_gcu_single_tensor(inputTensors[r], true);
    TORCH_CHECK(device == outputTensors[r].device() &&
                    device == inputTensors[r].device(),
                "Tensors must be on the same device")
    inSplitSizes.push_back(inputTensors[r].numel());
    outSplitSizes.push_back(outputTensors[r].numel());
    total_numel += inputTensors[r].numel();
  }

  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          inputTensors,                        // inputTensors
                          outputTensors,                       // outputTensors
                          rank_,                               // rank
                          "all_to_all",                        // colName
                          total_numel,                         // inNelems
                          total_numel,                         // outNelems
                          inputTensors.front().scalar_type(),  // dType
                          inSplitSizes,                        // inSplitSizes
                          outSplitSizes,                       // outSplitSizes
                          globalRankStart,   // globalRankStart
                          globalRankStride,  // globalRankStride
                          this->getSize());  // worldSize

  return collective(
      inputTensors[0], outputTensors[0],
      [&](at::Tensor& /* unused */, at::Tensor& /* unused */, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        all2all(outputTensors, inputTensors, comm, stream);
        return ecclSuccess;
      },
      [&](torch_gcu::GCUStream&,
          c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>& work) {
        if (avoidRecordStreams_) {
          // inputTensor0 and outputTensor0 are stashed redundantly by
          // collective(), but that's ok.
          auto& v = work->stashed_for_allocator_safety_;
          v->insert(v->end(), inputTensors.begin(), inputTensors.end());
          v->insert(v->end(), outputTensors.begin(), outputTensors.end());
        }
      },
      [](torch_gcu::GCUStream&,
         c10::intrusive_ptr<ProcessGroupECCL::WorkECCL>& work) {},
      c10d::OpType::ALLTOALL, "eccl:all_to_all");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::send(
    std::vector<at::Tensor>& tensors, int dstRank, int /* unused */) {
  DIST_API_TRACE_FUNC();
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  check_gcu_single_tensor(tensor, true);

  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          tensors,                             // inputTensors
                          tensors,                             // outputTensors
                          dstRank,                             // dst rank
                          "send",                              // colName
                          tensor.numel(),                      // inNelems
                          tensor.numel(),                      // outNelems
                          tensor.scalar_type(),                // dType
                          std::vector<int64_t>(),              // inSplitSizes
                          std::vector<int64_t>(),              // outSplitSizes
                          globalRankStart,   // globalRankStart
                          globalRankStride,  // globalRankStride
                          this->getSize());  // worldSize

  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& input, ecclComm_t comm, torch_gcu::GCUStream& stream,
          int dst) {
        return ecclSend(torch_gcu::gcu_data_ptr(input), input.numel(),
                        getEcclDataType(input.scalar_type()), dst, comm,
                        stream);
      },
      dstRank, c10d::OpType::SEND,
      c10::str("eccl:send ", rank_, "->", dstRank).c_str());
  return ret;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::recv(
    std::vector<at::Tensor>& tensors, int srcRank, int /* unused */) {
  DIST_API_TRACE_FUNC();
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  check_gcu_single_tensor(tensor, true);

  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          tensors,                             // inputTensors
                          tensors,                             // outputTensors
                          srcRank,                             // src rank
                          "recv",                              // colName
                          tensor.numel(),                      // inNelems
                          tensor.numel(),                      // outNelems
                          tensor.scalar_type(),                // dType
                          std::vector<int64_t>(),              // inSplitSizes
                          std::vector<int64_t>(),              // outSplitSizes
                          globalRankStart,   // globalRankStart
                          globalRankStride,  // globalRankStride
                          this->getSize());  // worldSize

  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& output, ecclComm_t comm, torch_gcu::GCUStream& stream,
          int src) {
        return ecclRecv(torch_gcu::gcu_data_ptr(output), output.numel(),
                        getEcclDataType(output.scalar_type()), src, comm,
                        stream);
      },
      srcRank, c10d::OpType::RECV,
      c10::str("eccl:recv ", rank_, "<-", srcRank).c_str());
  return ret;
}

void ProcessGroupECCL::groupStart() {
  C10D_ECCL_CHECK(ecclGroupStart(), c10::nullopt);
  ++ecclActiveGroupCounter_;
}

void ProcessGroupECCL::groupEnd() {
  C10D_ECCL_CHECK(ecclGroupEnd(), c10::nullopt);
  --ecclActiveGroupCounter_;
}

void ProcessGroupECCL::groupEndNonblocking(std::shared_ptr<ECCLComm> comm) {
#ifndef ECCL_HAS_COMM_NONBLOCKING
  C10D_ECCL_CHECK(ecclGroupEnd(), c10::nullopt);
#else
  if (!eccl_use_nonblocking()) {
    C10D_ECCL_CHECK(ecclGroupEnd(), c10::nullopt);
  } else {
    C10D_ECCL_CHECK_TIMEOUT_GROUPEND(ecclGroupEnd(), comm, c10::nullopt);
  }
#endif
  --ecclActiveGroupCounter_;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const c10d::GatherOptions& opts) {
  DIST_API_TRACE_FUNC();
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupECCL::gather: " + msg);
  };

  c10d::assertRootRank(invalidArgument, opts.rootRank, size_);

  TORCH_CHECK(inputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto inputTensor = inputTensors.back();

  std::vector<at::Tensor> outputs;

  if (getRank() == opts.rootRank) {
    if (outputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element output list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (outputTensors[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect output list size " << outputTensors[0].size()
         << ". Output list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = inputTensor.options();
    const auto& sizes = inputTensor.sizes();
    c10d::assertTypeAndSizesMatch(invalidArgument, outputTensors[0], options,
                                  sizes);
    outputs = outputTensors[0];
  } else {
    // if not in the root rank, initialize outputs as empty list
    if (outputTensors.size() != 0) {
      invalidArgument("requires empty output on non-root");
    }
    outputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    // outputs.emplace_back();
    // Workaround(torch_gcu): extractStorages() is very slow when
    // tensor.has_value() is false
    outputs.emplace_back(inputTensors.front());
  }

  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          inputTensors,                        // inputTensors
                          outputTensors,                       // outputTensors
                          opts.rootRank,                       // root rank
                          "gather",                            // colName
                          inputTensor.numel(),                 // inNelems
                          inputTensor.numel() * this->getSize(),  // outNelems
                          inputTensor.scalar_type(),              // dType
                          std::vector<int64_t>(),  // inSplitSizes
                          std::vector<int64_t>(),  // outSplitSize
                          globalRankStart,         // globalRankStart
                          globalRankStride,        // globalRankStride
                          this->getSize());        // worldSize

  // avoidRecordStreams_ note: collective() will stash inputTensors and
  // outputs, which == outputTensors[0] on the root rank where it matters.
  return collective(
      inputTensor,
      outputs[0],  // just to fit the collective interface
      [&](at::Tensor& /* unused */, at::Tensor& /* unused */, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        const auto root = opts.rootRank;
        if (getRank() == root) {
          if (!avoidRecordStreams_) {
            for (auto output : outputs) {
              torch_gcu::GCUCachingAllocator::recordStream(
                  output.storage().data_ptr(), stream);
            }
          }
        }
        ecclGroupStart();
        if (getRank() == root) {
          for (const auto r : c10::irange(getSize())) {
            if (r != root) {
              auto* recvbuff =
                  reinterpret_cast<char*>(torch_gcu::gcu_data_ptr(outputs[r]));
              C10D_ECCL_CHECK(
                  ecclRecv(recvbuff, outputs[r].numel(),
                           getEcclDataType(outputs[r].scalar_type()), r, comm,
                           stream),
                  c10::nullopt);
            } else {
              // on its own rank, simply copy from the input
              outputs[r].copy_(inputTensors[0]);
            }
          }
        } else {
          C10D_ECCL_CHECK(
              ecclSend(torch_gcu::gcu_data_ptr(inputTensors[0]),
                       inputTensors[0].numel(),
                       getEcclDataType(inputTensors[0].scalar_type()), root,
                       comm, stream),
              c10::nullopt);
        }
        ecclGroupEnd();
        return ecclSuccess;
      },
      c10d::OpType::GATHER, "eccl:gather");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ScatterOptions& opts) {
  DIST_API_TRACE_FUNC();
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupECCL::scatter: " + msg);
  };

  c10d::assertRootRank(invalidArgument, opts.rootRank, size_);

  TORCH_CHECK(outputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto outputTensor = outputTensors.back();

  std::vector<at::Tensor> inputs;

  if (getRank() == opts.rootRank) {
    if (inputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element input list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (inputTensors[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect input list size " << inputTensors[0].size()
         << ". Input list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = outputTensor.options();
    const auto& sizes = outputTensor.sizes();
    c10d::assertTypeAndSizesMatch(invalidArgument, inputTensors[0], options,
                                  sizes);
    inputs = inputTensors[0];
  } else {
    // if not in the root rank, initialize inputTensors as empty place holder
    // with an empty list
    if (inputTensors.size() != 0) {
      invalidArgument("requires empty input on non-root");
    }
    inputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    inputs.emplace_back();
  }

  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          inputTensors,                        // inputTensors
                          outputTensors,                       // outputTensors
                          opts.rootRank,                       // root rank
                          "scatter",                           // colName
                          outputTensor.numel() * this->getSize(),  // inNelems
                          outputTensor.numel(),                    // outNelems
                          outputTensor.scalar_type(),              // dType
                          std::vector<int64_t>(),  // inSplitSizes
                          std::vector<int64_t>(),  // outSplitSize
                          globalRankStart,         // globalRankStart
                          globalRankStride,        // globalRankStride
                          this->getSize());        // worldSize

  // avoidRecordStreams_ note: collective() will stash outputTensors and
  // inputs, which == inputTensors[0] on the root rank where it matters.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  return collective(
      outputTensor,
      inputs[0],  // just to fit the collective interface
      [&](at::Tensor& /* unused */, at::Tensor& /* unused */, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        const auto root = opts.rootRank;
        if (getRank() == root) {
          if (!avoidRecordStreams) {
            for (auto input : inputs) {
              torch_gcu::GCUCachingAllocator::recordStream(
                  input.storage().data_ptr(), stream);
            }
          }
        }
        ecclGroupStart();
        if (getRank() == root) {
          for (const auto r : c10::irange(getSize())) {
            if (r != root) {
              auto* sendbuff =
                  reinterpret_cast<char*>(torch_gcu::gcu_data_ptr(inputs[r]));
              C10D_ECCL_CHECK(ecclSend(sendbuff, inputs[r].numel(),
                                       getEcclDataType(inputs[r].scalar_type()),
                                       r, comm, stream),
                              c10::nullopt);
            } else {
              // on its own rank, simply copy it to the output
              outputTensors[0].copy_(inputs[r]);
            }
          }
        } else {
          C10D_ECCL_CHECK(
              ecclRecv(torch_gcu::gcu_data_ptr(outputTensors[0]),
                       outputTensors[0].numel(),
                       getEcclDataType(outputTensors[0].scalar_type()), root,
                       comm, stream),
              c10::nullopt);
        }
        ecclGroupEnd();
        return ecclSuccess;
      },
      c10d::OpType::SCATTER, "eccl:scatter", avoidRecordStreams);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */, int /* unused */) {
  C10_THROW_ERROR(NotImplementedError,
                  "ProcessGroupECCL does not support recvAnysource");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupECCL::_allgather_base(
    at::Tensor& output_tensor, at::Tensor& input_tensor,
    const c10d::AllgatherOptions& opts) {
  DIST_API_TRACE_FUNC();
  check_gcu_single_tensor(input_tensor);
  check_gcu_single_tensor(output_tensor);

  if (input_tensor.dtype() != output_tensor.dtype()) {
    C10_THROW_ERROR(TypeError,
                    "output tensor must have the same type as input tensor");
  }

  if (input_tensor.numel() * size_ != output_tensor.numel()) {
    C10_THROW_ERROR(ValueError,
                    "output tensor size must be equal to world_size times "
                    "input tensor size");
  }

  RECORD_PARAM_COMMS_DATA(static_cast<int>(this->getSequenceNumberForGroup() +
                                           1),  // seq + 1 to match collective
                          std::make_tuple(pg_uid_, pg_desc_),  // PG name tuple
                          input_tensor,                        // inputTensors
                          output_tensor,                       // outputTensors
                          rank_,                               // rank
                          "_allgather_base",                   // colName
                          input_tensor.numel(),                // inNelems
                          output_tensor.numel(),               // outNelems
                          output_tensor.scalar_type(),         // dType
                          std::vector<int64_t>(),              // inSplitSizes
                          std::vector<int64_t>(),              // outSplitSize
                          globalRankStart,   // globalRankStart
                          globalRankStride,  // globalRankStride
                          this->getSize());  // worldSize

  // avoidRecordStreams_ note: collective() will stash inputs and outputs.
  // Note 2: for asyncOp = false, we don't want to record streams because we
  // know that the ECCL stream will join back to the "current" stream right
  // after this op. So we might just as well keep the stream ownership of the
  // input/output tensors unchanged. The benefit would be that the
  // allocation/free of the tensors would look deterministic to the "current"
  // stream so that the caching allocator can reuse memory pool for this stream
  // in a clever way. This setting is added for libraries like FSDP which uses
  // `all_gather_into_tensor`.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  return collective(
      input_tensor, output_tensor,
      [&](at::Tensor& input, at::Tensor& output, ecclComm_t comm,
          torch_gcu::GCUStream& stream) {
        if (!avoidRecordStreams) {
          torch_gcu::GCUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        return ecclAllGather(torch_gcu::gcu_data_ptr(input),
                             torch_gcu::gcu_data_ptr(output), input.numel(),
                             getEcclDataType(input.scalar_type()), comm,
                             stream.stream());
      },
      c10d::OpType::_ALLGATHER_BASE, "eccl:_all_gather_base",
      avoidRecordStreams);
}

}  // namespace c10d_gcu
#endif  // USE_C10D_ECCL
