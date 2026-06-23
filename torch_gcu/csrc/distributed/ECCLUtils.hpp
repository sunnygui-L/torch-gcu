#pragma once
#ifdef USE_C10D_ECCL

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <mutex>

#include "eccl.h"

#if ECCL_VERSION_CODE >= ECCL_VERSION(3, 5, 1)
#define ENABLE_ECCL_PREMUL_SUM_SUPPORT
#endif

#define ENABLE_ECCL_ERROR_CHECKING

static std::string ecclGetStatusString(ecclResult_t result_code) {
  std::string status_msg;
  switch (result_code) {
    case ecclSuccess:
      status_msg = "ecclSuccess";
      break;
    case ecclUnhandledTopsError:
      status_msg = "ecclUnhandledTopsError";
      break;
    case ecclSystemError:
      status_msg = "ecclSystemError";
      break;
    case ecclInternalError:
      status_msg = "ecclInternalError";
      break;
    case ecclInvalidArgument:
      status_msg = "ecclInvalidArgument";
      break;
    case ecclInvalidUsage:
      status_msg = "ecclInvalidUsage";
      break;
    case ecclNumResults:
      status_msg = "ecclNumResults";
      break;
    default:
      status_msg = "Unknown ecclStatus";
      break;
  }
  return status_msg;
}

// Macro to throw on a non-successful ECCL return value.
#define C10D_ECCL_CHECK(cmd, failureReason)                               \
  do {                                                                    \
    ecclResult_t result = cmd;                                            \
    if (result != ecclSuccess) {                                          \
      std::string err = "ECCL error in: " + std::string(__FILE__) + ":" + \
                        std::to_string(__LINE__) + ", " +                 \
                        ecclGetStatusString(result);                      \
      TORCH_CHECK_WITH(DistBackendError, false, err);                     \
    }                                                                     \
  } while (0)

// Macro to throw on a non-successful ECCL return value, non-blocking.
#define C10D_ECCL_CHECK_TIMEOUT(cmd, comm, failureReason)                     \
  ecclResult_t result = cmd;                                                  \
  auto startTimepoint = std::chrono::steady_clock::now();                     \
  while (result == ecclInProgress) {                                          \
    if (eccl_nonblocking_timeout() > 0) {                                     \
      auto currentTimepoint = std::chrono::steady_clock::now();               \
      auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(    \
                             currentTimepoint - startTimepoint)               \
                             .count();                                        \
      if (timeElapsed > eccl_nonblocking_timeout()) {                         \
        std::string err = "ECCL timeout in: " + std::string(__FILE__) + ":" + \
                          std::to_string(__LINE__) + ", " +                   \
                          ecclGetStatusString(result);                        \
        TORCH_CHECK_WITH(DistBackendError, false, err);                       \
      }                                                                       \
    }                                                                         \
    ecclCommGetAsyncError(comm, &result);                                     \
  }                                                                           \
  if (result != ecclSuccess) {                                                \
    std::string err = "ECCL error in: " + std::string(__FILE__) + ":" +       \
                      std::to_string(__LINE__) + ", " +                       \
                      ecclGetStatusString(result);                            \
    TORCH_CHECK_WITH(DistBackendError, false, err);                           \
  }

#define C10D_ECCL_CHECK_TIMEOUT_GROUPEND(cmd, comms_, failureReason)           \
  ecclResult_t state = cmd;                                                    \
  auto startTimepoint = std::chrono::steady_clock::now();                      \
  if (state == ecclInProgress) {                                               \
    for (const auto i : c10::irange(comms_.size())) {                          \
      do {                                                                     \
        if (eccl_nonblocking_timeout() > 0) {                                  \
          auto currentTimepoint = std::chrono::steady_clock::now();            \
          auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>( \
                                 currentTimepoint - startTimepoint)            \
                                 .count();                                     \
          if (timeElapsed > eccl_nonblocking_timeout()) {                      \
            std::string err = "ECCL timeout in: " + std::string(__FILE__) +    \
                              ":" + std::to_string(__LINE__) + ", " +          \
                              ecclGetStatusString(state);                      \
            TORCH_CHECK_WITH(DistBackendError, false, err);                    \
          }                                                                    \
        }                                                                      \
        ecclCommGetAsyncError(comms_[i]->getEcclComm(), &state);               \
      } while (state == ecclInProgress);                                       \
      if (state != ecclSuccess) {                                              \
        break; /* fall through to failed case */                               \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  if (state != ecclSuccess) {                                                  \
    std::string err = "ECCL error in: " + std::string(__FILE__) + ":" +        \
                      std::to_string(__LINE__) + ", " +                        \
                      ecclGetStatusString(state);                              \
    TORCH_CHECK_WITH(DistBackendError, false, err);                            \
  }

// Macro to print and abort on a non-successful ECCL return value.
#define C10D_ECCL_ASSERT(cmd)                                           \
  do {                                                                  \
    ecclResult_t result = cmd;                                          \
    if (result != ecclSuccess) {                                        \
      std::string err = ecclGetStatusString(result);                    \
      fprintf(stderr, "ECCL error in: %s:%d, %s\n", __FILE__, __LINE__, \
              err.c_str());                                             \
      abort();                                                          \
    }                                                                   \
  } while (0)

namespace c10d_gcu {

std::string getEcclVersion();
std::string ecclGetErrorWithVersion(ecclResult_t error);
bool eccl_use_nonblocking();
int eccl_nonblocking_timeout();
// RAII helper class to manage Eccl group API and GCU free mutex.
// The destructor is allowed to throw since this helper class only
// manages group and lock lifetimes.
struct AutoEcclGroup {
  AutoEcclGroup();
  AutoEcclGroup(ecclComm_t comm, bool comm_nonblocking);
  ~AutoEcclGroup() noexcept(false);
  ecclComm_t comm_;
  bool comm_nonblocking_;
};

// RAII wrapper for ECCL communicator
class ECCLComm {
 public:
  explicit ECCLComm(ecclComm_t ecclComm)
      : ecclComm_(ecclComm),
        aborted_(false),
        ecclAsyncErr_(ecclSuccess),
        commFailureReason_(c10::nullopt) {}

  ECCLComm() : ECCLComm(nullptr) {}

  ~ECCLComm() noexcept {
    // Add lock in this destructor, as aborted_ needs to be read after memory
    // barrier here.
    std::unique_lock<std::mutex> lock(mutex_);
    if (ecclComm_ && !aborted_) {
      C10D_ECCL_ASSERT(::ecclCommDestroy(ecclComm_));
    }
  }

  static std::shared_ptr<ECCLComm> create(int numRanks, int rank,
                                          ecclUniqueId commId) {
    auto comm = std::make_shared<ECCLComm>();
    C10D_ECCL_CHECK(
        ecclCommInitRank(&(comm->ecclComm_), numRanks, commId, rank),
        c10::nullopt);
    comm->ecclId_ = commId;
    comm->rank_ = rank;
    return comm;
  }

#ifdef ECCL_HAS_COMM_SPLIT
  static std::shared_ptr<ECCLComm> split(ECCLComm* source, int color_id,
                                         int rank, ecclConfig_t& config) {
    auto comm = std::make_shared<ECCLComm>();
    C10D_ECCL_CHECK(ecclCommSplit(source->ecclComm_, color_id, rank,
                                  &(comm->ecclComm_), &config),
                    c10::nullopt);
    ++source->ecclCommSplitCounter_;
    comm->rank_ = rank;
    return comm;
  }
#endif
  ecclUniqueId getEcclId() { return ecclId_; }

  // Must not be copyable
  ECCLComm(const ECCLComm&) = delete;
  ECCLComm& operator=(const ECCLComm&) = delete;

  // Do not support move assignment as there is no valid use case
  ECCLComm& operator=(ECCLComm&& other) = delete;

  // Move constructable
  ECCLComm(ECCLComm&& other) {
    // Using other's lock, as it reads other's states
    // Can not use this.mutex_, as this object is being constructed.
    std::unique_lock<std::mutex> lock(other.mutex_);
    std::swap(ecclComm_, other.ecclComm_);
    std::swap(aborted_, other.aborted_);
    std::swap(ecclAsyncErr_, other.ecclAsyncErr_);
  }

  ecclComm_t getEcclComm();

  c10::optional<std::string> getEcclCommFailureReason() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return commFailureReason_;
  }

  void ecclCommAbort(
      c10::optional<std::string> /* commFailureReason */ = c10::nullopt) {
    std::unique_lock<std::mutex> lock(mutex_);
// This is a NOOP, if error checks are disabled.
#ifdef ECCL_HAS_COMM_REGISTER
    // Deregister all registered segments before aborting.
    for (auto& it : registeredSegmentHandles_) {
      void* handle = it.second;
      C10D_ECCL_CHECK(::ecclCommDeregister(ecclComm_, handle),
                      c10::str("Failed to deregister segment handle ", handle,
                               " on ecclComm_ ", ecclComm_));
    }
    registeredSegmentHandles_.clear();
#endif
    return;
  }

  bool isAborted() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return aborted_;
  }

  ecclResult_t checkForEcclError() {
    std::unique_lock<std::mutex> lock(mutex_);
    // Always return success, if error checks are disabled.
    return ecclSuccess;
  }

  ecclResult_t registerSegment(void* ptr, size_t size) {
    std::unique_lock<std::mutex> lock(mutex_);
#ifdef ECCL_HAS_COMM_REGISTER
    // We register only segments from cache allocator
    // which are guaranteed to be with disjoint addr ranges. Thus, a ptr always
    // maps to a unique handle and should not be registered before the current
    // ptr is deregistered and freed.
    TORCH_CHECK(registeredSegmentHandles_.count(ptr) == 0, "Segment with ptr ",
                ptr, " has already been registered on ecclComm_ ", ecclComm_);

    void* handle;
    C10D_ECCL_CHECK(ecclCommRegister(ecclComm_, ptr, size, &handle),
                    c10::str("Failed to register segment with ptr ", ptr,
                             ", size ", size, " on ecclComm_ ", ecclComm_));
    registeredSegmentHandles_[ptr] = handle;
    return ecclSuccess;
#else
    return ecclInvalidUsage;
#endif
  }

  ecclResult_t deregisterSegment(void* ptr) {
    std::unique_lock<std::mutex> lock(mutex_);
#ifdef ECCL_HAS_COMM_REGISTER
    TORCH_CHECK(registeredSegmentHandles_.count(ptr) == 1, "Segment with ptr ",
                ptr, " is not registered on ecclComm_ ", ecclComm_);

    void* handle = registeredSegmentHandles_[ptr];
    C10D_ECCL_CHECK(ecclCommDeregister(ecclComm_, handle),
                    c10::str("Failed to deregister segment handle ", handle,
                             ", with ptr ", ptr, " on ecclComm_ ", ecclComm_));
    registeredSegmentHandles_.erase(ptr);
    return ecclSuccess;
#else
    return ecclInvalidUsage;
#endif
  }

 public:
  ecclComm_t ecclComm_;
  // Unique eccl_id for this communicator.
  ecclUniqueId ecclId_;
  bool aborted_;
  ecclResult_t ecclAsyncErr_;
  mutable std::mutex mutex_;
  // Rank that this communicator corresponds to.
  int rank_;
  // Optional reason for communicator failure, provided by ProcessGroupECCL for
  // better error messaging.
  c10::optional<std::string> commFailureReason_;
  bool initialized_{false};
#ifdef ECCL_HAS_COMM_REGISTER
  // Stores handlers for tensors registered by ECCL
  std::unordered_map<void*, void*> registeredSegmentHandles_;
#endif
};

// Helper that automatically cleans up premul sums.
struct ecclRedOpRAII {
  ecclRedOpRAII() = default;
  ecclRedOpRAII(ecclRedOp_t op) : op_(op) {}
  ecclRedOpRAII(ecclRedOp_t op, ecclComm_t comm)
      : op_(op), comm_(comm), premul_sum_(true) {}
  ecclRedOpRAII(const ecclRedOpRAII&) = delete;
  ecclRedOpRAII& operator=(const ecclRedOpRAII&) = delete;
  ecclRedOpRAII(ecclRedOpRAII&& tmp) : ecclRedOpRAII() {
    std::swap(tmp.op_, this->op_);
    std::swap(tmp.comm_, this->comm_);
    std::swap(tmp.premul_sum_, this->premul_sum_);
  }
#if defined(ENABLE_ECCL_PREMUL_SUM_SUPPORT)
  ~ecclRedOpRAII() {
    if (premul_sum_) {
      ecclRedOpDestroy(op_, comm_);
    }
  }
#endif
  operator ecclRedOp_t() const { return op_; }
  ecclRedOp_t op_;
  ecclComm_t comm_;
  bool premul_sum_ = false;
};

}  // namespace c10d_gcu

#endif  // USE_C10D_ECCL
