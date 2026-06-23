#ifdef USE_C10D_ECCL

#include "torch_gcu/csrc/distributed/ECCLUtils.hpp"

#include <c10/util/CallOnce.h>
#include <c10/util/env.h>

#include <mutex>

namespace c10d_gcu {

ecclComm_t ECCLComm::getEcclComm() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (aborted_) {
    auto commFailureMsg = commFailureReason_ != c10::nullopt
                              ? c10::str(" Original reason for failure was: ",
                                         *commFailureReason_)
                              : "";
    TORCH_CHECK(false, c10::str("ECCL communicator was aborted on rank ", rank_,
                                ". ", commFailureMsg));
  }
  return ecclComm_;
}

std::string getEcclVersion() {
  // static c10::once_flag ecclGetVersionFlag;
  // static std::string versionString;

  // c10::call_once(ecclGetVersionFlag, []() {
  //   int version;
  //   ecclResult_t status = ecclGetVersion(&version);
  //   // can't compute the version if call did not return successfully or
  //   version
  //   // code < 100 (corresponding to 0.1.0)
  //   if (status != ecclSuccess || version < 100) {
  //     versionString = "Unknown ECCL version";
  //   } else {
  //     // ECCL changed version coding starting 2.9
  //     const int majorBase = version < 2900 ? 1000 : 10000;
  //     const int minorBase = 100;
  //     auto ecclMajor = version / majorBase;
  //     auto ecclMinor = (version % majorBase) / minorBase;
  //     auto ecclPatch =
  //         version % (ecclMajor * majorBase + ecclMinor * minorBase);
  //     versionString = std::to_string(ecclMajor) + "." +
  //         std::to_string(ecclMinor) + "." + std::to_string(ecclPatch);
  //   }
  // });

  return "3.0";
}

bool eccl_use_nonblocking() { return false; }

int _parse_eccl_nonblocking_timeout() {
  int timeout = -1;
  return timeout;
}

int eccl_nonblocking_timeout() {
  static int timeout = _parse_eccl_nonblocking_timeout();
  return timeout;
}

AutoEcclGroup::AutoEcclGroup() {
  comm_nonblocking_ = false;
  comm_ = nullptr;
  C10D_ECCL_CHECK(ecclGroupStart(), c10::nullopt);
}

AutoEcclGroup::AutoEcclGroup(ecclComm_t comm, bool comm_nonblocking) {
  comm_ = comm;
  comm_nonblocking_ = comm_nonblocking;
  C10D_ECCL_CHECK(ecclGroupStart(), c10::nullopt);
}

AutoEcclGroup::~AutoEcclGroup() noexcept(false) {
  C10D_ECCL_CHECK(ecclGroupEnd(), c10::nullopt);
}

std::string ecclGetErrorWithVersion(ecclResult_t error) {
  int version{3};
  // ecclGetVersion(&version);
  return std::string(ecclGetStatusString(error)) + ", ECCL version " +
         std::to_string(version);
}

}  // namespace c10d_gcu

#endif  // USE_C10D_ECCL
