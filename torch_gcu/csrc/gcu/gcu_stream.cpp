#include "gcu/gcu_stream.h"

#include <c10/util/CallOnce.h>
#include <c10/util/irange.h>

#include <atomic>
#include <cstdint>
#include <torch_gcu/csrc/distributed/ECCLUtils.hpp>

#include "aten/op_debug_config.h"
#include "gcu/gcu_exception.h"
#include "gcu/gcu_functions.h"
#include "gcu/gcu_graphs_utils.h"
#include "gcu/gcu_guard.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_macros.h"
#include "gcu/gcu_utils.h"
#include "gcu/sys_util.h"
#include "topsaten/topsaten_define.h"

namespace torch_gcu {

namespace {

// Global stream state and constants
static c10::once_flag init_flag;
static c10::DeviceIndex num_gcus = -1;
static int kStreamsPerPoolBits = 5;
static constexpr int kMaxStreamNum = 32;
static int kStreamsPerPool = kMaxStreamNum;
static constexpr unsigned int kDefaultFlags = topsStreamNonBlocking;
static constexpr int kStreamTypeBits = 4;
static bool kUseGcuDefaultStream = true;

static int max_stream_priorities;

static c10::once_flag device_flags[C10_COMPILE_TIME_MAX_GCUS];
static std::atomic<uint32_t>
    priority_counters[max_compile_time_stream_priorities]
                     [C10_COMPILE_TIME_MAX_GCUS];

static topsStream_t streams[max_compile_time_stream_priorities]
                           [C10_COMPILE_TIME_MAX_GCUS][kMaxStreamNum];

// runtime default stream not support setlimit, so create new stream;
static topsStream_t kDefaultStream[C10_COMPILE_TIME_MAX_GCUS];
// See Pytorch Note [c10::StreamId assignment]
class StreamIdType {
  // StreamIdType encodes whether this stream is DEFAULT, EXTernal or
  // for all other native streams, the stream priority (higher value is higher
  // priority)
 private:
  uint8_t stream_type;

 public:
  static const uint8_t DEFAULT = 0x0;
  static const uint8_t EXT = 0xF;

 public:
  StreamIdType(const uint8_t _stream_type) : stream_type(_stream_type) {}

  bool isExt() const { return EXT == stream_type; }

  bool isDefault() const { return DEFAULT == stream_type; }

  uint8_t getStreamType() const { return stream_type; }
};

std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  if (s.isDefault()) {
    stream << "DEFAULT";
  } else if (s.isExt()) {
    stream << "EXT";
  } else {
    stream << "PRIORITY " << int(s.getStreamType());
  }
  return stream;
}

static inline StreamIdType streamIdType(c10::StreamId s) {
  // Externally allocated streams have their id being the cudaStream_ptr
  // so the last bit will be 0
  if ((!(s & 1)) && s) {
    return StreamIdType(StreamIdType::EXT);
  }
  // last bit is external/internal stream, the mask should start from second
  // rightmost bit
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  auto val = (s >> 1) & mask_for_type;
  TORCH_INTERNAL_ASSERT(val || !(s & 1), "invalid StreamId", s);
  return StreamIdType(val);
}

static inline size_t streamIdIndex(c10::StreamId s) {
  return static_cast<size_t>((s >> (kStreamTypeBits + 1)) &
                             ((1 << kStreamsPerPoolBits) - 1));
}

c10::StreamId makeStreamId(StreamIdType st, size_t si) {
  if (st.isDefault()) {
    return static_cast<c10::StreamId>(0);
  }
  return (static_cast<c10::StreamId>(si) << (kStreamTypeBits + 1)) |
         static_cast<c10::StreamId>(st.getStreamType() << 1) | 1;
}

// Thread-local current streams
static thread_local std::unique_ptr<c10::StreamId[]> current_streams = nullptr;

// Populates global values.
// Warning: this function must only be called once!
static void initGlobalStreamState() {
  num_gcus = device_count();
  // Check if the number of GCUs matches the expected compile-time max number
  // of GCUs.
  TORCH_CHECK(
      num_gcus <= C10_COMPILE_TIME_MAX_GCUS,
      "Number of GCU devices on the machine is larger than the compiled "
      "max number of gpus expected (",
      C10_COMPILE_TIME_MAX_GCUS, "). Increase that and recompile.");
  int leastPriority = -1, greatestPriority = -1;
  // gcu not support now
  // C10_GCU_CHECK(
  //     topsDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

  // greatestPriority is negative
  auto range = leastPriority - greatestPriority + 1;
  max_stream_priorities = range >= max_compile_time_stream_priorities
                              ? max_compile_time_stream_priorities
                              : range;
}

// Creates the low and high priority stream pools for the specified device
// Warning: only call once per device!
static void initDeviceStreamState(c10::DeviceIndex device_index) {
  // Switches to the requested device so streams are properly associated
  // with it.
  GCUGuard device_guard{device_index};
  kStreamsPerPool =
      torch_gcu::util::GetEnvInt("ENFLAME_PT_STREAM_NUM", kMaxStreamNum);
  kStreamsPerPool = std::max(kStreamsPerPool, kMaxStreamNum);
  kStreamsPerPoolBits = std::log2(kStreamsPerPool);
  kStreamsPerPool = 1 << kStreamsPerPoolBits;
  for (const auto i : c10::irange(kStreamsPerPool)) {
    for (const auto p : c10::irange(max_stream_priorities)) {
      auto& stream = streams[p][device_index][i];
      // auto pri = -p;  // lower number is higher priority
      C10_GCU_CHECK(topsStreamCreateWithFlags(&stream, kDefaultFlags));
      // maybe can do gcu stream trace like gpu
      // const c10::impl::PyInterpreter* interp = GCUTrace::get_trace();
      // if (C10_UNLIKELY(interp)) {
      //   (*interp)->trace_gpu_stream_creation(
      //       reinterpret_cast<uintptr_t>(stream));
      //   priority_counters[p][device_index] = 0;
      // }
    }
  }
}

// Init front-end to ensure initialization only occurs once
static void initGCUStreamsOnce() {
  // Inits default streams (once, globally)
  c10::call_once(init_flag, initGlobalStreamState);

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to default streams
  current_streams = std::make_unique<c10::StreamId[]>(num_gcus);
  kUseGcuDefaultStream = util::GetEnvBool("USE_GCU_DEFAULT_STREAM", true);
  for (const auto i : c10::irange(num_gcus)) {
    current_streams[i] = makeStreamId(StreamIdType::DEFAULT, 0);
    if (!kUseGcuDefaultStream) {
      GCUGuard device_guard{i};
      auto& stream = kDefaultStream[i];
      C10_GCU_CHECK(topsStreamCreateWithFlags(&stream, kDefaultFlags));
    }
  }
}

// Helper to verify the GCU index is valid
static inline void check_gcu(c10::DeviceIndex device_index) {
  TORCH_INTERNAL_ASSERT(device_index >= 0 && device_index < num_gcus);
}

// Helper to determine the index of the stream to return
// Note: Streams are returned round-robin (see note in GCUStream.h)
static uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

GCUStream GCUStreamForId(c10::DeviceIndex device_index,
                         c10::StreamId stream_id) {
  return GCUStream(
      GCUStream::UNCHECKED,
      c10::Stream(c10::Stream::UNSAFE,
                  c10::Device(c10::DeviceType::PrivateUse1, device_index),
                  stream_id));
}

}  // anonymous namespace

// See Note [StreamId assignment]
topsStream_t GCUStream::stream() const {
  c10::DeviceIndex device_index = stream_.device_index();
  c10::StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  size_t si = streamIdIndex(stream_id);
  if (st.isDefault()) {
    TORCH_INTERNAL_ASSERT(
        si == 0, "Unrecognized stream ", stream_,
        " (I think this should be the default stream, but I got a non-zero "
        "index ",
        si, ").",
        " Did you manufacture the StreamId yourself?  Don't do that; use the",
        " official API like getStreamFromPool() to get a new stream.");
    if (kUseGcuDefaultStream) {
      return nullptr;
    } else {
      return kDefaultStream[device_index];
    }
  } else if (st.isExt()) {
    return reinterpret_cast<topsStream_t>(stream_id);
  } else {
    auto streamType = st.getStreamType();
    TORCH_INTERNAL_ASSERT(
        streamType >= 1 && streamType <= max_stream_priorities,
        "Unrecognized stream ", stream_,
        " (I didn't recognize the stream type, ", st, " with the value ",
        streamType, ")");
    return streams[st.getStreamType() - 1][device_index][si];
  }
}

// Returns a stream from the requested pool
// Note: when called the first time on a device, this will create the
// stream pools for that device.
GCUStream getStreamFromPool(const int priority, c10::DeviceIndex device_index) {
  initGCUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
    // SetTargetDevice();
  }
  // TORCH_CHECK(
  //     priority <= 0,
  //     "Expected gcu stream priority to be less than or equal to 0, got ",
  //     priority);
  check_gcu(device_index);
  // Initializes the stream pools (once)
  c10::call_once(device_flags[device_index], initDeviceStreamState,
                 device_index);
  auto pri_idx = std::clamp(-priority, 0, max_stream_priorities - 1);
  const auto idx = get_idx(priority_counters[pri_idx][device_index]);
  StreamIdType id_type = StreamIdType(pri_idx + 1);
  return GCUStreamForId(device_index, makeStreamId(id_type, idx));
}

GCUStream getStreamFromPool(const bool isHighPriority,
                            c10::DeviceIndex device) {
  initGCUStreamsOnce();
  int priority = isHighPriority ? -max_stream_priorities + 1 : 0;
  return getStreamFromPool(priority, device);
}

GCUStream getStreamFromExternal(topsStream_t ext_stream,
                                c10::DeviceIndex device_index) {
  // The stream pointer will be the actual id
  return GCUStreamForId(device_index, reinterpret_cast<int64_t>(ext_stream));
}

GCUStream getDefaultGCUStream(c10::DeviceIndex device_index) {
  initGCUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
    // SetTargetDevice();
  }
  check_gcu(device_index);
  return GCUStreamForId(device_index, makeStreamId(StreamIdType::DEFAULT, 0));
}

GCUStream getCurrentGCUStream(c10::DeviceIndex device_index) {
  initGCUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
    // SetTargetDevice();
  }
  check_gcu(device_index);
  return GCUStreamForId(device_index, current_streams[device_index]);
}

void setCurrentGCUStream(GCUStream stream) {
  initGCUStreamsOnce();
  current_streams[stream.device_index()] = stream.id();
}

std::ostream& operator<<(std::ostream& os, const GCUStream& s) {
  os << s.unwrap();
  if (s.stream() != nullptr) {
    size_t cluster_num, sip_num;
    auto ret = topsStreamGetLaunchLimit(s.stream(), &cluster_num, &sip_num);
    if (ret == topsSuccess) {
      os << " cluster_num:" << cluster_num << ",sip_num:" << sip_num;
    }
    // clear the last error
    (void)topsGetLastError();
  }
  return os;
}

bool GCUStream::query() const {
  c10::DeviceGuard guard{stream_.device()};
  topsError_t err = topsStreamQuery(stream());

  if (err == topsSuccess) {
    return true;
  } else if (err != topsErrorNotReady) {
    C10_GCU_CHECK(err);
  } else {
    (void)topsGetLastError();
  }

  return false;
}

bool GCUStream::set_limit(const size_t cluster_num, const size_t sip_num) {
  if (stream() == nullptr) {
    TORCH_WARN(
        "Not support default stream,please export "
        "USE_GCU_DEFAULT_STREAM=false");
    return false;
  }
  C10_GCU_CHECK(topsStreamSetLaunchLimit(stream(), cluster_num, sip_num));
  CHECK_TOPSATEN_CALL(topsatenInit());
  return true;
}

bool GCUStream::get_limit(size_t& cluster_num, size_t& sip_num) const {
  if (stream() == nullptr) {
    TORCH_WARN(
        "Not support default stream,please export "
        "USE_GCU_DEFAULT_STREAM=false");
    return false;
  }

  auto ret = topsStreamGetLaunchLimit(stream(), &cluster_num, &sip_num);
  if (ret != topsSuccess) {
    return false;
  }
  return true;
}

void GCUStream::synchronize() const {
  c10::DeviceGuard guard{stream_.device()};
  StreamSynchronize(stream());
}

int GCUStream::priority() const {
  c10::DeviceGuard guard{stream_.device()};
  int priority = 0;
  // gcu not support now
  // C10_GCU_CHECK(topsStreamGetPriority(stream(), &priority));
  return priority;
}

std::tuple<int, int> GCUStream::priority_range() {
  int least_priority = 0, greatest_priority = 0;
  // gcu not support now
  // C10_GCU_CHECK(
  //     topsDeviceGetStreamPriorityRange(&least_priority,
  //     &greatest_priority));
  // TORCH_INTERNAL_ASSERT(least_priority == 0,
  //                       "Unexpected GCU stream priority range");
  // TORCH_INTERNAL_ASSERT(greatest_priority <= -1,
  //                       "Unexpected GCU stream priority range");
  // greatest_priority =
  //     std::max(-max_compile_time_stream_priorities + 1, greatest_priority);
  return std::make_tuple(least_priority, greatest_priority);
}

void maybeGCUStreamSynchronize(const GCUStream& stream) {
  if (OpDebugConfig::GetInstance().enableSyncMode()) {
    StreamSynchronize(stream);
  }
}

}  // namespace torch_gcu
