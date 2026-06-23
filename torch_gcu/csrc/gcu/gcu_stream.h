/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <tops/tops_runtime_api.h>

#include <cstddef>
#include <tuple>

#include "gcu/gcu_macros.h"

/*
 * Stream pool note.
 *
 * A GCUStream is an abstraction of an actual topsStream on the GCU. GCUStreams
 * are backed by topsStreams, but they use several pools to minimize the costs
 * associated with creating, retaining, and destroying topsStreams.
 *
 * There are three pools per device, and a device's pools are lazily created.
 *
 * The first pool contains only the default stream. When the default stream
 * is requested it's returned.
 *
 * The second pool is the "low priority" or "default priority" streams. In
 * HIP builds there is no distinction between streams in this pool and streams
 * in the third pool (below). There are 32 of these streams per device, and
 * when a stream is requested one of these streams is returned round-robin.
 * That is, the first stream requested is at index 0, the second at index 1...
 * to index 31, then index 0 again.
 *
 * This means that if 33 low priority streams are requested, the first and
 * last streams requested are actually the same stream (under the covers)
 * and kernels enqueued on them cannot run concurrently.
 *
 * The third pool is the "high priority" streams. The third pool acts like
 * the second pool except the streams are created with a higher priority.
 *
 * These pools suggest that stream users should prefer many short-lived streams,
 * as the cost of acquiring and releasing streams is effectively zero. If
 * many longer-lived streams are required in performance critical scenarios
 * then the functionality here may need to be extended to allow, for example,
 * "reserving" a subset of the pool so that other streams do not accidentally
 * overlap the performance critical streams.
 *
 * Note: although the notion of "current stream for device" is thread local
 * (every OS thread has a separate current stream, as one might expect),
 * the stream pool is global across all threads; stream 0 is always stream 0
 * no matter which thread you use it on.  Multiple threads can synchronize
 * on the same stream.  Although the GCU documentation is not very clear
 * on the matter, streams are thread safe; e.g., it is safe to enqueue
 * a kernel on the same stream from two different threads.
 */

namespace torch_gcu {

static constexpr int max_compile_time_stream_priorities = 4;

// Value object representing a GCU stream.  This is just a wrapper
// around c10::Stream, but it comes with a little extra GCU-specific
// functionality (conversion to topsStream_t), and a guarantee that
// the wrapped c10::Stream really is a GCU stream.
class TORCH_GCU_API GCUStream {
 public:
  enum Unchecked { UNCHECKED };

  // Construct a GCUStream from a Stream.  This construction is checked,
  // and will raise an error if the Stream is not, in fact, a GCU stream.
  explicit GCUStream(c10::Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == c10::DeviceType::PrivateUse1);
  }

  // Construct a GCUStream from a Stream with no error checking.
  // This constructor uses the "named" constructor idiom, and can
  // be invoked as: GCUStream(GCUStream::UNCHECKED, stream)
  explicit GCUStream(Unchecked, c10::Stream stream) : stream_(stream) {}

  bool operator==(const GCUStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const GCUStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  // Implicit conversion to topsStream_t.
  operator topsStream_t() const { return stream(); }

  // Implicit conversion to Stream (a.k.a., forget that the stream is a
  // GCU stream).
  operator c10::Stream() const { return unwrap(); }

  // Used to avoid baking in device type explicitly to Python-side API.
  c10::DeviceType device_type() const { return c10::DeviceType::PrivateUse1; }

  // Get the GCU device index that this stream is associated with.
  c10::DeviceIndex device_index() const { return stream_.device_index(); }

  // Get the full Device that this stream is associated with.  The Device
  // is guaranteed to be a GCU device.
  c10::Device device() const {
    return c10::Device(c10::DeviceType::PrivateUse1, device_index());
  }

  // Return the stream ID corresponding to this particular stream.
  c10::StreamId id() const { return stream_.id(); }

  bool query() const;

  // Set cluster_num and sip_num for this stream, success returns True, failure
  // returns False
  bool set_limit(const size_t cluster_num, const size_t sip_num);

  // Return cluster_num and sip_num for this stream, success returns True,
  // failure returns False
  bool get_limit(size_t& cluster_num, size_t& sip_num) const;

  void synchronize() const;

  int priority() const;

  // Explicit conversion to topsStream_t.
  topsStream_t stream() const;

  // Explicit conversion to Stream.
  c10::Stream unwrap() const { return stream_; }

  // Reversibly pack a GCUStream into a struct representation.
  struct c10::StreamData3 pack3() const {
    return stream_.pack3();
  }

  // Unpack a GCUStream from the 3 fields generated by pack().
  static GCUStream unpack3(c10::StreamId stream_id,
                           c10::DeviceIndex device_index,
                           c10::DeviceType device_type) {
    return GCUStream(
        c10::Stream::unpack3(stream_id, device_index, device_type));
  }

  static std::tuple<int, int> priority_range();

 private:
  c10::Stream stream_;
};

/**
 * Get a new stream from the GCU stream pool.  You can think of this
 * as "creating" a new stream, but no such creation actually happens;
 * instead, streams are preallocated from the pool and returned in a
 * round-robin fashion.
 *
 * You can request a stream from the high priority pool by setting
 * isHighPriority to true, or a stream for a specific device by setting device
 * (defaulting to the current GCU stream.)
 */
TORCH_GCU_API GCUStream getStreamFromPool(const bool isHighPriority = false,
                                          c10::DeviceIndex device = -1);
// no default priority to disambiguate overloads
TORCH_GCU_API GCUStream getStreamFromPool(const int priority,
                                          c10::DeviceIndex device = -1);

/**
 * Get a GCUStream from a externally allocated one.
 *
 * This is mainly for interoperability with different libraries where we
 * want to operate on a non-torch allocated stream for data exchange or similar
 * purposes
 */
TORCH_GCU_API GCUStream getStreamFromExternal(topsStream_t ext_stream,
                                              c10::DeviceIndex device_index);

/**
 * Get the default GCU stream, for the passed GCU device, or for the
 * current device if no device index is passed.  The default stream is
 * where most computation occurs when you aren't explicitly using
 * streams.
 */
TORCH_GCU_API GCUStream getDefaultGCUStream(c10::DeviceIndex device_index = -1);

/**
 * Get the current GCU stream, for the passed GCU device, or for the
 * current device if no device index is passed.  The current GCU stream
 * will usually be the default GCU stream for the device, but it may
 * be different if someone called 'setCurrentGCUStream' or used 'StreamGuard'
 * or 'GCUStreamGuard'.
 */
TORCH_GCU_API GCUStream getCurrentGCUStream(c10::DeviceIndex device_index = -1);

/**
 * Set the current stream on the device of the passed in stream to be
 * the passed in stream.  Yes, you read that right: this function
 * has *nothing* to do with the current device: it toggles the current
 * stream of the device of the passed stream.
 *
 * Confused?  Avoid using this function; prefer using 'GCUStreamGuard' instead
 * (which will switch both your current device and current stream in the way you
 * expect, and reset it back to its original state afterwards).
 */
TORCH_GCU_API void setCurrentGCUStream(GCUStream stream);

TORCH_GCU_API std::ostream& operator<<(std::ostream& stream,
                                       const GCUStream& s);

/**
 * Internal Interface
 * Synchronize stream when export ENFLAME_PT_OP_DEBUG_CONFIG="op_sync_mode=true"
 */
TORCH_GCU_API void maybeGCUStreamSynchronize(const GCUStream& stream);

}  // namespace torch_gcu

namespace std {
template <>
struct hash<torch_gcu::GCUStream> {
  size_t operator()(torch_gcu::GCUStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
}  // namespace std
