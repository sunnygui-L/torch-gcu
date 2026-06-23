/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include "gcu/gcu_guard_impl.h"
#include "gcu/gcu_macros.h"

namespace torch_gcu {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard
// boilerplate]

/// A variant of DeviceGuard that is specialized for GCU.  It accepts
/// integer indices (interpreting them as GCU devices) and is a little
/// more efficient than DeviceGuard (it compiles to straight line
/// topsSetDevice/topsGetDevice calls); however, it can only be used
/// from code that links against GCU directly.

struct TORCH_GCU_API GCUGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit GCUGuard() = delete;

  /// Set the current GCU device to the passed device index.
  explicit GCUGuard(c10::DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current GCU device to the passed device.  Errors if the passed
  /// device is not a GCU device.
  explicit GCUGuard(c10::Device device) : guard_(device) {}

  // Copy is not allowed
  GCUGuard(const GCUGuard&) = delete;
  GCUGuard& operator=(const GCUGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  GCUGuard(GCUGuard&& other) = delete;
  GCUGuard& operator=(GCUGuard&& other) = delete;

  /// Sets the GCU device to the given device.  Errors if the given device
  /// is not a GCU device.
  void set_device(c10::Device device) { guard_.set_device(device); }

  /// Sets the GCU device to the given device.  Errors if the given device
  /// is not a GCU device.  (This method is provided for uniformity with
  /// DeviceGuard).
  void reset_device(c10::Device device) { guard_.reset_device(device); }

  /// Sets the GCU device to the given device index.
  void set_index(c10::DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set upon construction of the guard
  c10::Device original_device() const { return guard_.original_device(); }

  /// Returns the last device that was set via `set_device`, if any, otherwise
  /// the device passed during construction.
  c10::Device current_device() const { return guard_.current_device(); }

 private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<GCUGuardImpl> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for GCU.  See
/// GCUGuard for when you can use this.
struct TORCH_GCU_API OptionalGCUGuard {
  /// Create an uninitialized OptionalGCUGuard.
  explicit OptionalGCUGuard() : guard_() {}

  /// Set the current GCU device to the passed Device, if it is not nullopt.
  explicit OptionalGCUGuard(c10::optional<c10::Device> device_opt)
      : guard_(device_opt) {}

  /// Set the current GCU device to the passed device index, if it is not
  /// nullopt
  explicit OptionalGCUGuard(c10::optional<c10::DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  // Copy is not allowed
  OptionalGCUGuard(const OptionalGCUGuard&) = delete;
  OptionalGCUGuard& operator=(const OptionalGCUGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalGCUGuard(OptionalGCUGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalGCUGuard& operator=(OptionalGCUGuard&& other) = delete;

  /// Sets the GCU device to the given device, initializing the guard if it
  /// is not already initialized.  Errors if the given device is not a GCU
  /// device.
  void set_device(c10::Device device) { guard_.set_device(device); }

  /// Sets the GCU device to the given device, initializing the guard if it is
  /// not already initialized.  Errors if the given device is not a GCU device.
  /// (This method is provided for uniformity with OptionalDeviceGuard).
  void reset_device(c10::Device device) { guard_.reset_device(device); }

  /// Sets the GCU device to the given device index, initializing the guard if
  /// it is not already initialized.
  void set_index(c10::DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set immediately prior to initialization of the
  /// guard, or nullopt if the guard is uninitialized.
  c10::optional<c10::Device> original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  c10::optional<c10::Device> current_device() const {
    return guard_.current_device();
  }

  /// Restore the original GCU device, resetting this guard to uninitialized
  /// state.
  void reset() { guard_.reset(); }

 private:
  c10::impl::InlineOptionalDeviceGuard<GCUGuardImpl> guard_;
};

/// A variant of StreamGuard that is specialized for GCU.  See GCUGuard
/// for when you can use this.
struct TORCH_GCU_API GCUStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit GCUStreamGuard() = delete;

  /// Set the current GCU device to the device associated with the passed
  /// stream, and set the current GCU stream on that device to the passed
  /// stream. Errors if the Stream is not a GCU stream.
  explicit GCUStreamGuard(c10::Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  GCUStreamGuard(const GCUStreamGuard&) = delete;
  GCUStreamGuard& operator=(const GCUStreamGuard&) = delete;

  /// Move is disallowed, as GCUStreamGuard does not have an uninitialized
  /// state, which is required for moves on types with nontrivial destructors.
  GCUStreamGuard(GCUStreamGuard&& other) = delete;
  GCUStreamGuard& operator=(GCUStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a GCU stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on GCU, use GCUMultiStreamGuard instead.
  void reset_stream(c10::Stream stream) { guard_.reset_stream(stream); }

  /// Returns the GCU stream that was set at the time the guard was
  /// constructed.
  GCUStream original_stream() const {
    return GCUStream(GCUStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent GCU stream that was set using this device guard,
  /// either from construction, or via set_stream.
  GCUStream current_stream() const {
    return GCUStream(GCUStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent GCU device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  c10::Device current_device() const { return guard_.current_device(); }

  /// Returns the GCU device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  c10::Device original_device() const { return guard_.original_device(); }

 private:
  c10::impl::InlineStreamGuard<GCUGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for GCU.  See
/// GCUGuard for when you can use this.
struct TORCH_GCU_API OptionalGCUStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalGCUStreamGuard() : guard_() {}

  /// Set the current GCU device to the device associated with the passed
  /// stream, and set the current GCU stream on that device to the passed
  /// stream. Errors if the Stream is not a GCU stream.
  explicit OptionalGCUStreamGuard(c10::Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit OptionalGCUStreamGuard(c10::optional<c10::Stream> stream_opt)
      : guard_(stream_opt) {}

  /// Copy is disallowed
  OptionalGCUStreamGuard(const OptionalGCUStreamGuard&) = delete;
  OptionalGCUStreamGuard& operator=(const OptionalGCUStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalGCUStreamGuard(OptionalGCUStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalGCUStreamGuard& operator=(OptionalGCUStreamGuard&& other) = delete;

  /// Resets the currently set GCU stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the guard if it was not previously initialized.
  void reset_stream(c10::Stream stream) { guard_.reset_stream(stream); }

  /// Returns the GCU stream that was set at the time the guard was most
  /// recently initialized, or nullopt if the guard is uninitialized.
  c10::optional<GCUStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return c10::make_optional(GCUStream(GCUStream::UNCHECKED, r.value()));
    } else {
      return c10::nullopt;
    }
  }

  /// Returns the most recent GCU stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is
  /// initialized, or nullopt if the guard is uninitialized.
  c10::optional<GCUStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return c10::make_optional(GCUStream(GCUStream::UNCHECKED, r.value()));
    } else {
      return c10::nullopt;
    }
  }

  /// Restore the original GCU device and stream, resetting this guard to
  /// uninitialized state.
  void reset() { guard_.reset(); }

 private:
  c10::impl::InlineOptionalStreamGuard<GCUGuardImpl> guard_;
};

/// A variant of MultiStreamGuard that is specialized for GCU.
struct TORCH_GCU_API GCUMultiStreamGuard {
  explicit GCUMultiStreamGuard(at::ArrayRef<GCUStream> streams)
      : guard_(unwrapStreams(streams)) {}

  /// Copy is disallowed
  GCUMultiStreamGuard(const GCUMultiStreamGuard&) = delete;
  GCUMultiStreamGuard& operator=(const GCUMultiStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  GCUMultiStreamGuard(GCUMultiStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  GCUMultiStreamGuard& operator=(GCUMultiStreamGuard&& other) = delete;

 private:
  c10::impl::InlineMultiStreamGuard<GCUGuardImpl> guard_;

  static std::vector<c10::Stream> unwrapStreams(
      at::ArrayRef<GCUStream> topsStreams) {
    std::vector<c10::Stream> streams;
    streams.reserve(topsStreams.size());
    for (const GCUStream& topsStream : topsStreams) {
      streams.push_back(topsStream);
    }
    return streams;
  }
};

}  // namespace torch_gcu
