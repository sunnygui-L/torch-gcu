
/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <ATen/native/Resize.h>
#include <ATen/native/ResizeCommon.h>

#include "gcu/gcu_guard.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_utils.h"

namespace torch_gcu {

namespace aotops {

void resize_bytes(at::StorageImpl* storage, size_t size_bytes,
                  bool is_narrow = false);

static inline void maybe_resize_storage(at::TensorImpl* self,
                                        size_t new_size_bytes,
                                        bool is_narrow = false) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in Resize.h)
  if (self->numel() == 0) {
    return;
  }

  const at::Storage& storage = self->unsafe_storage();
  TORCH_CHECK(storage, "Tensor: invalid null storage");
  if (new_size_bytes > storage.nbytes()) {
    resize_bytes(storage.unsafeGetStorageImpl(), new_size_bytes, is_narrow);
  }
}

inline at::TensorImpl* resize_impl_(at::TensorImpl* self, at::IntArrayRef size,
                                    at::OptionalIntArrayRef stride,
                                    bool device_guard = true) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  // NB: We don't need to hold the device guard when calling from TH
  OptionalGCUGuard guard;
  if (device_guard) {
    guard.set_index(self->storage().device().index());
  }

  auto dtype = self->dtype();
  const auto itemsize = dtype.itemsize();
  const auto storage_offset = self->storage_offset();
  size_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    storage_size = at::detail::computeStorageNbytes(size, *stride, itemsize,
                                                    storage_offset);
  } else {
    self->set_sizes_contiguous(size);
    storage_size = at::detail::computeStorageNbytesContiguous(size, itemsize,
                                                              storage_offset);
  }
  maybe_resize_storage(self, storage_size,
                       is_narrow_type(c10::typeMetaToScalarType(dtype)));

  return self;
}

void resize_out(const at::Tensor& out, c10::IntArrayRef sizes,
                c10::IntArrayRef strides, const c10::TensorOptions& options);

bool resize_output(const at::Tensor& output, at::IntArrayRef shape);

}  // namespace aotops

}  // namespace torch_gcu
