/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include <ATen/NamedTensorUtils.h>
#include <ATen/core/NamedTensor.h>

#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_utils.h"
#include "topsaten/topsaten_ops.h"

namespace torch_gcu {

namespace aotops {

bool equal(const at::Tensor& self, const at::Tensor& src) {
  if (!at::namedinference::are_names_equal(self.unsafeGetTensorImpl(),
                                           src.unsafeGetTensorImpl())) {
    return false;
  }
  at::NoNamesGuard guard;
  TORCH_CHECK(self.device() == src.device(),
              "Cannot compare two tensors on "
              "different devices. Got: ",
              self.device(), " and ", src.device());
  if (self.sizes() != src.sizes()) {
    return false;
  }
  if (self.numel() == 0) {
    return true;
  }

  // This is the same optimization done in the cpu_equal. Since the flags like
  // neg/conj should be already handled outside the gcu_equal, it should be safe
  // to have the following fast path by ensuring the storage and strides exactly
  // the same.
  if (self.is_alias_of(src) && self.storage_offset() == src.storage_offset() &&
      self.dtype() == src.dtype() &&
      self.is_contiguous() == src.is_contiguous() &&
      self.strides().equals(src.strides())
      // Extra checks to ensure the safety in case cuda_equal is directly called
      // in C++.
      && self.layout() == src.layout() && self.is_neg() == src.is_neg() &&
      self.is_conj() == src.is_conj()) {
    return true;
  }

  bool result = false;

  auto stream = getCurrentGCUStream(self.get_device());
  auto op_info = [&]() -> std::string {
    std::stringstream ss;
    // clang-format off
    ss << "topsatenEqual"
       << " :\n"
       << tensorArgsToString({self, src}, {})
       << "output: " << "bool " << result << "\n"
       << "stream: " << (topsStream_t)stream << "\n";
    // clang-format on
    return ss.str();
  };
  PTDLOG(OP) << op_info();

  topsatenStatus_t status =
      topsaten::topsatenEqual(result, topsaten_variable(self).value,
                              topsaten_variable(src).value, stream);
  CHECK_TOPSATEN_CALL(status, op_info);
  torch_gcu::stream_synchronize(stream);

  return result;
}

}  // namespace aotops

}  // namespace torch_gcu