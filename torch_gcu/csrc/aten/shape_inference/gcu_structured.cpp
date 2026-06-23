/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include "aten/shape_inference/gcu_structured.h"

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/aot_ops/gcu_resize.h"
#include "gcu/gcu_guard.h"

#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

at::Tensor create_out(at::IntArrayRef sizes, at::IntArrayRef strides,
                      const at::TensorOptions &options) {
  if (strides.empty()) {
    return empty(sizes, options);
  } else {
    return empty_strided(sizes, strides, options);
  }
}

c10::optional<at::Tensor> maybe_create_proxy(const at::Tensor &out,
                                             at::IntArrayRef sizes,
                                             at::IntArrayRef strides,
                                             const at::TensorOptions &options) {
  if (out.strides() != strides) {
    return empty_strided(sizes, strides, options);
  }
  return c10::nullopt;
}

void check_inplace(const at::Tensor &self, at::IntArrayRef sizes,
                   const at::TensorOptions &options) {
  // These checks are needed on those operators that:
  //   1) don't use 'TensorIterator' (e.g. 'addmm' and 'baddbmm')
  //   2) have particular typing rules (e.g. 'cumsum' and 'cumprod')
  // For other operators (e.g. 'add'), 'TensorIterator' already checks
  // these things separately.
  TORCH_CHECK(options.dtype() == self.dtype(),
              "Bad in-place call: ", "input tensor dtype ", self.dtype(),
              " and output tensor dtype ", options.dtype(), " should match");
  TORCH_CHECK(options.device() == self.device(),
              "Bad in-place call: ", "input tensor device ", self.device(),
              " and output tensor device ", options.device(), " should match");
  TORCH_CHECK(sizes == self.sizes(),
              "Bad in-place call: ", "input tensor size ", self.sizes(),
              " and output tensor size ", sizes, " should match");
}

at::Tensor clone_shape_infer(
    const at::Tensor &src,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format =
      optional_memory_format.value_or(at::MemoryFormat::Preserve);
  at::Tensor self;
  if (memory_format == at::MemoryFormat::Preserve) {
    if (src.is_non_overlapping_and_dense()) {
      // Copy all strides, this is marginally faster than calling empty_like
      self = at::empty_strided_symint(src.sym_sizes(), src.sym_strides(),
                                      src.options());
    } else {
      self = at::empty_like(src);
    }
  } else {
    self = at::empty_like(src, src.options(), memory_format);
  }

  return self;
}

at::Tensor contiguous_shape_infer(const at::Tensor &self,
                                  at::MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }
  TORCH_CHECK(
      memory_format != at::MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");

  return clone_shape_infer(self, memory_format);
}

}  // namespace aotops

}  // namespace torch_gcu