/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#pragma once
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/Tensor.h>

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/aot_ops/gcu_resize.h"
#include "gcu/gcu_guard.h"

namespace torch_gcu {

namespace aotops {

at::Tensor create_out(at::IntArrayRef sizes, at::IntArrayRef strides,
                      const at::TensorOptions &options);

c10::optional<at::Tensor> maybe_create_proxy(const at::Tensor &out,
                                             at::IntArrayRef sizes,
                                             at::IntArrayRef strides,
                                             const at::TensorOptions &options);

void check_inplace(const at::Tensor &self, at::IntArrayRef sizes,
                   const at::TensorOptions &options);

at::Tensor clone_shape_infer(
    const at::Tensor &src,
    c10::optional<c10::MemoryFormat> optional_memory_format);

at::Tensor contiguous_shape_infer(
    const at::Tensor &self,
    at::MemoryFormat memory_format = at::MemoryFormat::Contiguous);

}  // namespace aotops

}  // namespace torch_gcu