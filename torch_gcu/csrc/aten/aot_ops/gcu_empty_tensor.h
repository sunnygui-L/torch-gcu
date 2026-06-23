/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <ATen/DeviceGuard.h>
#include <ATen/EmptyTensor.h>

namespace torch_gcu {

namespace aotops {

at::Tensor empty(at::IntArrayRef size, at::TensorOptions option);

at::Tensor empty_tmp(at::IntArrayRef size, at::TensorOptions option);

at::Tensor empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                         const at::TensorOptions& options);
}  // namespace aotops

}  // namespace torch_gcu
