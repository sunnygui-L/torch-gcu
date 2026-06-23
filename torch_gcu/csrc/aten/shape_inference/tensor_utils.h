/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

// #include <c10/util/ArrayRef.h>
#include <ATen/TensorUtils.h>

namespace torch_gcu {

void checkSameGCU(at::CheckedFrom c, const at::TensorArg& t1,
                  const at::TensorArg& t2);

void checkAllSameGCU(at::CheckedFrom c, at::ArrayRef<at::TensorArg> tensors);

}  // namespace torch_gcu
