/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include <ATen/core/TensorBody.h>

namespace torch_gcu {
namespace aotops {
at::Tensor& uniform__shape_infer(at::Tensor& self, double from, double to,
                                 c10::optional<at::Generator> generator);
}  // namespace aotops
}  // namespace torch_gcu
