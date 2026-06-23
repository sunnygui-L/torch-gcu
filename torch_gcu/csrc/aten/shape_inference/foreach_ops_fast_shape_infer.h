/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include <ATen/Tensor.h>

namespace torch_gcu {

namespace aotops {

std::vector<at::Tensor> foreach_op_fast_shape_infer(at::TensorList input);

void foreach_op__fast_shape_infer(at::TensorList input);

std::vector<at::Tensor> foreach_op_fast_shape_infer(at::TensorList tensors,
                                                    const at::Tensor& scalar);

void foreach_op__fast_shape_infer(at::TensorList tensors,
                                  const at::Tensor& scalar);

}  // namespace aotops

}  // namespace torch_gcu
