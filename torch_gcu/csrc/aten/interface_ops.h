#pragma once

#include <ATen/ATen.h>

#include "gcu/gcu_macros.h"

namespace torch_gcu {
namespace interface_ops {
TORCH_GCU_API at::Tensor _copy_from(const at::Tensor& src,
                                    const at::Tensor& dst, bool non_blocking);
}
}  // namespace torch_gcu
