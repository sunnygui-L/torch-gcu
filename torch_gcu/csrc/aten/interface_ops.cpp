#include "aten/interface_ops.h"

#include "aot_ops/gcu_aot_ops.h"

namespace torch_gcu {
namespace interface_ops {

at::Tensor _copy_from(const at::Tensor& src, const at::Tensor& dst,
                      bool non_blocking) {
  return torch_gcu::aotops::_copy_from(src, dst, non_blocking);
}

}  // namespace interface_ops
}  // namespace torch_gcu