#include "aten/aot_ops/gcu_ops.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_utils.h"

namespace torch_gcu {

namespace aotops {

bool is_set_to(const at::Tensor& self, const at::Tensor& src) {
  // clang-format off
  PTDLOG(OP) << "is_set_to" << ": {\n"
             << tensorArgsToString({self, src}, {})
             << "}\n";
  // clang-format on

  if (self.storage().unsafeGetStorageImpl() ==
          src.storage().unsafeGetStorageImpl() &&
      self.storage_offset() == src.storage_offset() &&
      self.dim() == src.dim()) {
    for (const auto d : c10::irange(self.dim())) {
      if (self.size(d) != src.size(d) || self.stride(d) != src.stride(d)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

}  // namespace aotops

}  // namespace torch_gcu
