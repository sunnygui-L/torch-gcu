#include <ATen/native/ComplexHelper.h>

#include "aten/aot_ops/gcu_ops.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_utils.h"

namespace torch_gcu {

namespace aotops {

at::Tensor view_as_real(const at::Tensor& self) {
  // clang-format off
  PTDLOG(OP) << "view_as_real" << ": {\n"
             << tensorArgsToString({self}, {})
             << "}\n";
  // clang-format on

  TORCH_CHECK(!self.is_conj(),
              "view_as_real doesn't work on unresolved conjugated tensors.  To "
              "resolve the conjugate tensor so you can view it as real, use "
              "self.resolve_conj(); however, be warned that the resulting "
              "tensor will NOT alias the original.");
  return at::native::_view_as_real_physical(self);
}

at::Tensor view_as_complex(const at::Tensor& self) {
  // clang-format off
  PTDLOG(OP) << "view_as_complex" << ": {\n"
             << tensorArgsToString({self}, {})
             << "}\n";
  // clang-format on

  TORCH_CHECK(self.scalar_type() == at::kFloat ||
                  self.scalar_type() == at::kDouble ||
                  self.scalar_type() == at::kHalf,
              "view_as_complex is only supported for half, float and double "
              "tensors, but got a tensor of scalar type: ",
              self.scalar_type());

  auto old_sizes = self.sym_sizes();
  TORCH_CHECK(!old_sizes.empty(),
              "Input tensor must have one or more dimensions");
  TORCH_CHECK(old_sizes[old_sizes.size() - 1] == 2,
              "Tensor must have a last dimension of size 2");
  at::SymDimVector new_sizes(old_sizes.begin(), old_sizes.end() - 1);

  const auto new_strides =
      at::native::computeStrideForViewAsComplex(self.sym_strides());
  const auto complex_type = c10::toComplexType(self.scalar_type());

  TORCH_CHECK(self.sym_storage_offset() % 2 == 0,
              "Tensor must have a storage_offset divisible by 2");
  const auto new_storage_offset = self.sym_storage_offset() / 2;

  return at::native::view_tensor(self, complex_type, new_storage_offset,
                                 new_sizes, new_strides);
}

}  // namespace aotops

}  // namespace torch_gcu
