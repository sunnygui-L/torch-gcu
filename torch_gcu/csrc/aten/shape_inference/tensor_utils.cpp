#include "aten/shape_inference/tensor_utils.h"

#include <ATen/TensorUtils.h>

namespace torch_gcu {

void checkAllSame(at::CheckedFrom c, at::ArrayRef<at::TensorArg> tensors,
                  void (*fn)(at::CheckedFrom, const at::TensorArg&,
                             const at::TensorArg&)) {
  const at::TensorArg* t0 = nullptr;
  for (auto& t : tensors) {
    if (!t->defined()) continue;
    if (t0 != nullptr) {
      fn(c, *t0, t);
    } else {
      t0 = &t;
    }
  }
}

void checkSameGCU(at::CheckedFrom c, const at::TensorArg& t1,
                  const at::TensorArg& t2) {
  if (t1->is_cpu() || t2->is_cpu()) {
    std::ostringstream oss;
    if (t1->is_cpu()) {
      oss << "Tensor for " << t1 << " is on CPU, ";
    }
    if (t2->is_cpu()) {
      oss << "Tensor for " << t2 << " is on CPU, ";
    }
    oss << "but expected " << ((!t1->is_cpu() && !t2->is_cpu()) ? "them" : "it")
        << " to be on GPU (while checking arguments for " << c << ")";
    AT_ERROR(oss.str());
  }
  TORCH_CHECK(t1->get_device() == t2->get_device(), "Expected tensor for ", t1,
              " to have the same device as tensor for ", t2, "; but device ",
              t1->get_device(), " does not equal ", t2->get_device(),
              " (while checking arguments for ", c, ")");
}

void checkAllSameGCU(at::CheckedFrom c, at::ArrayRef<at::TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameGCU);
}

}  // namespace torch_gcu
