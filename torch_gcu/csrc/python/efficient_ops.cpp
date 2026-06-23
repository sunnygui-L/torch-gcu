/*
 * Copyright 2025-2026 Enflame. All Rights Reserved.
 */
#include "python/efficient_ops.h"

#include <vector>

#include "efficient_ops/ops.h"

namespace torch_gcu {

void RegisterEfficientOps(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("_int32_indices_index", [](const at::Tensor& self,
                                   const py::list& indices) {
    c10::List<c10::optional<at::Tensor>> indices_;
    for (const auto& index : indices) {
      if (index.is_none()) {
        indices_.push_back(c10::nullopt);
      } else {
        indices_.push_back(index.cast<at::Tensor>());
      }
    }
    return torch_gcu::efficient::device_int32_indices_index(self, indices_);
  });
}

}  // namespace torch_gcu