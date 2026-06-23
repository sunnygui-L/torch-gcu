/*
 * Copyright 2025-2026 Enflame. All Rights Reserved.
 */
#include "python/extra_ops.h"

#include <vector>

#include "aten/interface_ops.h"

namespace torch_gcu {

void RegisterExtraOps(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("_copy", [](const at::Tensor& src, const at::Tensor& dst,
                    const bool& non_blocking) {
    return torch_gcu::interface_ops::_copy_from(src, dst, non_blocking);
  });
}

}  // namespace torch_gcu