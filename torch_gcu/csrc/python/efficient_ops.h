/*
 * Copyright 2025-2026 Enflame. All Rights Reserved.
 */
#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "gcu/gcu_macros.h"

namespace torch_gcu {

TORCH_GCU_API void RegisterEfficientOps(PyObject *module);

}  // namespace torch_gcu