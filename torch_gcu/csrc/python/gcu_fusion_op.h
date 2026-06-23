/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 *
 */
#pragma once

#include <Python.h>

namespace torch_gcu {

void RegisterFusionOp(PyObject* module);

}  // namespace torch_gcu