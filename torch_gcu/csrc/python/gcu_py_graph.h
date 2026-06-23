/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <torch/csrc/python_headers.h>
#include "gcu/gcu_macros.h"

TORCH_GCU_API void THGPGraph_init(PyObject* module);
