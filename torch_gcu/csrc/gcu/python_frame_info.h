/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#pragma once

#include <string>

#include "gcu/gcu_macros.h"

namespace torch_gcu {

typedef std::string (*PythonCallstackFn)();

TORCH_GCU_API void SetPythonCallstackFn(PythonCallstackFn f);

TORCH_GCU_API std::string GetPythonFrame();

}  // namespace torch_gcu
