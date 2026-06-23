/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <mutex>

#include "gcu/gcu_macros.h"

namespace torch_gcu {

TORCH_GCU_API const char* get_gcu_check_suffix() noexcept;

TORCH_GCU_API std::mutex* getFreeMutex();

}  // namespace torch_gcu