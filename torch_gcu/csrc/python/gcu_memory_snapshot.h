/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 *
 */

// reference from torch/csrc/cuda/memory_snapshot.h

#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>

#include <cstdint>
#include <string>

namespace torch_gcu {

// C++-only versions of these, for python use
// those defined in gcu_module.cpp which also record python state.
void _record_memory_history(bool enabled, bool record_context = true,
                            int64_t trace_alloc_max_entries = 1,
                            bool trace_alloc_record_context = false,
                            bool record_cpp_context = false);

void _record_memory_history(c10::optional<std::string> enabled = "all",
                            c10::optional<std::string> context = "all",
                            std::string stacks = "all",
                            size_t max_entries = UINT64_MAX);

}  // namespace torch_gcu
