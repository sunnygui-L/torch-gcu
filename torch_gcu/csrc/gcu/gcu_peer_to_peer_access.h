/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <c10/macros/Macros.h>

#include <cstdint>

namespace torch_gcu {

namespace detail {

void init_p2p_access_cache(int64_t num_devices);

}  // namespace detail

bool get_p2p_access(int source_dev, int dest_dev);

}  // namespace torch_gcu
