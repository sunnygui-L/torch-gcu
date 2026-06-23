/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <c10/core/Allocator.h>

#include "gcu/gcu_stream.h"

namespace torch_gcu {

// A caching allocator for GCU host allocations (pinned memory).
c10::Allocator* getCachingHostAllocator();

// Records an event in the specified stream.
// The allocation corresponding to the input `ptr`/`ctx` will not be re-used
// until the event has occurred.
bool CachingHostAllocator_recordEvent(void* ptr, void* ctx, GCUStream stream);

// Releases cached pinned memory allocations via topsHostFree
void CachingHostAllocator_emptyCache();

inline at::DataPtr HostAlloc(size_t size) {
  return getCachingHostAllocator()->allocate(size);
}

inline at::Allocator* getPinnedMemoryAllocator() {
  return getCachingHostAllocator();
}

}  // namespace torch_gcu
