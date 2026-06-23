/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */
#include "gcu/gcu_custom_api.h"

#include <mutex>

#include "gcu/gcu_caching_allocator.h"
#include "gcu/gcu_context.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_hooks.h"
#include "gcu/gcu_utils.h"

namespace torch_gcu {

// Get DatePtr from GCU-Tensor
void* getGCUTensorDataPtr(const at::Tensor& input) {
  return gcu_data_ptr(input);
}

namespace {

void GCUAtenInitialize() {
  // for topsaten malloc/free, sync/async
  CHECK_TOPSATEN_CALL(
      topsatenMallocAsyncFuncRegister(&torch_gcu::topsatenMallocAsync));
  CHECK_TOPSATEN_CALL(topsatenMallocFuncRegister(&torch_gcu::topsatenMalloc));
  CHECK_TOPSATEN_CALL(
      topsatenFreeAsyncFuncRegister(&torch_gcu::topsatenFreeAsync));
  CHECK_TOPSATEN_CALL(topsatenFreeFuncRegister(&torch_gcu::topsatenFree));
}

}  // namespace

GCUInitialization::GCUInitialization() {
  static std::once_flag init_gcu_flag;
  std::call_once(init_gcu_flag, []() {
    GCUAtenInitialize();
    detail::RegisterGCUHooks();
    registerGcuAllocator();
  });
}

}  // namespace torch_gcu
