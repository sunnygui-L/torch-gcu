/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <tops/tops_runtime_api.h>

#include "gcu/logging.h"

// Reference:
// https://github.com/pytorch/pytorch/blob/v1.10.0/c10/cuda/CUDAException.h
// GCU Runtime API CHECK MACRO

namespace torch_gcu {

class GCUError : public c10::Error {
  using c10::Error::Error;
};

}  // namespace torch_gcu

#ifdef STRIP_ERROR_MESSAGES
#define C10_GCU_CHECK(EXPR)                                      \
  do {                                                           \
    topsError_t __err = EXPR;                                    \
    if (__err != topsSuccess) {                                  \
      throw torch_gcu::GCUError(                                 \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
          TORCH_CHECK_MSG(false, ""));                           \
    }                                                            \
  } while (0)
#else
#define C10_GCU_CHECK(EXPR)                                                  \
  do {                                                                       \
    topsError_t __err = (EXPR);                                              \
    if (__err != topsSuccess) {                                              \
      PTCHECK(false) << "GCU error: "                                        \
                     << "(" << #EXPR << ") = " << topsGetErrorString(__err); \
    }                                                                        \
  } while (0)
#endif

#define C10_GCU_CHECK_WARN(EXPR)                              \
  do {                                                        \
    topsError_t __err = EXPR;                                 \
    if (__err != topsSuccess) {                               \
      auto error_unused C10_UNUSED = topsGetLastError();      \
      TORCH_WARN("GCU warning: ", topsGetErrorString(__err)); \
    }                                                         \
  } while (0)

#define C10_GCU_KERNEL_LAUNCH_CHECK() C10_GCU_CHECK(topsGetLastError())

// Intentionally ignore a GCU error
#define C10_GCU_IGNORE_ERROR(EXPR)                              \
  do {                                                          \
    const topsError_t __err = EXPR;                             \
    if (C10_UNLIKELY(__err != topsSuccess)) {                   \
      topsError_t error_unused C10_UNUSED = topsGetLastError(); \
      (void)error_unused;                                       \
    }                                                           \
  } while (0)

// Indicates that a GCU error is handled in a non-standard way
#define C10_GCU_ERROR_HANDLED(EXPR) EXPR

int warning_ingore(bool ingore);

#define IGNORE_OVERRIDE_OPERATOR_WARNING \
  static int _temp_ignore_warning = warning_ingore(true);

#define RESTORE_OVERRIDE_OPERATOR_WARNING \
  static int _temp_restore_warning = warning_ingore(false);

#define AT_GCU_CHECK(EXPR) C10_GCU_CHECK(EXPR)