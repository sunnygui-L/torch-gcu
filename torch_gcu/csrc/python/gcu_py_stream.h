/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#ifndef THGP_STREAM_INC
#define THGP_STREAM_INC

#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>

#include "gcu/gcu_stream.h"

struct THGPStream : THPStream {
  torch_gcu::GCUStream gcu_stream;
};
extern PyObject* THGPStreamClass;

TORCH_GCU_API void THGPStream_init(PyObject* module);

inline bool THGPStream_Check(PyObject* obj) {
  return THGPStreamClass && PyObject_IsInstance(obj, THGPStreamClass);
}

#endif  // THGP_STREAM_INC
