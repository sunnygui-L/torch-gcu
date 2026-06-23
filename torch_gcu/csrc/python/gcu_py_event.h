/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */
#ifndef THGP_EVENT_INC
#define THGP_EVENT_INC

#include <torch/csrc/python_headers.h>

#include "gcu/gcu_event.h"
#include "gcu/gcu_macros.h"

struct THGPEvent {
  PyObject_HEAD torch_gcu::GCUEvent gcu_event;
};
extern PyObject* THGPEventClass;

TORCH_GCU_API void THGPEvent_init(PyObject* module);

inline bool THGPEvent_Check(PyObject* obj) {
  return THGPEventClass && PyObject_IsInstance(obj, THGPEventClass);
}

#endif  // THCP_EVENT_INC