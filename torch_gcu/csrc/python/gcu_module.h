/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#ifndef THGP_GCU_MODULE_INC
#define THGP_GCU_MODULE_INC

#include <torch/csrc/THP.h>

#include "gcu/gcu_macros.h"

PyObject* THGPModule_getDevice_wrap(PyObject* self);
PyObject* THGPModule_setDevice_wrap(PyObject* self, PyObject* arg);
PyObject* THGPModule_getDeviceName_wrap(PyObject* self, PyObject* arg);
PyObject* THGPModule_getDriverVersion(PyObject* self);
PyObject* THGPModule_isDriverSufficient(PyObject* self);
// PyObject* THGPModule_getCurrentBlasHandle_wrap(PyObject* self);

TORCH_GCU_API PyMethodDef* THGPModule_get_methods();

namespace torch_gcu {

TORCH_GCU_API void initModule(PyObject* module);

}  // namespace torch_gcu

#endif  // THGP_GCU_MODULE_INC
