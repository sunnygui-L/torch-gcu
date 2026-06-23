#pragma once
/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#include <ATen/Device.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>

#include "gcu/gcu_macros.h"

namespace torch_gcu {

TORCH_GCU_API PyObject* THGPDevice_pynew(PyTypeObject* type, PyObject* args,
                                         PyObject* kwargs);

void ReplaceDeviceNewMethod() { THPDeviceType.tp_new = THGPDevice_pynew; }

TORCH_GCU_API PyObject* THGPGenerator_pynew(PyTypeObject* type, PyObject* args,
                                            PyObject* kwargs);

void ReplaceGeneratorNewMethod() {
  auto th_generator_class_pt =
      reinterpret_cast<PyTypeObject*>(THPGeneratorClass);
  th_generator_class_pt->tp_new = THGPGenerator_pynew;
}

}  // namespace torch_gcu