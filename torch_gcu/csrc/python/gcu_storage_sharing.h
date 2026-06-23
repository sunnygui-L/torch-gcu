#pragma once
#include <torch/csrc/utils/pybind.h>

#include "gcu/gcu_macros.h"

TORCH_GCU_API PyMethodDef* THGPStorage_getSharingMethods();

void THGPStorage_Sharing_methods(PyObject* module);