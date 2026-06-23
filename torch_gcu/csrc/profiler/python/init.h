#pragma once

#include <Python.h>
#include <torch_gcu/csrc/profiler/collection.h>
#include <torch_gcu/csrc/profiler/python/pybind.h>

namespace pybind11::detail {
using torch_gcu::profiler::impl::TensorID;

#define STRONG_POINTER_TYPE_CASTER(T) \
  template <>                         \
  struct type_caster<T> : public strong_pointer_type_caster<T> {};

STRONG_POINTER_TYPE_CASTER(torch_gcu::profiler::impl::StorageImplData);
STRONG_POINTER_TYPE_CASTER(torch_gcu::profiler::impl::AllocationID);
STRONG_POINTER_TYPE_CASTER(torch_gcu::profiler::impl::TensorImplAddress);
STRONG_POINTER_TYPE_CASTER(torch_gcu::profiler::impl::PyModuleSelf);
STRONG_POINTER_TYPE_CASTER(torch_gcu::profiler::impl::PyModuleCls);
STRONG_POINTER_TYPE_CASTER(torch_gcu::profiler::impl::PyOptimizerSelf);
#undef STRONG_POINTER_TYPE_CASTER

template <>
struct type_caster<TensorID> : public strong_uint_type_caster<TensorID> {};
}  // namespace pybind11::detail

namespace torch_gcu::profiler {

void initPythonBindings(PyObject* module);

}  // namespace torch_gcu::profiler
