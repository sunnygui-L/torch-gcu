/*
 * Copyright 2020-2023 Enflame. All Rights Reserved.
 */
#include "python/torch_util.h"

#include <Python.h>
#include <frameobject.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>

#include "gcu/util.h"
#include "python/env.h"

// reference to torch2.10.0 torch/csrc/utils/object_ptr.cpp
template <>
TORCH_PYTHON_API void THPPointer<PyCodeObject>::free() {
  if (ptr && C10_LIKELY(Py_IsInitialized())) Py_DECREF(ptr);
}

template class THPPointer<PyCodeObject>;

namespace torch_gcu {

at::Tensor CopyTensor(const at::Tensor& ref) {
  return ref.to(ref.options(), /*non_blocking=*/false, /*copy=*/true);
}

// Same as above, with an additional cast.
at::Tensor CopyTensor(const at::Tensor& ref, at::ScalarType dest_type,
                      bool copy) {
  return ref.to(ref.options().dtype(dest_type), /*non_blocking=*/false, copy);
}

at::ScalarType GetScalarType(at::Scalar scalar) {
  if (scalar.isFloatingPoint()) {
    return at::kDouble;
  } else if (scalar.isIntegral(/*includeBool=*/false)) {
    return at::kLong;
  } else if (scalar.isBoolean()) {
    return at::kBool;
  } else if (scalar.isComplex()) {
    return at::kComplexDouble;
  }
  PTCHECK(0) << "Unknown type for scalar";
  return at::ScalarType::Undefined;
}

at::Tensor UnwrapNumber(const at::Tensor& tensor, at::ScalarType dtype) {
  return tensor.unsafeGetTensorImpl()->is_wrapped_number() ? tensor.to(dtype)
                                                           : tensor;
}

bool IsScalar(const at::Tensor& t) { return (t.dim() == 0 && t.numel() == 1); }

}  // namespace torch_gcu
