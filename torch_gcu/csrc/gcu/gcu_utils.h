/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <ATen/Tensor.h>

#include "gcu/gcu_macros.h"
#include "gcu/philox_gcu_state_raw.h"
#include "topsaten/topsaten_define.h"

namespace torch_gcu {

TORCH_GCU_API inline bool is_gcu(const at::Tensor& self) {
  return self.device().is_privateuseone();
}

TORCH_GCU_API inline void warn_type_narrow(const c10::ScalarType& dtype) {
  if (dtype == c10::ScalarType::Double) {
    TORCH_WARN_ONCE(
        "GCU not support ", dtype,
        " use Float replace, maybe lead to unexpected overflow issues.");
  } else if (dtype == c10::ScalarType::Long) {
    TORCH_WARN_ONCE(
        "GCU not support ", dtype,
        " use Int replace, maybe lead to unexpected overflow issues.");
  } else if (dtype == c10::ScalarType::UInt64) {
    TORCH_WARN_ONCE(
        "GCU not support ", dtype,
        " use UInt32 replace, maybe lead to unexpected overflow issues.");
  } else if (dtype == c10::ScalarType::ComplexDouble) {
    TORCH_WARN_ONCE(
        "GCU not support ", dtype,
        " use ComplexFloat replace, maybe lead to unexpected overflow issues.");
  }
}

TORCH_GCU_API inline bool is_cpu_scalar(const at::Tensor& tensor) {
  return tensor.dim() == 0 && tensor.is_cpu();
}

TORCH_GCU_API at::Scalar scalarTensorToScalar(const at::Tensor& tensor);

/**
 * @brief Deal with the return status of TOPSOP API.
 *
 * @param status The return status of TOPSOP API.
 */
TORCH_GCU_API void CHECK_TOPSATEN_CALL(topsatenStatus_t status);

/**
 * @brief Checks the result of a TOPSOP call and throws an exception if it
 * fails.
 *
 * @param status The status of the TOPSOP call.
 * @param op_info A function that returns a string containing information about
 * the operation.
 */
TORCH_GCU_API void CHECK_TOPSATEN_CALL(topsatenStatus_t status,
                                       std::function<std::string()> op_info);

TORCH_GCU_API topsatenTensor createTopsatenTensor(const at::Tensor& tensor);

TORCH_GCU_API topsatenTensor createTopsatenTensor_tmp(const at::Tensor& tensor);

TORCH_GCU_API topsatenTensor
optionalTensorToTopsatenTensor(const c10::optional<at::Tensor>& opt_tensor);

TORCH_GCU_API topsatenScalar_t scalarTensorToTopsatenScalar(
    const at::Tensor& tensor, const c10::ScalarType& scalar_type);

TORCH_GCU_API topsatenTensor
optionalTensorRefToTopsatenTensor(const at::OptionalTensorRef opt_tensor);

TORCH_GCU_API topsatenScalar_t
scalarTensorToTopsatenScalar(const at::Tensor& tensor);

TORCH_GCU_API topsatenDataType_t
scalarTypeToTopsatenDataType(const c10::ScalarType& scalar_type);

TORCH_GCU_API topsatenDataType_t
scalarTypeToTopsatenDataType_tmp(const c10::ScalarType& scalar_type);

TORCH_GCU_API topsatenDataType_t optionalScalarTypeToTopsatenDataType(
    const c10::optional<at::ScalarType>& opt_type);

TORCH_GCU_API topsatenScalar_t scalarToTopsatenScalar(
    const at::Scalar& value, const c10::ScalarType& scalar_type);

TORCH_GCU_API topsatenScalar_t scalarToTopsatenScalar(const at::Scalar& value);

TORCH_GCU_API topsatenScalar_t
optionalScalarToTopsatenScalar(const c10::optional<at::Scalar>& opt_scalar);

TORCH_GCU_API topsatenScalar_t optionalScalarRefToTopsatenScalar(
    const c10::OptionalRef<at::Scalar> opt_scalar);

TORCH_GCU_API topsatenScalar_t
optionalScalarToTopsatenScalar(const c10::optional<int64_t>& opt_int);

TORCH_GCU_API topsatenScalar_t
optionalScalarToTopsatenScalar(const c10::optional<double>& opt_double);

TORCH_GCU_API topsatenScalar_t
optionalScalarToTopsatenScalar(const c10::optional<bool>& opt_bool);

TORCH_GCU_API topsatenSize_t
intArrayRefToTopsatenSize(const at::IntArrayRef& dims);

TORCH_GCU_API topsatenSize_t optionalIntArrayRefToTopsatenSize(
    const at::OptionalIntArrayRef& opt_dims, at::DimVector& dim_vec);

TORCH_GCU_API topsatenGenerator_t
getDefaultTopsatenGenerator(const at::Tensor& self);

TORCH_GCU_API topsatenGenerator_t
generatorToTopsatenGenerator(const at::Generator& generator);

TORCH_GCU_API topsatenPhiloxState_t
philoxStateToTopsatenPhilocState(const PhiloxGcuState& state);

TORCH_GCU_API topsatenGenerator_t optionalGeneratorToTopsatenGenerator(
    const at::Tensor& self, const c10::optional<at::Generator>& generator);

TORCH_GCU_API topsatenMemoryFormat_t
memoryFormatToTopsatenMemoryFormat(const at::MemoryFormat& at_format);

TORCH_GCU_API topsatenMemoryFormat_t optionalMemoryFormatToTopsatenMemoryFormat(
    const c10::optional<at::MemoryFormat>& opt_memory_format);

TORCH_GCU_API std::string storageToString(const at::Storage& storage);

TORCH_GCU_API std::string tensorToString(const at::Tensor& tensor);

TORCH_GCU_API std::string tensorToString(
    const c10::optional<at::Tensor>& tensor);

TORCH_GCU_API std::string tensorVectorToString(
    const std::vector<at::Tensor>& tensors);

template <typename T>
TORCH_GCU_API std::string tensorListToString(const std::vector<T>& tensors) {
  std::stringstream ss;
  for (const auto& t : tensors) {
    ss << tensorToString(t);
  }
  return ss.str();
}

TORCH_GCU_API std::string tensorArgsToString(
    const std::vector<at::Tensor>& ins, const std::vector<at::Tensor>& outs);

/**
 * @brief Save a tensor data to file, can be import in python by
 *        torch.load(file)
 *
 * @param tensor: tensor to be saved
 * @param file: file path to save tensor, if file not exist will create it
 */
TORCH_GCU_API void SaveTensorToFile(const at::Tensor& tensor,
                                    const std::string& file);

TORCH_GCU_API void SaveStrToFile(const std::string& val,
                                 const std::string& file);

TORCH_GCU_API topsatenScatterComputationType_t
getScatterOperatorEnum(const c10::string_view reduce);

/**
 * @brief Convert a topsatenTensor's metadata and device data to a
 *        human-readable string for debugging.
 *
 * Copies device memory to host, then formats shape, dtype, strides
 * and element values into a string.
 *
 * @param xt       The topsatenTensor to inspect.
 * @param name     Optional label prepended to the output.
 * @param max_elems Maximum number of elements to print (0 = all).
 */
TORCH_GCU_API std::string topsatenTensorToString(const topsatenTensor& xt,
                                                 const std::string& name = "",
                                                 int64_t max_elems = 0);

/**
 * @brief Print a topsatenTensor's metadata and data to stderr.
 *
 * Convenience wrapper around topsatenTensorToString for quick
 * debugging from a debugger or temporary code.
 */
TORCH_GCU_API void debugPrintTopsatenTensor(const topsatenTensor& xt,
                                            const std::string& name = "",
                                            int64_t max_elems = 0);

}  // namespace torch_gcu