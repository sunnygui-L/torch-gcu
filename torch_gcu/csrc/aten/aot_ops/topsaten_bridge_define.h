/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <ATen/ATen.h>

#include <tuple>

#include "aten/aot_ops/gcu_op_string.h"
#include "gcu/gcu_functions.h"
#include "gcu/gcu_stream.h"
#include "gcu/philox_gcu_state_raw.h"
#include "topsaten/topsaten_cnn.h"
#include "topsaten/topsaten_define.h"
#include "topsaten/topsaten_ops.h"

namespace torch_gcu {

template <typename T>
struct topsaten_out_variable {
  topsaten_out_variable(T& var) { value = &var; }

  T* value;
};

template <>
struct topsaten_out_variable<at::Tensor> {
  topsaten_out_variable(const at::Tensor& tensor) {
    owned_value = createTopsatenTensor(tensor);
    value = &owned_value;
  }

  topsatenTensor owned_value;
  topsatenTensor* value;
};

template <>
struct topsaten_out_variable<std::vector<at::Tensor>> {
  topsaten_out_variable(const std::vector<at::Tensor>& tensor_vec) {
    for (const at::Tensor& tensor : tensor_vec) {
      owned_value.push_back(createTopsatenTensor(tensor));
    }
    value = &owned_value;
  }

  std::vector<topsatenTensor> owned_value;
  std::vector<topsatenTensor>* value;
};

template <>
struct topsaten_out_variable<at::TensorList> {
  topsaten_out_variable(const at::TensorList& tensor_list) {
    auto tensor_vec = tensor_list.vec();
    for (const auto& var : tensor_vec) {
      owned_value.push_back(optionalTensorToTopsatenTensor(var));
    }
    value = &owned_value;
  }

  std::vector<topsatenTensor> owned_value;
  std::vector<topsatenTensor>* value;
};

template <typename T>
struct topsaten_variable {
  topsaten_variable(const T& var) { value = var; }

  T value;
};

template <>
struct topsaten_variable<at::Tensor> {
  topsaten_variable(const at::Tensor& tensor) {
    value = createTopsatenTensor(tensor);
  }

  topsatenTensor value;
};

template <>
struct topsaten_variable<c10::optional<at::Tensor>> {
  topsaten_variable(const c10::optional<at::Tensor>& opt_tensor) {
    value = optionalTensorToTopsatenTensor(opt_tensor);
  }

  topsatenTensor value;
};

template <>
struct topsaten_variable<at::OptionalTensorRef> {
  topsaten_variable(const at::OptionalTensorRef opt_tensor) {
    value = optionalTensorRefToTopsatenTensor(opt_tensor);
  }

  topsatenTensor value;
};

template <>
struct topsaten_variable<at::Scalar> {
  topsaten_variable(const at::Scalar& scalar) {
    value = scalarToTopsatenScalar(scalar);
  }

  topsatenScalar_t value;
};

template <>
struct topsaten_variable<c10::optional<at::Scalar>> {
  topsaten_variable(const c10::optional<at::Scalar>& opt_scalar) {
    value = optionalScalarToTopsatenScalar(opt_scalar);
  }

  topsatenScalar_t value;
};

template <>
struct topsaten_variable<c10::OptionalRef<at::Scalar>> {
  topsaten_variable(const c10::OptionalRef<at::Scalar> opt_scalar) {
    value = optionalScalarRefToTopsatenScalar(opt_scalar);
  }

  topsatenScalar_t value;
};

template <>
struct topsaten_variable<c10::optional<int64_t>> {
  topsaten_variable(const c10::optional<int64_t>& opt_int) {
    value = optionalScalarToTopsatenScalar(opt_int);
  }

  topsatenScalar_t value;
};

template <>
struct topsaten_variable<c10::optional<double>> {
  topsaten_variable(const c10::optional<double>& opt_double) {
    value = optionalScalarToTopsatenScalar(opt_double);
  }

  topsatenScalar_t value;
};

template <>
struct topsaten_variable<c10::optional<bool>> {
  topsaten_variable(const c10::optional<bool>& opt_bool) {
    value = optionalScalarToTopsatenScalar(opt_bool);
  }

  topsatenScalar_t value;
};

template <>
struct topsaten_variable<c10::string_view> {
  topsaten_variable(const c10::string_view& approximate) {
    value = approximate.cbegin();
  }

  const char* value = nullptr;
};

template <>
struct topsaten_variable<c10::optional<c10::string_view>> {
  topsaten_variable(const c10::optional<c10::string_view>& opt_string) {
    if (opt_string.has_value()) {
      value = opt_string.value().cbegin();
    }
  }

  const char* value = nullptr;
};

template <>
struct topsaten_variable<c10::ScalarType> {
  topsaten_variable(const c10::ScalarType& scalar_type) {
    value = scalarTypeToTopsatenDataType(scalar_type);
  }

  topsatenDataType_t value;
};

template <>
struct topsaten_variable<c10::optional<at::ScalarType>> {
  topsaten_variable(const c10::optional<at::ScalarType>& opt_type) {
    value = optionalScalarTypeToTopsatenDataType(opt_type);
  }

  topsatenDataType_t value;
};

template <>
struct topsaten_variable<at::IntArrayRef> {
  topsaten_variable(const at::IntArrayRef& array) {
    value = intArrayRefToTopsatenSize(array);
  }

  topsatenSize_t value;
};

template <>
struct topsaten_variable<at::OptionalIntArrayRef> {
  topsaten_variable(const at::OptionalIntArrayRef& opt_array) {
    value = optionalIntArrayRefToTopsatenSize(opt_array, dim_vec);
  }

  at::DimVector dim_vec;
  topsatenSize_t value;
};

template <typename T>
struct topsaten_variable<at::ArrayRef<T>> {
  topsaten_variable(const at::ArrayRef<T>& array) { value = array.vec(); }

  std::vector<T> value;
};

template <typename T>
struct topsaten_variable<c10::optional<at::ArrayRef<T>>> {
  topsaten_variable(const c10::optional<at::ArrayRef<T>>& opt_array) {
    if (opt_array.has_value()) {
      value = opt_array.value().vec();
    }
  }

  std::vector<T> value;
};

template <>
struct topsaten_variable<c10::List<c10::optional<at::Tensor>>> {
  topsaten_variable(const c10::List<c10::optional<at::Tensor>>& list) {
    for (const c10::optional<at::Tensor>& var : list) {
      value.push_back(optionalTensorToTopsatenTensor(var));
    }
  }

  std::vector<topsatenTensor> value;
};

template <>
struct topsaten_variable<at::ITensorListRef> {
  topsaten_variable(const at::ITensorListRef& list) {
    auto materialized = list.materialize();
    for (const auto& var : materialized) {
      value.push_back(optionalTensorToTopsatenTensor(var));
    }
  }

  std::vector<topsatenTensor> value;
};

template <>
struct topsaten_variable<at::TensorList> {
  topsaten_variable(const at::TensorList& tensor_list) {
    auto tensor_vec = tensor_list.vec();
    for (const auto& var : tensor_vec) {
      value.push_back(optionalTensorToTopsatenTensor(var));
    }
  }

  std::vector<topsatenTensor> value;
};

template <>
struct topsaten_variable<std::vector<at::Tensor>> {
  topsaten_variable(const std::vector<at::Tensor>& tensor_vec) {
    for (const auto& var : tensor_vec) {
      value.push_back(optionalTensorToTopsatenTensor(var));
    }
  }

  std::vector<topsatenTensor> value;
};

template <>
struct topsaten_variable<at::ArrayRef<at::Scalar>> {
  topsaten_variable(const at::ArrayRef<at::Scalar>& scalar_list) {
    auto scalar_vec = scalar_list.vec();
    for (const auto& var : scalar_vec) {
      value.push_back(scalarToTopsatenScalar(var));
    }
  }

  std::vector<topsatenScalar_t> value;
};

template <>
struct topsaten_variable<at::MemoryFormat> {
  topsaten_variable(const at::MemoryFormat& memory_format) {
    value = memoryFormatToTopsatenMemoryFormat(memory_format);
  }

  topsatenMemoryFormat_t value;
};

template <>
struct topsaten_variable<c10::optional<at::MemoryFormat>> {
  topsaten_variable(const c10::optional<at::MemoryFormat>& opt_memory_format) {
    value = optionalMemoryFormatToTopsatenMemoryFormat(opt_memory_format);
  }

  topsatenMemoryFormat_t value;
};

template <>
struct topsaten_variable<at::Generator> {
  topsaten_variable(const at::Generator& generator) {
    value = generatorToTopsatenGenerator(generator);
  }

  topsatenGenerator_t value;
};

template <>
struct topsaten_variable<PhiloxGcuState> {
  topsaten_variable(const PhiloxGcuState& state) {
    value = philoxStateToTopsatenPhilocState(state);
  }

  topsatenPhiloxState_t value;
};

#define DEFINE_BRIDGE_TOPSATENOP_WITH_NAMESPACE(topsatenop, namespace)   \
  template <typename T, typename... Args>                                \
  void bridge_##topsatenop##_out1(const T& out, const Args&... inputs) { \
    auto stream = getCurrentGCUStream();                                 \
    auto op_info = [&]() -> std::string {                                \
      return get_op_info(#topsatenop, out, inputs..., stream);           \
    };                                                                   \
    PTDLOG(OP) << op_info();                                             \
    auto xout = topsaten_variable(out);                                  \
    topsatenStatus_t status = namespace ::topsatenop(                    \
        xout.value, topsaten_variable<Args>(inputs).value..., stream);   \
    CHECK_TOPSATEN_CALL(status, op_info);                                \
    maybeGCUStreamSynchronize(stream);                                   \
  }

#define DEFINE_BRIDGE_TOPSATENOP(topsatenop) \
  DEFINE_BRIDGE_TOPSATENOP_WITH_NAMESPACE(topsatenop, topsaten)

#define DEFINE_BRIDGE_TOPSEXATSATENOP_OUT2_WITH_NAMESPACE(topsatenop,       \
                                                          namespace)        \
  template <typename T1, typename T2, typename... Args>                     \
  void bridge_##topsatenop##_out2(const T1& out1, const T2& out2,           \
                                  const Args&... inputs) {                  \
    auto stream = getCurrentGCUStream();                                    \
    auto op_info = [&]() -> std::string {                                   \
      return get_op_info_out2(#topsatenop, out1, out2, inputs..., stream);  \
    };                                                                      \
    PTDLOG(OP) << op_info();                                                \
    auto xout1 = topsaten_variable(out1);                                   \
    auto xout2 = topsaten_variable(out2);                                   \
    topsatenStatus_t status = namespace ::topsatenop(                       \
        xout1.value, xout2.value, topsaten_variable<Args>(inputs).value..., \
        stream);                                                            \
    CHECK_TOPSATEN_CALL(status, op_info);                                   \
    maybeGCUStreamSynchronize(stream);                                      \
  }

#define DEFINE_BRIDGE_TOPSATENOP_OUT2(topsatenop) \
  DEFINE_BRIDGE_TOPSEXATSATENOP_OUT2_WITH_NAMESPACE(topsatenop, topsaten)

#define DEFINE_BRIDGE_TOPSATENOP_OUT3(topsatenop)                          \
  template <typename T1, typename T2, typename T3, typename... Args>       \
  void bridge_##topsatenop##_out3(const T1& out1, const T2& out2,          \
                                  const T3& out3, const Args&... inputs) { \
    auto stream = getCurrentGCUStream();                                   \
    auto op_info = [&]() -> std::string {                                  \
      return get_op_info_out3(#topsatenop, out1, out2, out3, inputs...,    \
                              stream);                                     \
    };                                                                     \
    PTDLOG(OP) << op_info();                                               \
    auto xout1 = topsaten_variable(out1);                                  \
    auto xout2 = topsaten_variable(out2);                                  \
    auto xout3 = topsaten_variable(out3);                                  \
    topsatenStatus_t status = topsaten::topsatenop(                        \
        xout1.value, xout2.value, xout3.value,                             \
        topsaten_variable<Args>(inputs).value..., stream);                 \
    CHECK_TOPSATEN_CALL(status, op_info);                                  \
    maybeGCUStreamSynchronize(stream);                                     \
  }

#define DEFINE_BRIDGE_TOPSATENOP_OUT4(topsatenop)                             \
  template <typename T1, typename T2, typename T3, typename T4,               \
            typename... Args>                                                 \
  void bridge_##topsatenop##_out4(const T1& out1, const T2& out2,             \
                                  const T3& out3, const T4& out4,             \
                                  const Args&... inputs) {                    \
    auto stream = getCurrentGCUStream();                                      \
    auto op_info = [&]() -> std::string {                                     \
      return get_op_info_out4(#topsatenop, out1, out2, out3, out4, inputs..., \
                              stream);                                        \
    };                                                                        \
    PTDLOG(OP) << op_info();                                                  \
    auto xout1 = topsaten_variable(out1);                                     \
    auto xout2 = topsaten_variable(out2);                                     \
    auto xout3 = topsaten_variable(out3);                                     \
    auto xout4 = topsaten_variable(out4);                                     \
    topsatenStatus_t status = topsaten::topsatenop(                           \
        xout1.value, xout2.value, xout3.value, xout4.value,                   \
        topsaten_variable<Args>(inputs).value..., stream);                    \
    CHECK_TOPSATEN_CALL(status, op_info);                                     \
    maybeGCUStreamSynchronize(stream);                                        \
  }

#define DEFINE_BRIDGE_TOPSATENOP_MULTI_OUT(topsatenop)                         \
  template <typename... OutArgs, typename... Args, size_t... I>                \
  void bridge_##topsatenop##_multi_out(std::index_sequence<I...>, bool sync,   \
                                       const std::tuple<OutArgs&...>& outputs, \
                                       const Args&... inputs) {                \
    auto stream = getCurrentGCUStream();                                       \
    auto op_info = [&]() -> std::string {                                      \
      return get_op_info_multi_output(#topsatenop, outputs, inputs...,         \
                                      stream);                                 \
    };                                                                         \
    PTDLOG(OP) << op_info();                                                   \
    topsatenStatus_t status = topsaten::topsatenop(                            \
        std::forward_as_tuple(                                                 \
            *topsaten_out_variable<OutArgs>(std::get<I>(outputs)).value...),   \
        topsaten_variable<Args>(inputs).value..., stream);                     \
    CHECK_TOPSATEN_CALL(status, op_info);                                      \
    if (sync) {                                                                \
      torch_gcu::stream_synchronize(stream);                                   \
    } else {                                                                   \
      maybeGCUStreamSynchronize(stream);                                       \
    }                                                                          \
  }

}  // namespace torch_gcu
