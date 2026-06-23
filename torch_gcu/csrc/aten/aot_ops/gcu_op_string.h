/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <ATen/ATen.h>

#include <iterator>
#include <vector>

#include "gcu/gcu_hardware.h"
#include "gcu/gcu_stream.h"
#include "gcu/gcu_utils.h"
#include "gcu/philox_gcu_state_raw.h"

namespace torch_gcu {

namespace {

template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& arr) {
  std::copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(os, ", "));
  return os;
}

}  // namespace

template <typename T>
inline std::string get_variable_info(const T& var, const std::string& name) {
  std::stringstream ss;
  ss << "{\n" << var << "\n}\n";
  return name + ": " + ss.str() + "\n";
}

// topsatenGenerator_t
// topsatenScalar_t
// at::ScalarType
// at::MemoryFormat
// GCUStream
template <typename T>
struct variable_info {
  variable_info(const T& var, const std::string& name) {
    info = get_variable_info(var, name);
  }

  std::string info;
};

template <>
struct variable_info<at::Tensor> {
  variable_info(const at::Tensor& tensor, const std::string& name) {
    info = name + ": " + tensorToString(tensor) + "\n";
  }

  std::string info;
};

template <>
struct variable_info<c10::optional<at::Tensor>> {
  variable_info(const c10::optional<at::Tensor>& opt_tensor,
                const std::string& name) {
    info = name + ": " + tensorToString(opt_tensor) + "\n";
  }

  std::string info;
};

template <>
struct variable_info<at::OptionalTensorRef> {
  variable_info(const at::OptionalTensorRef opt_tensor,
                const std::string& name) {
    info = name + ": " + tensorToString(opt_tensor.getTensorRef()) + "\n";
  }

  std::string info;
};

template <>
struct variable_info<at::Scalar> {
  variable_info(const at::Scalar& scalar, const std::string& name) {
    auto xscalar = scalarToTopsatenScalar(scalar);
    info = get_variable_info(xscalar, name);
  }

  std::string info;
};

template <>
struct variable_info<c10::optional<at::Scalar>> {
  variable_info(const c10::optional<at::Scalar>& opt_scalar,
                const std::string& name) {
    auto xscalar = optionalScalarToTopsatenScalar(opt_scalar);
    info = get_variable_info(xscalar, name);
  }

  std::string info;
};

template <>
struct variable_info<c10::OptionalRef<at::Scalar>> {
  variable_info(const c10::OptionalRef<at::Scalar> opt_scalar,
                const std::string& name) {
    auto xscalar = optionalScalarRefToTopsatenScalar(opt_scalar);
    info = get_variable_info(xscalar, name);
  }

  std::string info;
};

template <>
struct variable_info<c10::optional<int64_t>> {
  variable_info(const c10::optional<int64_t>& opt_int,
                const std::string& name) {
    auto xscalar = optionalScalarToTopsatenScalar(opt_int);
    info = get_variable_info(xscalar, name);
  }

  std::string info;
};

template <>
struct variable_info<c10::optional<double>> {
  variable_info(const c10::optional<double>& opt_double,
                const std::string& name) {
    auto xscalar = optionalScalarToTopsatenScalar(opt_double);
    info = get_variable_info(xscalar, name);
  }

  std::string info;
};

template <>
struct variable_info<c10::optional<bool>> {
  variable_info(const c10::optional<bool>& opt_bool, const std::string& name) {
    auto xscalar = optionalScalarToTopsatenScalar(opt_bool);
    info = get_variable_info(xscalar, name);
  }

  std::string info;
};

template <>
struct variable_info<c10::string_view> {
  variable_info(const c10::string_view& approximate, const std::string& name) {
    std::stringstream ss;
    ss << std::basic_string_view<char>(approximate.data(), approximate.size());
    info = name + ": " + ss.str() + "\n";
  }

  std::string info;
};

template <>
struct variable_info<c10::optional<c10::string_view>> {
  variable_info(const c10::optional<c10::string_view>& opt_string,
                const std::string& name) {
    if (opt_string.has_value()) {
      std::stringstream ss;
      ss << std::basic_string_view<char>(opt_string.value().data(),
                                         opt_string.value().size());
      info = name + ": " + ss.str() + "\n";
    } else {
      info = name + ": " + "None" + "\n";
    }
  }

  std::string info;
};

template <>
struct variable_info<c10::optional<at::ScalarType>> {
  variable_info(const c10::optional<at::ScalarType>& opt_type,
                const std::string& name) {
    auto xtype = optionalScalarTypeToTopsatenDataType(opt_type);
    info = get_variable_info(xtype, name);
  }

  std::string info;
};

template <>
struct variable_info<at::OptionalIntArrayRef> {
  variable_info(const at::OptionalIntArrayRef& opt_array,
                const std::string& name) {
    auto xarray = optionalIntArrayRefToTopsatenSize(opt_array, dim_vec);
    info = get_variable_info(xarray, name);
  }

  at::DimVector dim_vec;
  std::string info;
};

template <typename T>
struct variable_info<c10::optional<at::ArrayRef<T>>> {
  variable_info(const c10::optional<at::ArrayRef<T>>& opt_array,
                const std::string& name) {
    std::vector<T> xarray;
    if (opt_array.has_value()) {
      xarray = opt_array.value().vec();
    }
    info = get_variable_info(xarray, name);
  }

  std::string info;
};

template <>
struct variable_info<at::TensorList> {
  variable_info(const at::TensorList& tensor_list, const std::string& name) {
    std::string list_info = "";
    auto tensor_vec = tensor_list.vec();
    for (uint64_t i = 0; i < tensor_vec.size(); i++) {
      std::string var_name = "TensorList value " + std::to_string(i);
      list_info += var_name + ": " + tensorToString(tensor_vec[i]) + "\n";
    }
    info = name + ": {\n\n" + list_info + "}\n\n";
  }

  std::string info;
};

template <>
struct variable_info<std::vector<at::Tensor>> {
  variable_info(const std::vector<at::Tensor>& tensor_vec,
                const std::string& name) {
    std::string list_info = "";
    for (uint64_t i = 0; i < tensor_vec.size(); i++) {
      std::string var_name = "TensorList value " + std::to_string(i);
      list_info += var_name + ": " + tensorToString(tensor_vec[i]) + "\n";
    }
    info = name + ": {\n\n" + list_info + "}\n\n";
  }

  std::string info;
};

template <>
struct variable_info<c10::List<c10::optional<at::Tensor>>> {
  variable_info(const c10::List<c10::optional<at::Tensor>>& list,
                const std::string& name) {
    std::string list_info = "";
    for (uint64_t i = 0; i < list.size(); i++) {
      std::string var_name = "list value " + std::to_string(i);
      list_info += var_name + ": " + tensorToString(list[i]) + "\n";
    }
    info = name + ": {\n\n" + list_info + "}\n\n";
  }

  std::string info;
};

template <>
struct variable_info<at::ITensorListRef> {
  variable_info(const at::ITensorListRef& list, const std::string& name) {
    std::string list_info = "";
    auto materialized = list.materialize();
    for (uint64_t i = 0; i < materialized.size(); i++) {
      std::string var_name = "list value " + std::to_string(i);
      list_info +=
          var_name + ": " + tensorToString(materialized[i].get()) + "\n";
    }
    info = name + ": {\n\n" + list_info + "}\n\n";
  }

  std::string info;
};

template <>
struct variable_info<c10::optional<at::MemoryFormat>> {
  variable_info(const c10::optional<at::MemoryFormat>& opt_memory_format,
                const std::string& name) {
    auto xmemory_format =
        optionalMemoryFormatToTopsatenMemoryFormat(opt_memory_format);
    info = get_variable_info(xmemory_format, name);
  }

  std::string info;
};

template <>
struct variable_info<at::Generator> {
  variable_info(const at::Generator& gen, const std::string& name) {
    std::stringstream ss;
    ss << "{\n";
    ss << "Generator's seed is " << gen.current_seed() << ",\n";
    ss << "Generator's offset is " << gen.get_offset() << ".\n";
    ss << "}";
    info = name + ": " + ss.str() + "\n";
  }

  std::string info;
};

template <>
struct variable_info<PhiloxGcuState> {
  variable_info(const PhiloxGcuState& state, const std::string& name) {
    std::stringstream ss;
    ss << "{\n";
    ss << "State's captured is " << state.captured_ << ",\n";
    ss << "State's offset_intragraph is " << state.offset_intragraph_ << ",\n";
    if (state.captured_) {
      ss << "State's seed.ptr is " << state.seed_.ptr << ",\n";
      ss << "State's offset.ptr is " << state.offset_.ptr << ".\n";
    } else {
      ss << "State's seed.val is " << state.seed_.val << ",\n";
      ss << "State's offset.val is " << state.offset_.val << ".\n";
    }
    ss << "}";
    info = name + ": " + ss.str() + "\n";
  }
  std::string info;
};

template <typename T, typename... Args>
inline std::string get_op_info(const std::string& op_name, const T& out,
                               const Args&... inputs) {
  int i = 0;
  std::string var_name_base = "input ";
  auto inputs_info = std::initializer_list<std::string>{
      variable_info<Args>(inputs, var_name_base + std::to_string(i++)).info...};
  std::string op_info = op_name + ": {\n\n";
  op_info += variable_info(out, "output").info;
  for (std::string input_info : inputs_info) {
    op_info += input_info;
  }
  op_info += "}\n";
  return op_info;
}

template <typename... Args, size_t... I>
inline std::string get_multi_output_info(const std::tuple<Args&...>& out,
                                         std::index_sequence<I...>) {
  auto outputs_info = std::initializer_list<std::string>{
      variable_info<Args>(std::get<I>(out), "output " + std::to_string(I))
          .info...};
  std::string out_info = "";

  for (std::string output_info : outputs_info) {
    out_info += output_info;
  }
  return out_info;
}

template <typename... ArgsOut, typename... Args>
inline std::string get_op_info_multi_output(const std::string& op_name,
                                            const std::tuple<ArgsOut&...>& out,
                                            const Args&... inputs) {
  int i = 0;
  std::string var_name_base = "input ";
  auto inputs_info = std::initializer_list<std::string>{
      variable_info<Args>(inputs, var_name_base + std::to_string(i++)).info...};

  auto output_info = get_multi_output_info(
      out,
      std::make_index_sequence<std::tuple_size_v<std::tuple<ArgsOut...>>>{});

  std::string op_info = op_name + ": {\n\n";
  op_info += output_info;
  for (std::string input_info : inputs_info) {
    op_info += input_info;
  }
  op_info += "}\n";
  return op_info;
}

template <typename T1, typename T2, typename... Args>
inline std::string get_op_info_out2(const std::string& op_name, const T1& out1,
                                    const T2& out2, const Args&... inputs) {
  int i = 0;
  std::string var_name_base = "input ";
  auto inputs_info = std::initializer_list<std::string>{
      variable_info<Args>(inputs, var_name_base + std::to_string(i++)).info...};
  std::string op_info = op_name + ": {\n\n";
  op_info += variable_info(out1, "output 0").info;
  op_info += variable_info(out2, "output 1").info;
  for (std::string input_info : inputs_info) {
    op_info += input_info;
  }
  op_info += "}\n";
  return op_info;
}

template <typename T1, typename T2, typename T3, typename... Args>
inline std::string get_op_info_out3(const std::string& op_name, const T1& out1,
                                    const T2& out2, const T3& out3,
                                    const Args&... inputs) {
  int i = 0;
  std::string var_name_base = "input ";
  auto inputs_info = std::initializer_list<std::string>{
      variable_info<Args>(inputs, var_name_base + std::to_string(i++)).info...};
  std::string op_info = op_name + ": {\n\n";
  op_info += variable_info(out1, "output 0").info;
  op_info += variable_info(out2, "output 1").info;
  op_info += variable_info(out3, "output 2").info;
  for (std::string input_info : inputs_info) {
    op_info += input_info;
  }
  op_info += "}\n";
  return op_info;
}

template <typename T1, typename T2, typename T3, typename T4, typename... Args>
inline std::string get_op_info_out4(const std::string& op_name, const T1& out1,
                                    const T2& out2, const T3& out3,
                                    const T4& out4, const Args&... inputs) {
  int i = 0;
  std::string var_name_base = "input ";
  auto inputs_info = std::initializer_list<std::string>{
      variable_info<Args>(inputs, var_name_base + std::to_string(i++)).info...};
  std::string op_info = op_name + ": {\n\n";
  op_info += variable_info(out1, "output 0").info;
  op_info += variable_info(out2, "output 1").info;
  op_info += variable_info(out3, "output 2").info;
  op_info += variable_info(out4, "output 3").info;
  for (std::string input_info : inputs_info) {
    op_info += input_info;
  }
  op_info += "}\n";
  return op_info;
}

}  // namespace torch_gcu
