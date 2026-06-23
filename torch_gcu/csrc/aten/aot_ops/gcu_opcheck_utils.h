/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/core/IListRef_inl.h>

#include "gcu/gcu_hardware.h"
#include "gcu/gcu_utils.h"

namespace torch_gcu {

template <typename T>
void print_arg(std::stringstream& ss, int64_t idx, const T& var) {
  ss << "arg:" << idx << "\n";
  auto v = at::IValue(var);
  ss << "IValue: \n" << v.tagKind() << ", " << v << "\n";
}

template <>
void print_arg<at::Tensor>(std::stringstream& ss, int64_t idx,
                           const at::Tensor& var);

template <>
void print_arg<c10::optional<at::Tensor>>(
    std::stringstream& ss, int64_t idx,
    const c10::optional<at::Tensor>& opt_tensor);

template <>
void print_arg<std::vector<at::Tensor>>(std::stringstream& ss, int64_t idx,
                                        const std::vector<at::Tensor>& list);

template <>
void print_arg<c10::List<at::Tensor>>(std::stringstream& ss, int64_t idx,
                                      const c10::List<at::Tensor>& list);

template <>
void print_arg<at::TensorList>(std::stringstream& ss, int64_t idx,
                               const at::TensorList& tensor_list);

template <>
void print_arg<c10::List<c10::optional<at::Tensor>>>(
    std::stringstream& ss, int64_t idx,
    const c10::List<c10::optional<at::Tensor>>& list);

template <typename Tuple, size_t... I>
void print_tuple(std::stringstream& ss, int64_t /*idx*/, const Tuple& tuple,
                 std::index_sequence<I...>) {
  static_cast<void>(std::initializer_list<int>{
      (print_arg(ss, int64_t(I), std::get<I>(tuple)), 0)...});
}

template <typename... Args>
void print_arg(std::stringstream& ss, int64_t idx,
               const std::tuple<Args...>& tuple) {
  using tuple_type = std::tuple<Args...>;
  print_tuple(ss, idx, tuple,
              std::make_index_sequence<std::tuple_size_v<tuple_type>>{});
}

void print_args(std::stringstream& ss, int64_t idx);

template <class T, class... Args>
void print_args(std::stringstream& ss, int64_t idx, T first, Args... args) {
  print_arg(ss, idx, first);
  idx++;
  print_args(ss, idx, args...);
}

void dump_tensor(std::string flie_name, const at::Tensor& tensor);

template <typename T>
void dump_arg(std::string path, const T& var, int64_t args_num) {
  std::stringstream ss;
  auto v = at::IValue(var);
  ss << "IValue: \n" << v.tagKind() << ", " << v << "\n";
  std::string file = path + "_args" + std::to_string(args_num) + ".txt";
  SaveStrToFile(ss.str(), file);
}

template <>
void dump_arg<at::Tensor>(std::string path, const at::Tensor& var,
                          int64_t args_num);

template <>
void dump_arg<c10::optional<at::Tensor>>(
    std::string path, const c10::optional<at::Tensor>& opt_tensor,
    int64_t args_num);

template <>
void dump_arg<std::vector<at::Tensor>>(std::string path,
                                       const std::vector<at::Tensor>& list,
                                       int64_t args_num);

template <>
void dump_arg<at::TensorList>(std::string path, const at::TensorList& list,
                              int64_t args_num);

template <>
void dump_arg<c10::List<at::Tensor>>(std::string path,
                                     const c10::List<at::Tensor>& list,
                                     int64_t args_num);

template <>
void dump_arg<at::ITensorListRef>(std::string path,
                                  const at::ITensorListRef& list_ref,
                                  int64_t args_num);

template <>
void dump_arg<c10::List<c10::optional<at::Tensor>>>(
    std::string path, const c10::List<c10::optional<at::Tensor>>& list,
    int64_t args_num);

template <typename Tuple, size_t... I>
void dump_tuple(std::string path, int64_t /*idx*/, const Tuple& tuple,
                std::index_sequence<I...>) {
  static_cast<void>(std::initializer_list<int>{
      (dump_arg(path, std::get<I>(tuple), int64_t(I)), 0)...});
}

template <typename... Args>
void dump_arg(std::string path, const std::tuple<Args...>& tuple, int64_t idx) {
  using tuple_type = std::tuple<Args...>;
  dump_tuple(path, idx, tuple,
             std::make_index_sequence<std::tuple_size_v<tuple_type>>{});
}

void dump_args(std::string path, int idx);

template <class T, class... Args>
void dump_args(std::string path, int idx, T first, Args... args) {
  dump_arg(path, first, idx);
  idx++;
  dump_args(path, idx, args...);
}

template <typename T>
T clone_arg(const T& var) {
  return var;
}

template <>
at::Tensor clone_arg<at::Tensor>(const at::Tensor& var);

template <>
c10::optional<at::Tensor> clone_arg<c10::optional<at::Tensor>>(
    const c10::optional<at::Tensor>& opt_tensor);

template <>
std::vector<at::Tensor> clone_arg<std::vector<at::Tensor>>(
    const std::vector<at::Tensor>& list);

template <>
c10::List<at::Tensor> clone_arg<c10::List<at::Tensor>>(
    const c10::List<at::Tensor>& list);

template <>
c10::List<c10::optional<at::Tensor>>
clone_arg<c10::List<c10::optional<at::Tensor>>>(
    const c10::List<c10::optional<at::Tensor>>& list);

template <typename... Args>
std::tuple<Args...> clone_args(const Args&... args) {
  return std::forward_as_tuple(clone_arg(args)...);
}

std::vector<at::Tensor> listref_to_vector(const at::ITensorListRef& list_ref);
}  // namespace torch_gcu
