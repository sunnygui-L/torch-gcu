/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#include "aten/aot_ops/gcu_opcheck_utils.h"

namespace torch_gcu {

template <>
void print_arg<at::Tensor>(std::stringstream& ss, int64_t idx,
                           const at::Tensor& var) {
  ss << "arg:" << idx << "\n";
  ss << "at::Tensor: \n" << tensorToString(var) << "\n";
}

template <>
void print_arg<c10::optional<at::Tensor>>(
    std::stringstream& ss, int64_t idx,
    const c10::optional<at::Tensor>& opt_tensor) {
  ss << "arg:" << idx << "\n";
  if (opt_tensor.has_value()) {
    ss << "optional<at::Tensor>: \n" << tensorToString(*opt_tensor) << "\n";
  } else {
    ss << "optional<at::Tensor>: \nc10::nullopt\n";
  }
}

template <>
void print_arg<std::vector<at::Tensor>>(std::stringstream& ss, int64_t idx,
                                        const std::vector<at::Tensor>& list) {
  ss << "arg:" << idx << "\n";
  ss << "at::TensorList: \n{\n"
     << "at::Tensor: \n"
     << tensorListToString(list) << "\n"
     << "}\n";
}

template <>
void print_arg<c10::List<at::Tensor>>(std::stringstream& ss, int64_t idx,
                                      const c10::List<at::Tensor>& list) {
  ss << "arg:" << idx << "\n";
  ss << "c10::List<at::Tensor>: \n{\n"
     << "at::Tensor: \n"
     << tensorListToString(list.vec()) << "\n"
     << "}\n";
}

template <>
void print_arg<at::TensorList>(std::stringstream& ss, int64_t idx,
                               const at::TensorList& tensor_list) {
  ss << "arg:" << idx << "\n";
  ss << "at::TensorList: \n{\n"
     << "at::Tensor: \n"
     << tensorListToString(tensor_list.vec()) << "\n"
     << "}\n";
}

template <>
void print_arg<c10::List<c10::optional<at::Tensor>>>(
    std::stringstream& ss, int64_t idx,
    const c10::List<c10::optional<at::Tensor>>& list) {
  ss << "arg:" << idx << "\n";
  ss << "c10::List<c10::optional<at::Tensor>>: \n{\n";
  for (c10::optional<at::Tensor> var : list) {
    if (var.has_value()) {
      ss << "optional<at::Tensor>: \n" << tensorToString(*var) << "\n";
    } else {
      ss << "optional<at::Tensor>: \nc10::nullopt\n";
    }
  }
  ss << "}\n";
}

void print_args(std::stringstream& ss, int64_t /*idx*/) { ss << std::endl; }

void dump_tensor(std::string flie_name, const at::Tensor& tensor) {
  if (tensor.defined()) {
    std::string full_name = flie_name + ".pt";
    SaveTensorToFile(tensor, full_name);
  } else {
    std::string val = "at::Tensor: \nundefined tensor\n";
    std::string full_name = flie_name + ".txt";
    SaveStrToFile(val, full_name);
  }
}

template <>
void dump_arg<at::Tensor>(std::string path, const at::Tensor& var,
                          int64_t args_num) {
  std::string file_name = path + "_args" + std::to_string(args_num);
  dump_tensor(file_name, var);
}

template <>
void dump_arg<c10::optional<at::Tensor>>(
    std::string path, const c10::optional<at::Tensor>& opt_tensor,
    int64_t args_num) {
  if (opt_tensor.has_value()) {
    std::string file_name = path + "_args" + std::to_string(args_num);
    dump_tensor(file_name, *opt_tensor);
  } else {
    std::string file = path + "_args" + std::to_string(args_num) + ".txt";
    SaveStrToFile("c10::optional<at::Tensor>: c10::nullopt", file);
  }
}

template <>
void dump_arg<std::vector<at::Tensor>>(std::string path,
                                       const std::vector<at::Tensor>& list,
                                       int64_t args_num) {
  int64_t list_num = 0;
  for (const auto& var : list) {
    std::string file_name = path + "_args" + std::to_string(args_num) + "_" +
                            std::to_string(list_num);
    dump_tensor(file_name, var);
    list_num++;
  }
}

template <>
void dump_arg<at::TensorList>(std::string path, const at::TensorList& list,
                              int64_t args_num) {
  int64_t list_num = 0;
  auto tensor_vec = list.vec();
  for (const auto& var : tensor_vec) {
    std::string file_name = path + "_args" + std::to_string(args_num) + "_" +
                            std::to_string(list_num);
    dump_tensor(file_name, var);
    list_num++;
  }
}

template <>
void dump_arg<at::ITensorListRef>(std::string path,
                                  const at::ITensorListRef& list_ref,
                                  int64_t args_num) {
  auto tensor_vec = listref_to_vector(list_ref);
  dump_arg(path, tensor_vec, args_num);
}

template <>
void dump_arg<c10::List<at::Tensor>>(std::string path,
                                     const c10::List<at::Tensor>& list,
                                     int64_t args_num) {
  int64_t list_num = 0;
  for (const auto& var : list) {
    std::string file_name = path + "_args" + std::to_string(args_num) + "_" +
                            std::to_string(list_num);
    dump_tensor(file_name, var);
    list_num++;
  }
}

template <>
void dump_arg<c10::List<c10::optional<at::Tensor>>>(
    std::string path, const c10::List<c10::optional<at::Tensor>>& list,
    int64_t args_num) {
  int64_t list_num = 0;
  for (c10::optional<at::Tensor> var : list) {
    if (var.has_value()) {
      std::string file_name = path + "_args" + std::to_string(args_num) + "_" +
                              std::to_string(list_num);
      dump_tensor(file_name, *var);
    } else {
      std::string file = path + "_args" + std::to_string(args_num) + "_" +
                         std::to_string(list_num) + ".txt";
      SaveStrToFile("c10::optional<at::Tensor>: \nc10::nullopt", file);
    }
    list_num++;
  }
}

void dump_args(std::string /*path*/, int /*idx*/) {}

template <>
at::Tensor clone_arg<at::Tensor>(const at::Tensor& var) {
  if (var.defined()) {
    return var.cpu().clone();
  } else {
    at::Tensor undef;
    return undef;
  }
}

template <>
c10::optional<at::Tensor> clone_arg<c10::optional<at::Tensor>>(
    const c10::optional<at::Tensor>& opt_tensor) {
  if (opt_tensor.has_value()) {
    return clone_arg(opt_tensor.value());
  } else {
    return c10::nullopt;
  }
}

template <>
std::vector<at::Tensor> clone_arg<std::vector<at::Tensor>>(
    const std::vector<at::Tensor>& list) {
  std::vector<at::Tensor> tensor_vec(list.size());
  for (size_t idx = 0; idx < list.size(); ++idx) {
    tensor_vec[idx] = clone_arg(list[idx]);
  }
  return tensor_vec;
}

template <>
c10::List<at::Tensor> clone_arg<c10::List<at::Tensor>>(
    const c10::List<at::Tensor>& list) {
  std::vector<at::Tensor> tensor_vec(list.size());
  for (size_t idx = 0; idx < list.size(); ++idx) {
    tensor_vec[idx] = clone_arg(list[idx]);
  }
  return c10::List<at::Tensor>(tensor_vec);
}

template <>
c10::List<c10::optional<at::Tensor>>
clone_arg<c10::List<c10::optional<at::Tensor>>>(
    const c10::List<c10::optional<at::Tensor>>& list) {
  std::vector<c10::optional<at::Tensor>> opt_tensor_vec(list.size());
  for (size_t idx = 0; idx < list.size(); ++idx) {
    opt_tensor_vec[idx] = clone_arg(list[idx]);
  }
  return c10::List<c10::optional<at::Tensor>>(opt_tensor_vec);
}

std::vector<at::Tensor> listref_to_vector(const at::ITensorListRef& list_ref) {
  auto materialize = list_ref.materialize();
  std::vector<at::Tensor> vec;
  vec.reserve(list_ref.size());
  for (const at::Tensor& var : materialize) {
    vec.emplace_back(var);
  }
  return vec;
}

}  // namespace torch_gcu
