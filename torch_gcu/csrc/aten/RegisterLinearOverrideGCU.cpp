/*
 * Copyright 2024 Enflame. All Rights Reserved.
 */

#include <ATen/Operators.h>
#include <ATen/Tensor.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/core/op_registration/adaption.h>
#include <torch/library.h>

#include "aten/aot_ops/gcu_op_check.h"
#include "aten/aot_ops/topsaten_bridge_define.h"
#include "aten/op_debug_config.h"
#include "aten/op_statistics.h"
#include "aten/shape_inference/shape_infer_func.h"
#include "gcu/sys_util.h"

namespace torch_gcu {

DEFINE_BRIDGE_TOPSATENOP(topsatenLinear);
DEFINE_BRIDGE_TOPSATENOP(topsatenMatmul);

namespace aotops {

namespace {

// Parse environment variable "TORCH_LINEAR_FLATTEN_3D"
static inline bool parseLinearFlatten3d() {
  // Uninitialized value
  static int value = -1;
  if (value == -1) {
    const char* env_str = std::getenv("TORCH_LINEAR_FLATTEN_3D");
    if (env_str != nullptr && strcmp(env_str, "1") == 0) {
      value = 1;
    } else {
      value = 0;
    }
  }
  return bool(value);
}

bool should_fold(const at::Tensor& tensor1, const at::Tensor& tensor2) {
  // We check that we can fold the larger tensor into a matrix and dispatch to
  // mm or mv rather than to bmm. We want to make sure we can do so without
  // incurring in any extra copy
  const auto tensor1_larger = tensor1.dim() >= tensor2.dim();

  // We order the tensors. t1 will be the larger tensor
  // We can always transpose tensor2 as the dimensions are always >= 1
  // (precondition from matmul) and tensor1_larger iff tensor2.dim() >
  // tensor1.dim(9
  const auto t1 = tensor1_larger
                      ? c10::MaybeOwned<at::Tensor>::borrowed(tensor1)
                      : c10::MaybeOwned<at::Tensor>::owned(tensor2.mT());
  const int64_t dim_t1 = t1->dim();
  const auto dim_t2 = tensor1_larger ? tensor2.dim() : tensor1.dim();

  // Just fold for dim_t1 >= 3 and (dim_t2 == 1 || dim_t2 == 2)
  if (!(dim_t1 >= 3 && dim_t2 <= 2)) {
    return false;
  }

  // In this case we *do* incur in an extra copy to avoid creating an
  // unnecessary large tensor in the backward Suppose we don't fold here. Let
  // t1.shape = [b, m, n] t2.shape = [n, k] like in a transformer t2 will be
  // expanded to a tensor of shape [b, n, k] and then we do t1.bmm(t2_expanded)
  // The issue appears in the backward.
  // The output gradient g of this operation would have shape [b, m, k]
  // The backward wrt. t2 of bmm would be given by t1.mH @ g, which has shape
  // [b, n, k] Then, the backward of expand is simply `sum(0)`. As such, we are
  // instantiating a tensor of shape [b, n, k] unnacessarily, which may cause a
  // large memory footprint, and in the worst case, an OOM
  bool t2_requires_grad =
      tensor1_larger ? tensor2.requires_grad() : tensor1.requires_grad();
  if (t2_requires_grad) {
    return true;
  }

  // Don't fold in this case, as we would have to call mm on the transposed
  // tensor, the result would be contiguous, and then we would need to transpose
  // it and call contiguous on it, thus having to copy the tensor
  if (tensor1.dim() == 2) {
    return false;
  }

  // Can always fold if the tensor is empty
  // This serves as a precondition for the code below
  if (t1->numel() == 0) {
    return true;
  }

  // t1->view(-1, t1->size(-1)) does not copy only when the first n-1 dimensions
  // are contiguous in the sense that t1_stride[i] =
  // t1_stride[i+1]*t1_shape[i+1]
  const auto t1_shape = t1->sizes();
  const auto t1_strides = t1->strides();
  for (auto i = int64_t{0}; i < dim_t1 - int64_t{2}; ++i) {
    if (t1_strides[i] != t1_strides[i + 1] * t1_shape[i + 1]) {
      return false;
    }
  }
  return true;
}

at::Tensor& mv_out_shape_infer(const at::Tensor& self, const at::Tensor& vec,
                               at::Tensor& result) {
  if (result.dim() > 1 ||
      (result.numel() != self.size(0) || result.numel() != 1)) {
    at::Tensor self_addmv = empty({self.size(0)}, vec.options());
    return addmv_out_shape_infer(self_addmv, self, vec, 0, 1, result);
  }
  return addmv_out_shape_infer(result, self, vec, 0, 1, result);
}

at::Tensor mv_shape_infer(const at::Tensor& self, const at::Tensor& vec) {
  at::Tensor result = empty({self.size(0)}, vec.options());
  // inplace version is more efficient if we can use it
  return addmv__shape_infer(result, self, vec, 0, 1);
}

at::Tensor& dot_out_shape_infer(const at::Tensor& self, const at::Tensor& other,
                                at::Tensor& result) {
  auto output_device = result.device();
  auto input1_device = self.device();
  auto input2_device = other.device();
  // check if the input & output tensors are on the same device.
  TORCH_CHECK(
      (output_device == input1_device) && (input1_device == input2_device),
      "dot: Expected the output and input tensors to be on the "
      "same device, but got the output tensor on ",
      output_device, ", the 'input' tensor on ", input1_device,
      ", and the 'other' tensor on ", input2_device);
  aotops::resize_output(result, {});
  TORCH_CHECK(result.scalar_type() == self.scalar_type(), "result dtype ",
              result.scalar_type(), " does not match input dtype ",
              self.scalar_type());
  return result;
}

at::Tensor dot_shape_infer(const at::Tensor& self, const at::Tensor& tensor) {
  c10::optional<at::Device> common_device = c10::nullopt;
  (void)common_device;  // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self,
                                            "dot_shape_infer", "self");
  c10::impl::check_and_update_common_device(common_device, tensor,
                                            "dot_shape_infer", "tensor");
  at::Tensor result = at::empty({}, self.options());
  return result;
}

at::Tensor _matmul_impl_shape_infer(at::Tensor& out, const at::Tensor& tensor1,
                                    const at::Tensor& tensor2) {
  at::NoNamesGuard guard;
  const auto dim_tensor1 = tensor1.dim();
  const auto dim_tensor2 = tensor2.dim();

  // This is checked up here to simplify the logic below
  // Note that the strings are just evaluated on failure, so almost always we
  // just evaluate the condition and move on
  TORCH_CHECK(dim_tensor1 != 0 && dim_tensor2 != 0,
              "both arguments to matmul need to be at least 1D, but they are ",
              dim_tensor1, "D and ", dim_tensor2, "D");

  const bool has_out = out.defined();

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    return has_out ? dot_out_shape_infer(tensor1, tensor2, out)
                   : dot_shape_infer(tensor1, tensor2);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    return has_out ? mv_out_shape_infer(tensor1, tensor2, out)
                   : mv_shape_infer(tensor1, tensor2);
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return has_out ? mm_out_shape_infer(tensor1.unsqueeze(0), tensor2, out)
                         .squeeze_(0)
                   : mm_shape_infer(tensor1.unsqueeze(0), tensor2).squeeze_(0);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return has_out ? mm_out_shape_infer(tensor1, tensor2, out)
                   : mm_shape_infer(tensor1, tensor2);
  } else if (should_fold(tensor1, tensor2)) {
    // dim_tensor1 >=3 && (dim_tensor2 == 1 || dim_tensor2 == 2) ||
    // dim_tensor2 >=3 && (dim_tensor1 == 1 || dim_tensor1 == 2)
    // and at least one of the following two conditions hold
    // - the small tensor requires grad (see should_fold for the why)
    // - we can fold the larger tensor t1 into a matrix as t1.view(-1,
    // t1.size(-1)) without copying

    // optimization: use mm instead of bmm by folding the batch of the larger
    // tensor into its leading matrix dimension
    const auto transpose = dim_tensor2 > dim_tensor1;
    const auto t1 = transpose ? c10::MaybeOwned<at::Tensor>::owned(tensor2.mT())
                              : c10::MaybeOwned<at::Tensor>::borrowed(tensor1);
    const auto t2 = !transpose
                        ? c10::MaybeOwned<at::Tensor>::borrowed(tensor2)
                        : dim_tensor1 == 2
                              ? c10::MaybeOwned<at::Tensor>::owned(tensor1.t())
                              : c10::MaybeOwned<at::Tensor>::borrowed(tensor1);
    // Invariant: t1->dim() >= 3 && (t2->dim() == 1 || t2->dim() == 2)
    //            and *t1 and *t2 are matmul-compatible

    // Why not t1->view(-1, sizes_1.back())?
    // If the last dim is 0, then view(-1, 0) won't work because the -1 becomes
    // ambiguous. This can happen in e.g. [3, 5, 0] @ [0, 0].
    const auto sizes_1 = t1->sizes();
    auto output_shape = at::DimVector(sizes_1.begin(), sizes_1.end() - 1);
    const auto folded_dim1 = c10::multiply_integers(output_shape);

    // Readjust output_shape if we are multiplying by a matrix
    const auto t2_is_matrix = t2->dim() == 2;
    if (t2_is_matrix) {
      output_shape.push_back(t2->sizes()[1]);
    }
    // This will almost always be a view.
    // It may not be a view if t2->requires_grad(). See should_fold for an
    // explanation
    const auto t1_folded = t1->reshape({folded_dim1, sizes_1.back()});
    if (!has_out) {
      if (t2_is_matrix) {
        const auto output =
            at::_unsafe_view(mm_shape_infer(t1_folded, *t2), output_shape);
        // This copies if we perform a 2D @ 3D and the first tensor
        // requires_grad See should_fold for why. If mm_out were differentiable,
        // we could use it here, and pass a result with the correct strides to
        // avoid this unnecessary copy.
        return transpose ? contiguous_shape_infer(output.mT()) : output;
      } else {
        return at::_unsafe_view(mv_shape_infer(t1_folded, *t2), output_shape);
      }
    } else {
      // See the !has_out branch for an explanation
      TORCH_INTERNAL_ASSERT(!(transpose && t2_is_matrix));

      // Resize output into the correct shape
      aotops::resize_output(out, output_shape);

      // We then reshape the output to the expected shape and call mm/mv
      // and transpose back if necessary
      auto reshaped_out = t2_is_matrix
                              ? out.reshape({folded_dim1, t2->sizes().back()})
                              : out.reshape({folded_dim1});
      if (t2_is_matrix) {
        mm_out_shape_infer(t1_folded, *t2, reshaped_out);
      } else {
        mv_out_shape_infer(t1_folded, *t2, reshaped_out);
      }
      if (!reshaped_out.is_alias_of(out)) {
        out.reshape(reshaped_out.sizes());
      }
      return out;
    }
  } else {
    // dim_tensor1 >= 3 || dim_tensor2 >= 3
    // We track m1 vs m2 separately even though they must match for nicer error
    // messages
    const int64_t n = dim_tensor1 > 1 ? tensor1.sizes().cend()[-2] : 1LL;
    const int64_t m1 = tensor1.sizes().back();
    auto batch_tensor1 =
        tensor1.sizes().slice(0, std::max<int64_t>(dim_tensor1 - 2, 0LL));
    const int64_t m2 =
        dim_tensor2 > 1 ? tensor2.sizes().cend()[-2] : tensor2.sizes().front();
    const int64_t p = dim_tensor2 > 1 ? tensor2.sizes().back() : 1LL;
    const at::IntArrayRef batch_tensor2(
        tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0LL));

    // Same optimization for the gradients as that in should_fold
    // If we're going to broadcast we force it to go through the should_fold
    // branch
    if (dim_tensor1 == 3 && dim_tensor2 == 3 &&
        batch_tensor1[0] != batch_tensor2[0]) {
      if (batch_tensor1[0] == 1 &&
          (tensor1.requires_grad() || at::isTensorSubclassLike(tensor1))) {
        return _matmul_impl_shape_infer(out, tensor1.squeeze(0), tensor2);
      }
      if (batch_tensor2[0] == 1 &&
          (tensor2.requires_grad() || at::isTensorSubclassLike(tensor2))) {
        return _matmul_impl_shape_infer(out, tensor1, tensor2.squeeze(0));
      }
    }

    auto output_shape = at::infer_size_dimvector(batch_tensor1, batch_tensor2);
    const int64_t expand_batch_product = c10::multiply_integers(output_shape);

    // flatten expanded batches
    const auto tensor1_expand_size = [&output_shape, n, m1] {
      at::DimVector ret(output_shape);
      ret.append({n, m1});
      return ret;
    }();
    const auto tensor1_expanded = tensor1.expand(tensor1_expand_size)
                                      .reshape({expand_batch_product, n, m1});
    // We need to treat the dim_tensor2 == 1 case separately as broadcasting
    // would not convert a vector of shape (n,) into a batch of matrices of
    // shape (*, n, 1)
    auto vector_rhs = dim_tensor2 == 1;
    const auto tensor2_expand_size = [&output_shape, m2, p, vector_rhs] {
      at::DimVector ret(output_shape);
      if (vector_rhs) {
        ret.push_back(m2);
      } else {
        ret.append({m2, p});
      }
      return ret;
    }();
    auto tensor2_expanded = tensor2.expand(tensor2_expand_size);
    if (vector_rhs) {
      tensor2_expanded =
          tensor2_expanded.reshape({expand_batch_product, m2}).unsqueeze(2);
    } else {
      tensor2_expanded =
          tensor2_expanded.reshape({expand_batch_product, m2, p});
    }

    if (dim_tensor1 > 1) {
      output_shape.push_back(n);
    }
    if (dim_tensor2 > 1) {
      output_shape.push_back(p);
    }

    if (!has_out) {
      if (vector_rhs) {
        return at::_unsafe_view(
            bmm_shape_infer(tensor1_expanded, tensor2_expanded).squeeze(-1),
            output_shape);
      } else {
        return at::_unsafe_view(
            bmm_shape_infer(tensor1_expanded, tensor2_expanded), output_shape);
      }
    } else {
      aotops::resize_output(out, output_shape);
      auto reshaped_out = out.reshape({expand_batch_product, n, p});
      bmm_out_shape_infer(tensor1_expanded, tensor2_expanded, reshaped_out);
      if (vector_rhs) {
        reshaped_out = reshaped_out.squeeze(-1);
      }
      return out;
    }
  }
}

at::Tensor matmul_shape_infer(const at::Tensor& tensor1,
                              const at::Tensor& tensor2) {
  auto maybe_outnames =
      at::namedinference::compute_matmul_outnames(tensor1, tensor2);
  at::Tensor result, unused;
  result = _matmul_impl_shape_infer(unused, tensor1, tensor2);
  at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

at::Tensor _flatten_nd_linear_shape_infer(const at::Tensor& input,
                                          const at::Tensor& weight,
                                          const at::Tensor& bias) {
  const auto input_sizes = input.sym_sizes();
  // can't use -1 in reshape because it errors when a dimension is 0
  c10::SymInt flattened_dim = 1;
  for (int64_t i = 0, ndim = input_sizes.size(); i < ndim - 1; ++i) {
    flattened_dim = flattened_dim * input_sizes[i];
  }
  auto inp_reshape = input.reshape_symint(
      {flattened_dim, input_sizes.at(input_sizes.size() - 1)});
  const auto result = addmm_shape_infer(bias, inp_reshape, weight.t());

  auto new_size = input_sizes.slice(0, input_sizes.size() - 1);
  c10::SymDimVector sizes_vec(new_size.begin(), new_size.end());
  sizes_vec.push_back(result.sym_size(1));
  return result.view_symint(sizes_vec);
}

at::Tensor linear_shape_infer(const at::Tensor& input, const at::Tensor& weight,
                              const c10::optional<at::Tensor>& bias_opt) {
  auto bias = bias_opt.has_value()
                  ? c10::MaybeOwned<at::Tensor>::borrowed(*bias_opt)
                  : c10::MaybeOwned<at::Tensor>::owned(std::in_place);

  const auto input_dim = input.dim();
  if (input_dim == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return addmm_shape_infer(*bias, input, weight.t());
  }
  if (bias->defined()) {
    // Also hit the fused path for contiguous 3D input, if not using xla
    // backend. Reshaping/flattening has some performance implications on xla.
    if (input.is_contiguous() && input_dim == 3) {
      return _flatten_nd_linear_shape_infer(input, weight, *bias);
    } else if (input.is_contiguous() && input.layout() == c10::kStrided &&
               weight.layout() == c10::kStrided && bias->dim() == 1) {
      return _flatten_nd_linear_shape_infer(input, weight, *bias);
    } else if (parseLinearFlatten3d() && input_dim == 3) {
      // If user forces flattening via env var
      const at::Tensor input_cont = contiguous_shape_infer(input);
      return _flatten_nd_linear_shape_infer(input_cont, weight, *bias);
    }
  }
  auto output = matmul_shape_infer(input, weight.t());
  if (bias->defined()) {
    // for composite compliance use out-of-place version of `add`
    if (at::isTensorSubclassLike(*bias) ||
        bias->_fw_grad(/*level*/ 0).defined()) {
      output = add_shape_infer(output, *bias);
    } else {
      add__shape_infer(output, *bias);
    }
  }
  return output;
}

}  // namespace

at::Tensor linear(const at::Tensor& input, const at::Tensor& weight,
                  const c10::optional<at::Tensor>& bias_opt) {
  auto out = linear_shape_infer(input, weight, bias_opt);
  bridge_topsatenLinear_out1(out, input, weight, bias_opt);
  return out;
}

at::Tensor matmul(const at::Tensor& self, const at::Tensor& other) {
  auto out = matmul_shape_infer(self, other);
  bridge_topsatenMatmul_out1(out, self, other);
  return out;
}

}  // namespace aotops

at::Tensor linear(const at::Tensor& input, const at::Tensor& weight,
                  const c10::optional<at::Tensor>& bias) {
  auto& cfg = torch_gcu::OpDebugConfig::GetInstance();
  OP_CALLTRACE(cfg, linear)
  PRINT_OP_NAME_ALL(cfg)
  GCU_CPU_FALLBACK(cfg, ATEN, linear, input, weight, bias)
  OP_COMMON_MACRO(input, weight, bias);
  static bool enable_op_check = cfg.enableOpCheck(__func__);
  static bool disable_op_check = cfg.disableOpCheck(__func__);
  bool op_check_scope = cfg.inOpCheckScope();
  bool op_check = (enable_op_check || op_check_scope) && (!disable_op_check);
  if (op_check) {
    OP_CHECK_INPUT_INFO_RECOED(input, weight, bias)
    auto clone_input = clone_args(input, weight, bias);
    auto clone_op_check_input = clone_args(input, weight, bias);
    auto&& xdevice_out = at::native::
        call_fallback_fn<&torch_gcu::gcu_opcheck_run, ATEN_OP(linear)>::call(
            std::get<0>(clone_op_check_input),
            std::get<1>(clone_op_check_input),
            std::get<2>(clone_op_check_input));
    auto gcu_out = aotops::linear(input, weight, bias);
    auto result = gcu_out_check(gcu_out, xdevice_out, std::string(__func__));
    if (result.acc_pass && !cfg.enableTestMode()) {
      PTDLOG(OP) << result.check_info;
    } else {
      OP_CHECK_DEBUG_INFO(cfg, ss, gcu_out, xdevice_out, clone_input)
    }
    return gcu_out;
  } else {
    return aotops::linear(input, weight, bias);
  }
}

at::Tensor matmul(const at::Tensor& self, const at::Tensor& other) {
  auto& cfg = torch_gcu::OpDebugConfig::GetInstance();
  OP_CALLTRACE(cfg, matmul)
  PRINT_OP_NAME_ALL(cfg)
  GCU_CPU_FALLBACK(cfg, ATEN, matmul, self, other)
  OP_COMMON_MACRO(self, other);
  static bool enable_op_check = cfg.enableOpCheck(__func__);
  static bool disable_op_check = cfg.disableOpCheck(__func__);
  bool op_check_scope = cfg.inOpCheckScope();
  bool op_check = (enable_op_check || op_check_scope) && (!disable_op_check);
  if (op_check) {
    OP_CHECK_INPUT_INFO_RECOED(self, other)
    auto clone_input = clone_args(self, other);
    auto clone_op_check_input = clone_args(self, other);
    auto&& xdevice_out = at::native::
        call_fallback_fn<&torch_gcu::gcu_opcheck_run, ATEN_OP(matmul)>::call(
            std::get<0>(clone_op_check_input),
            std::get<1>(clone_op_check_input));
    auto gcu_out = aotops::matmul(self, other);
    auto result = gcu_out_check(gcu_out, xdevice_out, std::string(__func__));
    if (result.acc_pass && !cfg.enableTestMode()) {
      PTDLOG(OP) << result.check_info;
    } else {
      OP_CHECK_DEBUG_INFO(cfg, ss, gcu_out, xdevice_out, clone_input)
    }
    return gcu_out;
  } else {
    return aotops::matmul(self, other);
  }
}

}  // namespace torch_gcu

// See template file RegisterDispatchDefinitions.ini
namespace at {
// NB: TORCH_LIBRARY_IMPL must be in an anonymous namespace to avoid
// ambiguity with conflicting identifiers that may have been defined in
// at namespace already.

namespace {

namespace {

at::Tensor wrapper_GCU__linear(const at::Tensor& input,
                               const at::Tensor& weight,
                               const c10::optional<at::Tensor>& bias) {
  c10::optional<at::Device> common_device = c10::nullopt;
  (void)common_device;  // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input,
                                            "wrapper_GCU__linear", "input");
  c10::impl::check_and_update_common_device(common_device, weight,
                                            "wrapper_GCU__linear", "weight");
  c10::impl::check_and_update_common_device(common_device, bias,
                                            "wrapper_GCU__linear", "bias");
  const OptionalDeviceGuard device_guard(device_of(input));
  auto& cfg = torch_gcu::OpDebugConfig::GetInstance();
  auto gcu_out = torch_gcu::linear(input, weight, bias);
  if (cfg.enableOpStatistics()) {
    auto op_schema = c10::Dispatcher::singleton()
                         .findSchemaOrThrow(ATEN_OP(linear)::name,
                                            ATEN_OP(linear)::overload_name)
                         .schema();
    std::vector<at::IValue> input_arguments = {
        at::IValue(input), at::IValue(weight), at::IValue(bias)};
    std::vector<at::IValue> return_arguments = {at::IValue(gcu_out)};
    torch_gcu::op_record(op_schema, input_arguments, return_arguments);
  }
  return gcu_out;
}

}  // namespace

at::Tensor& wrapper_GCU_out_linear_out(const at::Tensor& input,
                                       const at::Tensor& weight,
                                       const c10::optional<at::Tensor>& bias,
                                       at::Tensor& out) {
  auto wrapper_GCU_out_linear_out_tmp =
      wrapper_GCU__linear(input, weight, bias);
  at::_copy_from_and_resize(wrapper_GCU_out_linear_out_tmp, out);
  return out;
}

namespace {

at::Tensor wrapper_GCU__matmul(const at::Tensor& self,
                               const at::Tensor& other) {
  c10::optional<at::Device> common_device = c10::nullopt;
  (void)common_device;  // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self,
                                            "wrapper_GCU__matmul", "self");
  c10::impl::check_and_update_common_device(common_device, other,
                                            "wrapper_GCU__matmul", "other");
  const OptionalDeviceGuard device_guard(device_of(self));
  auto& cfg = torch_gcu::OpDebugConfig::GetInstance();
  auto gcu_out = torch_gcu::matmul(self, other);
  if (cfg.enableOpStatistics()) {
    auto op_schema = c10::Dispatcher::singleton()
                         .findSchemaOrThrow(ATEN_OP(matmul)::name,
                                            ATEN_OP(matmul)::overload_name)
                         .schema();
    std::vector<at::IValue> input_arguments = {at::IValue(self),
                                               at::IValue(other)};
    std::vector<at::IValue> return_arguments = {at::IValue(gcu_out)};
    torch_gcu::op_record(op_schema, input_arguments, return_arguments);
  }
  return gcu_out;
}

}  // namespace

at::Tensor& wrapper_GCU_out_matmul_out(const at::Tensor& self,
                                       const at::Tensor& other,
                                       at::Tensor& out) {
  auto wrapper_GCU_out_matmul_out_tmp = wrapper_GCU__matmul(self, other);
  at::_copy_from_and_resize(wrapper_GCU_out_matmul_out_tmp, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  bool enable_linear_override = false;
  std::string env_str = torch_gcu::util::GetEnvString(
      "ENFLAME_PT_ENABLE_LINEAR_OVERRIDE", "false");
  env_str.erase(0, env_str.find_first_not_of(" "));
  env_str.erase(env_str.find_last_not_of(" ") + 1);
  std::transform(env_str.begin(), env_str.end(), env_str.begin(), ::tolower);
  if (env_str == "true") enable_linear_override = true;

  if (enable_linear_override) {
    m.impl("linear", TORCH_FN(wrapper_GCU__linear));
    m.impl("linear.out", TORCH_FN(wrapper_GCU_out_linear_out));
    m.impl("matmul", TORCH_FN(wrapper_GCU__matmul));
    m.impl("matmul.out", TORCH_FN(wrapper_GCU_out_matmul_out));
  }
};

}  // namespace

}  // namespace at
