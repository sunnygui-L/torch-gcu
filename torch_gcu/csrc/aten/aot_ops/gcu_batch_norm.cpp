/*
 * Copyright 2022-2025 Enflame. All Rights Reserved.
 */

#include <ATen/AccumulateType.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/native/layer_norm.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/mean.h>
#include <topsaten/topsaten_ops.h>

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/aot_ops/gcu_ops.h"
#include "aten/aot_ops/topsaten_bridge.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_utils.h"

namespace torch_gcu {

namespace aotops {

static inline at::MemoryFormat suggest_memory_format_contig(
    const at::Tensor &t) {
  return t.is_contiguous() ? at::MemoryFormat::Contiguous
                           : (t.is_contiguous(at::MemoryFormat::ChannelsLast3d)
                                  ? at::MemoryFormat::ChannelsLast3d
                                  : at::MemoryFormat::ChannelsLast);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm(
    const at::Tensor &input, const c10::optional<at::Tensor> &weight_opt,
    const c10::optional<at::Tensor> &bias_opt,
    const c10::optional<at::Tensor> &running_mean_opt,
    const c10::optional<at::Tensor> &running_var_opt, bool training,
    double momentum, double eps) {
  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor &weight = *weight_maybe_owned;

  auto is_self_contiguous =
      input.is_contiguous() ||
      input.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      input.is_contiguous(at::MemoryFormat::ChannelsLast3d);

  at::Tensor output = aotops::empty(input.sizes().vec(), input.options());

  topsatenTensor xweight, xbias;
  bool has_weight = weight.defined();
  bool has_bias = bias_opt.has_value() && bias_opt->defined();
  if (has_weight) {
    xweight = topsaten_variable(weight).value;
  }
  if (has_bias) {
    xbias = topsaten_variable(bias_opt).value;
  }

  bool has_running_mean =
      running_mean_opt.has_value() && running_mean_opt->defined();
  bool has_running_var =
      running_var_opt.has_value() && running_var_opt->defined();

  at::Tensor save_var;
  at::Tensor save_mean;

  const bool mixed_type = at::native::is_mixed_type(
      input, weight, has_bias ? *bias_opt : at::Tensor(),
      has_running_mean ? *running_mean_opt : at::Tensor(),
      has_running_var ? *running_var_opt : at::Tensor());

  const int64_t ndim = input.dim();
  c10::DimVector reduce_dims(ndim - 1);
  reduce_dims[0] = 0;
  for (const auto i : c10::irange(2, ndim)) {
    reduce_dims[i - 1] = i;
  }
  if (mixed_type) {
    if (!training) {
      save_mean = aotops::empty({0}, input.options().dtype(at::kFloat));
      save_var = aotops::empty({0}, input.options().dtype(at::kFloat));
    } else {
      save_mean = is_self_contiguous
                      ? aotops::empty({input.size(1)},
                                      input.options().dtype(at::kFloat))
                      : at::mean(input, /*dim=*/reduce_dims, /*keepdim=*/false,
                                 at::kFloat);
      save_var =
          aotops::empty({input.size(1)}, input.options().dtype(at::kFloat));
    }
  } else {
    if (!training) {
      save_mean = aotops::empty({0}, input.options());
      save_var = aotops::empty({0}, input.options());
    } else {
      save_mean = is_self_contiguous
                      ? aotops::empty({input.size(1)}, input.options())
                      : at::mean(input, /*dim=*/reduce_dims, /*keepdim=*/false);
      save_var = aotops::empty({input.size(1)}, input.options());
    }
  }

  auto xsave_mean = topsaten_variable(save_mean).value;
  auto xsave_var = topsaten_variable(save_var).value;
  auto xrunning_mean = topsaten_variable(running_mean_opt).value;
  auto xrunning_var = topsaten_variable(running_var_opt).value;
  auto xinput = topsaten_variable(input).value;
  auto xoutput = topsaten_variable(output).value;

  auto stream = getCurrentGCUStream();
  auto op_info = [&]() -> std::string {
    std::stringstream ss;
    // clang-format off
    ss << "topsatenNativeBatchNorm"
       << " :\n"
       << tensorArgsToString({input}, {output, save_mean, save_var})
       << "weight: " << (has_weight ? tensorToString(weight) : "none") << "\n"
       << "bias: " << (has_bias ? tensorToString(bias_opt) : "none") << "\n"
       << "running_mean: "
       << (has_running_mean ? tensorToString(running_mean_opt) : "none") << "\n"
       << "running_var: "
       << (has_running_var ? tensorToString(running_var_opt) : "none") << "\n"
       << "training: " << training << "\n"
       << "momentum: " << momentum << "\n"
       << "eps: " << eps << "\n"
       << "stream: " << (topsStream_t)stream << "\n";
    // clang-format on
    return ss.str();
  };
  PTDLOG(OP) << op_info();
  CHECK_TOPSATEN_CALL(
      topsaten::topsatenNativeBatchNorm(
          xoutput, xsave_mean, xsave_var, xinput, xweight, xbias, xrunning_mean,
          xrunning_var, training, momentum, eps, stream),
      op_info);

  maybeGCUStreamSynchronize(stream);
  return std::make_tuple(output, save_mean, save_var);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _native_batch_norm_legit(
    const at::Tensor &input, const c10::optional<at::Tensor> &weight,
    const c10::optional<at::Tensor> &bias, at::Tensor &running_mean,
    at::Tensor &running_var, bool training, double momentum, double eps) {
  return aotops::native_batch_norm(input, weight, bias, running_mean,
                                   running_var, training, momentum, eps);
}

}  // namespace aotops

}  // namespace torch_gcu
