/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include "aten/shape_inference/batch_norm.h"

#include <ATen/AccumulateType.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/native/layer_norm.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/mean.h>

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/aot_ops/gcu_ops.h"
#include "aten/shape_inference/shape_infer_func.h"
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_shape_infer(
    const at::Tensor &input, const c10::optional<at::Tensor> &weight_opt,
    const c10::optional<at::Tensor> &bias_opt,
    const c10::optional<at::Tensor> &running_mean_opt,
    const c10::optional<at::Tensor> &running_var_opt, bool training,
    double momentum, double eps) {
  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor &weight = *weight_maybe_owned;
  const at::Tensor &bias =
      c10::value_or_else(bias_opt, [] { return at::Tensor(); });
  const at::Tensor &running_mean =
      c10::value_or_else(running_mean_opt, [] { return at::Tensor(); });
  const at::Tensor &running_var =
      c10::value_or_else(running_var_opt, [] { return at::Tensor(); });

  auto is_self_contiguous =
      input.is_contiguous() ||
      input.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      input.is_contiguous(at::MemoryFormat::ChannelsLast3d);

  const bool all_contiguous =
      is_self_contiguous && (!weight.defined() || weight.is_contiguous()) &&
      (!bias.defined() || bias.is_contiguous()) &&
      running_mean.is_contiguous() && running_var.is_contiguous();

  at::Tensor output =
      at::empty_like(input, all_contiguous ? suggest_memory_format_contig(input)
                                           : input.suggest_memory_format());

  at::Tensor save_var;
  at::Tensor save_mean;
  const bool mixed_type =
      at::native::is_mixed_type(input, weight, bias, running_mean, running_var);
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

  return std::make_tuple(output, save_mean, save_var);
}

}  // namespace aotops

}  // namespace torch_gcu