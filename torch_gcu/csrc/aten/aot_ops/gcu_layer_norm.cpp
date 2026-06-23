/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */

#include <ATen/AccumulateType.h>
#include <ATen/native/layer_norm.h>
#include <topsaten/topsaten_ops.h>

#include "aten/aot_ops/gcu_empty_tensor.h"
#include "aten/aot_ops/gcu_ops.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_utils.h"
namespace torch_gcu {

namespace aotops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm(
    const at::Tensor &input, at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor> &weight_opt,
    const c10::optional<at::Tensor> &bias_opt, double eps) {
  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor &weight = *weight_maybe_owned;
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor &bias = *bias_maybe_owned;
  topsatenTensor xweight, xbias;
  bool has_weight = weight.defined();
  bool has_bias = bias.defined();
  if (has_weight) {
    xweight = createTopsatenTensor(weight);
  }
  if (has_bias) {
    xbias = createTopsatenTensor(bias);
  }

  auto input_shape = input.sizes();
  std::vector<int64_t> stat_shape;
  const size_t axis = input.dim() - normalized_shape.size();
  for (const auto idx : c10::irange(axis)) {
    stat_shape.push_back(input_shape[idx]);
  }
  // NOTE:if is_cuda=false the acc_type for float will be double which is
  // different with cpu and gpu
  auto acc_type = at::toAccumulateType(input.scalar_type(), /*is_cuda=*/true);
  at::Tensor mean = aotops::empty(stat_shape, input.options().dtype(acc_type));
  at::Tensor rstd = aotops::empty(stat_shape, input.options().dtype(acc_type));
  auto output = aotops::empty(input.sizes(), input.options());

  auto xmean = createTopsatenTensor(mean);
  auto xrstd = createTopsatenTensor(rstd);
  auto xinput = createTopsatenTensor(input);
  auto xoutput = createTopsatenTensor(output);
  topsatenSize_t xshape = {normalized_shape.data(),
                           static_cast<int64_t>(normalized_shape.size())};
  auto stream = getCurrentGCUStream(input.device().index());
  topsatenScalar_t xeps = {TOPSATEN_DATA_F64, {.fval = eps}};
  auto op_info = [&]() -> std::string {
    std::stringstream ss;
    // clang-format off
    ss << "topsatenNativeLayerNorm" << " :\n"
       << tensorArgsToString({input}, {output, mean, rstd})
       << "normalized_shape: " << normalized_shape << "\n"
       << "weight: " << (has_weight ? tensorToString(weight) : "none") << "\n"
       << "bias: " << (has_bias ? tensorToString(bias) : "none") << "\n"
       << "eps: " << eps << "\n"
       << "stream: " << (topsStream_t)stream << "\n";
    // clang-format on
    return ss.str();
  };
  PTDLOG(OP) << op_info();
  CHECK_TOPSATEN_CALL(
      topsaten::topsatenNativeLayerNorm(xoutput, xmean, xrstd, xinput, xshape,
                                        xweight, xbias, xeps, stream),
      op_info);

  maybeGCUStreamSynchronize(stream);

  for (const auto idx : c10::irange(axis, input.dim())) {
    (void)idx;  // Suppress unused variable
    stat_shape.push_back(1);
  }
  mean = aotops::view(mean, stat_shape);
  rstd = aotops::view(rstd, stat_shape);
  return std::make_tuple(output, mean, rstd);
}

}  // namespace aotops

}  // namespace torch_gcu
