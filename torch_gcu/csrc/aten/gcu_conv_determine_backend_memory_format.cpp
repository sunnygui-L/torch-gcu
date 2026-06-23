/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */
#include "aten/gcu_conv_determine_backend_memory_format.h"

namespace torch_gcu {

at::MemoryFormat _determine_backend_memory_format(
    const at::Tensor &input, const at::Tensor &weight,
    const at::native::ConvBackend backend) {
  if (input.device().is_privateuseone()) {
    auto channel_last = at::MemoryFormat::ChannelsLast;
    auto channel_last3d = at::MemoryFormat::ChannelsLast3d;
    auto input_memory_format = input.suggest_memory_format();
    auto weight_memory_format = weight.suggest_memory_format();
    auto weight_ndim = weight.ndimension();

    bool can_use_channels_last_2d =
        (weight_ndim == 4) && ((input_memory_format == channel_last) ||
                               (weight_memory_format == channel_last));
    if (can_use_channels_last_2d) {
      return channel_last;
    }

    bool can_use_channels_last_3d =
        (weight_ndim == 5) && ((input_memory_format == channel_last3d) ||
                               (weight_memory_format == channel_last3d));
    if (can_use_channels_last_3d) {
      return channel_last3d;
    }

    return at::MemoryFormat::Contiguous;
  }
  return at::native::_determine_backend_memory_format(input, weight, backend);
}

}  // namespace torch_gcu
