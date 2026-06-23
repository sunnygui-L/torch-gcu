#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <tops/tops_runtime_api.h>

#include <cstddef>
#include <vector>

#include "gcu/gcu_macros.h"
#include "gcu/gcu_stream.h"

namespace torch_gcu {

using tensor_list2d = std::vector<std::vector<at::Tensor>>;

TORCH_GCU_API void ReleaseEcclCommList();

TORCH_GCU_API std::vector<at::Tensor>& broadcast_out(
    const at::Tensor& tensor, std::vector<at::Tensor>& out_tensors);
TORCH_GCU_API std::vector<at::Tensor> broadcast(const at::Tensor& tensor,
                                                at::IntArrayRef devices);
TORCH_GCU_API tensor_list2d broadcast_coalesced(at::TensorList tensors,
                                                at::IntArrayRef devices,
                                                size_t buffer_size);

TORCH_GCU_API std::vector<at::Tensor>& scatter_out(
    const at::Tensor& tensor, std::vector<at::Tensor>& out_tensors,
    int64_t dim = 0,
    const std::optional<std::vector<std::optional<GCUStream>>>& streams =
        c10::nullopt);

TORCH_GCU_API std::vector<at::Tensor> scatter(
    const at::Tensor& tensor, at::IntArrayRef devices,
    const std::optional<std::vector<int64_t>>& chunk_sizes = c10::nullopt,
    int64_t dim = 0,
    const std::optional<std::vector<std::optional<GCUStream>>>& streams =
        c10::nullopt);

TORCH_GCU_API at::Tensor& gather_out(at::TensorList tensors,
                                     at::Tensor& out_tensor, int64_t dim);

TORCH_GCU_API at::Tensor gather(at::TensorList tensors, int64_t dim,
                                std::optional<int32_t> destination_index);

}  // namespace torch_gcu
