/*
 * Copyright 2024-2025 Enflame. All Rights Reserved.
 */
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/ops/clamp_min_meta.h>

#include "ATen/core/op_registration/adaption.h"
#include "aten/shape_inference/shape_infer_func.h"

namespace torch_gcu {

namespace aotops {

void _amp_foreach_non_finite_check_and_unscale__shape_infer(
    at::TensorList scaled_grads, at::Tensor& found_inf,
    const at::Tensor& inv_scale) {
  if (scaled_grads.size() == 0) {
    return;
  }

  TORCH_CHECK(inv_scale.is_privateuseone(), "inv_scale must be a GCU tensor.");
  TORCH_CHECK(found_inf.is_privateuseone(), "found_inf must be a GCU tensor.");
  TORCH_CHECK(inv_scale.numel() == 1, "inv_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(inv_scale.scalar_type() == at::ScalarType::Float,
              "inv_scale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float,
              "found_inf must be a float tensor.");

  // Ensures client code (GradScaler) filtered scaled_grads by dtype.
  at::native::check_foreach_api_restrictions(scaled_grads);

  std::vector<std::vector<at::Tensor>> tensor_lists;

  // is_non_overlapping_and_dense() is not available in Python.
  // GradScaler can't filter for it. We need to filter here.
  if (at::native::can_use_fast_route(scaled_grads)) {
    // Hopefully common case.
    // can_use_fast_route is true, which confirms:
    //  - all scaled_grads are strided
    //  - all scaled_grads are non overlapping and dense
    //  - all scaled_grads are on the same device
    //  - all scaled_grads are of the same dtype
    TORCH_CHECK(scaled_grads[0].is_privateuseone(),
                "scaled_grads must be GCU tensors.");
    // Sets up MTA launch to use scaled_grads as-is.
    tensor_lists.emplace_back(scaled_grads.vec());
  } else {
    // Hopefully uncommon case.
    // can_use_fast_route is an all-or-nothing check.  In this path it was
    // false, so any of the above confirmations could have gone wrong. We filter
    // MTA-safe tensors into an MTA-able list. If a tensor is acceptable but not
    // MTA-safe, we fall back to the TensorIterator kernel. If a tensor is
    // unacceptable, we throw an error to blame GradScaler.
    tensor_lists.resize(1);
    tensor_lists[0].reserve(scaled_grads.size());
    auto expected_device = scaled_grads[0].device();
    const auto expected_dtype = scaled_grads[0].scalar_type();
    for (const at::Tensor& t : scaled_grads) {
      // Ensures GradScaler filtered scaled_grads by device.
      TORCH_CHECK(t.is_privateuseone(),
                  "one of scaled_grads was not a GCU tensor.");
      TORCH_CHECK(t.device() == expected_device,
                  "scaled_grads must be on the same device.");
      TORCH_CHECK(t.layout() == at::kStrided,
                  "one of scaled_grads was not a strided tensor.");
      if (!t.is_non_overlapping_and_dense() ||
          t.scalar_type() != expected_dtype) {
        continue;
      } else {
        tensor_lists[0].push_back(t);
      }
    }
    if (tensor_lists[0].size() == 0) {
      return;
    }
  }
}

at::Tensor& _amp_update_scale__shape_infer(at::Tensor& current_scale,
                                           at::Tensor& growth_tracker,
                                           const at::Tensor& found_inf,
                                           double growth_factor,
                                           double backoff_factor,
                                           int64_t growth_interval) {
  TORCH_CHECK(growth_tracker.is_privateuseone(),
              "growth_tracker must be a GCU tensor.");
  TORCH_CHECK(current_scale.is_privateuseone(),
              "current_scale must be a GCU tensor.");
  TORCH_CHECK(found_inf.is_privateuseone(), "found_inf must be a GCU tensor.");
  TORCH_CHECK(growth_tracker.numel() == 1,
              "growth_tracker must be a 1-element tensor.");
  TORCH_CHECK(current_scale.numel() == 1,
              "current_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(growth_tracker.scalar_type() == at::ScalarType::Int,
              "growth_tracker must be an int tensor.");
  TORCH_CHECK(current_scale.scalar_type() == at::ScalarType::Float,
              "current_scale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float,
              "found_inf must be a float tensor.");
  return current_scale;
}

}  // namespace aotops

}  // namespace torch_gcu