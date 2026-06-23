/*
 * Copyright 2023-2024 Enflame. All Rights Reserved.
 */

#include <ATen/AccumulateType.h>
#include <ATen/core/op_registration/adaption.h>

#include "aten/aot_ops/gcu_ops.h"

#include <ATen/AccumulateType.h>
#include "aten/shape_inference/shape_infer_func.h"
namespace torch_gcu {

namespace aotops {

at::Tensor& arange_out_shape_infer(const at::Scalar& start,
                                   const at::Scalar& end,
                                   const at::Scalar& step, at::Tensor& result) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::kHalf, at::kBFloat16, result.scalar_type(), "arange_gcu_shape_infer",
      [&]() {
        using accscalar_t = at::acc_type<scalar_t, false>;
        auto xstart = start.to<accscalar_t>();
        auto xend = end.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

        TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
        TORCH_CHECK(std::isfinite(static_cast<double>(xstart)) &&
                        std::isfinite(static_cast<double>(xend)),
                    "unsupported range: ", xstart, " -> ", xend);
        TORCH_CHECK(((xstep > 0) && (xend >= xstart)) ||
                        ((xstep < 0) && (xend <= xstart)),
                    "upper bound and larger bound inconsistent with step sign");

        // we use double precision for (start - end) / step
        // to compute size_d for consistency across devices.
        // The problem with using accscalar_t is that accscalar_t might be
        // float32 on gpu for a float32 scalar_t, but double on cpu for the
        // same, and the effective output size starts differing on CPU vs GPU
        // because of precision issues, which we dont want. the corner-case we
        // do want to take into account is int64_t, which has higher precision
        // than double
        double size_d;
        if constexpr (std::is_same_v<scalar_t, int64_t>) {
          int64_t sgn = (xstep > 0) - (xstep < 0);
          size_d = std::ceil((xend - xstart + xstep - sgn) / xstep);
        } else {
          size_d = std::ceil(
              static_cast<double>(end.to<double>() - start.to<double>()) /
              step.to<double>());
        }

        TORCH_CHECK(
            size_d >= 0 && size_d <= static_cast<double>(
                                         std::numeric_limits<int64_t>::max()),
            "invalid size, possible overflow?");

        int64_t size = static_cast<int64_t>(size_d);
        int64_t numel = result.numel();

        if (numel != size) {
          if (numel > 0) {
            TORCH_WARN(
                "The number of elements in the out tensor of shape ",
                result.sizes(), " is ", numel,
                " which does not match the computed number of elements ", size,
                ". Note that this may occur as a result of rounding error. "
                "The out tensor will be resized to a tensor of shape (",
                size, ",).");
          }
          result.resize_({size});
        }

        return;
      });

  return result;
}

at::Tensor& linspace_out_shape_infer(const at::Scalar& start,
                                     const at::Scalar& end, int64_t steps,
                                     at::Tensor& result) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");
  if (result.numel() != steps) {
    aotops::resize_(result, {steps}, c10::nullopt);
  }
  return result;
}

at::Tensor& range_out_shape_infer(const at::Scalar& start,
                                  const at::Scalar& end, const at::Scalar& step,
                                  at::Tensor& out) {
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device;  // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, out, __FUNCTION__,
                                            "out");
  const at::OptionalDeviceGuard device_guard(device_of(out));

  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, out.scalar_type(), "range_out", [&]() {
        using accscalar_t = at::acc_type<scalar_t, true>;
        auto xstart = start.to<accscalar_t>();
        auto xend = end.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

        TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
        TORCH_CHECK(std::isfinite(static_cast<double>(xstart)) &&
                        std::isfinite(static_cast<double>(xend)),
                    "unsupported range: ", xstart, " -> ", xend);
        TORCH_CHECK(((xstep > 0) && (xend >= xstart)) ||
                        ((xstep < 0) && (xend <= xstart)),
                    "upper bound and larger bound inconsistent with step sign");
        int64_t size = static_cast<int64_t>(((xend - xstart) / xstep) + 1);

        if (out.numel() != size) {
          aotops::resize_(out, {size}, c10::nullopt);
        }
      });

  return out;
}

at::Tensor& logspace_out_shape_infer(const at::Scalar& start,
                                     const at::Scalar& end, int64_t steps,
                                     double base, at::Tensor& out) {
  aotops::resize_(out, {steps}, c10::nullopt);
  return out;
}

}  // namespace aotops

}  // namespace torch_gcu
