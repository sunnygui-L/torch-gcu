/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.
 */
#include "aten/aot_ops/topsaten_bridge.h"
#include "aten/shape_inference/shape_infer_func.h"
#include "gcu/gcu_generator_impl.h"
#include "gcu/philox_gcu_state_raw.h"
#include "topsaten/topsaten_ops.h"

namespace torch_gcu {
namespace aotops {
namespace {

int64_t truncate_to_int32(int64_t value) {
  auto hardware = HardwareType::GetInstance().getHardware();
  const char* backend_name = (hardware == BackendType::kS60)    ? "s60"
                             : (hardware == BackendType::kL600) ? "l600"
                                                                : "gcu";

  if (value > INT32_MAX) {
    TORCH_WARN(
        "The from / to values in the random_ operator are out of the int32 "
        "range, ",
        "which is not supported on the ", backend_name, ", ",
        "and need to be truncated to the int32 data range, ",
        "here, truncate from ", value, " to ", INT32_MAX);
    return static_cast<int64_t>(INT32_MAX);
  }
  if (value < INT32_MIN) {
    TORCH_WARN(
        "The from / to values in the random_ operator are out of the int32 "
        "range, ",
        "which is not supported on the ", backend_name, ", ",
        "and need to be truncated to the int32 data range, ",
        "here, truncate from ", value, " to ", INT32_MIN);
    return static_cast<int64_t>(INT32_MIN);
  }
  return value;
}

}  // namespace

at::Tensor& random_(at::Tensor& self, int64_t from, ::std::optional<int64_t> to,
                    ::std::optional<at::Generator> generator) {
  auto trunc_from = truncate_to_int32(from);
  auto trunc_to = to.has_value()
                      ? ::std::optional<int64_t>(truncate_to_int32(*to))
                      : ::std::nullopt;
  // 1. Get offset and update state in generator
  auto gen = at::get_generator_or_default<GCUGeneratorImpl>(
      generator, getDefaultGCUGenerator());

  // TODO: Use topsaten to get offset
  uint64_t offset = self.numel();
  CHECK_TOPSATEN_CALL(
      topsaten::topsatenRandomGetOffset(offset, topsaten_variable(self).value));
  PhiloxGcuState state = gen->philox_gcu_state(offset);

  // 2. Call topsaten
  bridge_topsatenRandom_out1(self, trunc_from, trunc_to, state);
  return self;
}

at::Tensor& uniform_(at::Tensor& self, double from, double to,
                     ::std::optional<at::Generator> generator) {
  uniform__shape_infer(self, from, to, generator);
  if (self.numel() == 0) return self;

  auto gen = at::get_generator_or_default<GCUGeneratorImpl>(
      generator, getDefaultGCUGenerator());

  uint64_t offset = self.numel();
  CHECK_TOPSATEN_CALL(topsaten::topsatenRngUniformGetOffset(
      offset, topsaten_variable(self).value));
  PhiloxGcuState state = gen->philox_gcu_state(offset);

  bridge_topsatenRngUniform_out1(self, from, to, state);

  return self;
}

}  // namespace aotops
}  // namespace torch_gcu
