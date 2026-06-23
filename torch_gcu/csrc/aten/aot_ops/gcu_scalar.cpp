#include "aten/aot_ops/gcu_ops.h"
#include "gcu/gcu_functions.h"
#include "gcu/gcu_hardware.h"
#include "gcu/gcu_stream.h"
#include "gcu/gcu_utils.h"

namespace torch_gcu {

namespace aotops {

namespace {

#define GCU_INTEGRAL_TYPES at::kChar, at::kShort, at::kInt, at::kLong
#define GCU_BAREBONES_UNSIGNED_TYPES \
  at::kByte, at::kUInt16, at::kUInt32, at::kUInt64
#define GCU_FLOATING_TYPES at::kHalf, at::kBFloat16, at::kFloat
#define GCU_COMPLEX_TYPES at::kComplexHalf, at::kComplexFloat
#define GCU_FP8_TYPES at::kFloat8_e5m2, at::kFloat8_e4m3fn

}  // namespace

at::Scalar _local_scalar_dense(const at::Tensor& self) {
  // clang-format off
  PTDLOG(OP) << "_local_scalar_dense" << ": {\n"
             << tensorArgsToString({self}, {})
             << "}\n";
  // clang-format on

  at::Scalar r;
  AT_DISPATCH_V2(get_gcu_scalar_type(self.scalar_type()),
                 "_local_scalar_dense_gcu", AT_WRAP([&] {
                   // Create pinned memory for the scalar value to avoid
                   // implicit locking/sync in gcu library due to pageable
                   // memory
                   auto value = at::detail::empty_cpu(
                       {1},                                 /* size */
                       at::CppTypeToScalarType<scalar_t>(), /* dtype */
                       std::nullopt,                        /* layout */
                       std::nullopt,                        /* device */
                       true,                                /* pin_memory */
                       std::nullopt                         /* memory format */
                   );
                   topsStream_t stream = getCurrentGCUStream();
                   memcpy_and_sync((void*)value.const_data_ptr<scalar_t>(),
                                   gcu_data_ptr(self), sizeof(scalar_t),
                                   topsMemcpyDeviceToHost, stream);
                   r = at::Scalar(*value.const_data_ptr<scalar_t>());
                 }),
                 at::kBool, GCU_INTEGRAL_TYPES, GCU_BAREBONES_UNSIGNED_TYPES,
                 GCU_FLOATING_TYPES, GCU_COMPLEX_TYPES, GCU_FP8_TYPES);
  if (is_narrow_type(self.scalar_type())) {
    AT_DISPATCH_V2(self.scalar_type(), "_local_scalar_dense_gcu_type_narrow",
                   AT_WRAP([&] { r = at::Scalar(r.to<scalar_t>()); }),
                   at::kDouble, at::kLong, at::kUInt64, at::kComplexDouble);
  }

  return r;
}

}  // namespace aotops

}  // namespace torch_gcu
