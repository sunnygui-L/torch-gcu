#include "aten/foreach_op_utils.h"

#include <ATen/ScalarOps.h>
#include <ATen/native/ForeachUtils.h>
namespace torch_gcu {

namespace aotops {
/********************** check slow path functions **********************/
#define FOREACH_BINARY_OP_LIST_ALPHA_CHECK_SLOW_PATH(NAME)               \
  bool _foreach_##NAME##_check_slow_path(at::TensorList tensors1,        \
                                         at::TensorList tensors2,        \
                                         const at::Scalar& alpha) {      \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);      \
    return !at::native::can_use_fast_route({tensors1, tensors2}, alpha); \
  }                                                                      \
                                                                         \
  bool _foreach_##NAME##__check_slow_path(at::TensorList tensors1,       \
                                          at::TensorList tensors2,       \
                                          const at::Scalar& alpha) {     \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);      \
    return !at::native::can_use_fast_route({tensors1, tensors2}, alpha); \
  }

#define FOREACH_BINARY_OP_LIST_CHECK_SLOW_PATH(NAME, DIVISION_OP)            \
  bool _foreach_##NAME##_check_slow_path(at::TensorList tensors1,            \
                                         at::TensorList tensors2) {          \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);          \
    return !at::native::can_use_fast_route(tensors1, tensors2, DIVISION_OP); \
  }                                                                          \
                                                                             \
  bool _foreach_##NAME##__check_slow_path(at::TensorList tensors1,           \
                                          at::TensorList tensors2) {         \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);          \
    return !at::native::can_use_fast_route(tensors1, tensors2, DIVISION_OP); \
  }

#define FOREACH_BINARY_OP_SCALAR_CHECK_SLOW_PATH(NAME, DIVISION_OP)       \
  bool _foreach_##NAME##_check_slow_path(at::TensorList tensors,          \
                                         const at::Scalar& scalar) {      \
    at::native::check_foreach_api_restrictions(tensors);                  \
    return !at::native::can_use_fast_route(tensors, scalar, DIVISION_OP); \
  }                                                                       \
                                                                          \
  bool _foreach_##NAME##__check_slow_path(at::TensorList tensors,         \
                                          const at::Scalar& scalar) {     \
    at::native::check_foreach_api_restrictions(tensors);                  \
    return !at::native::can_use_fast_route(tensors, scalar, DIVISION_OP); \
  }

#define FOREACH_BINARY_OP_SCALARLIST_CHECK_SLOW_PATH(NAME, DIV_OP)            \
  bool _foreach_##NAME##_check_slow_path(at::TensorList tensors,              \
                                         at::ArrayRef<at::Scalar> scalars) {  \
    at::native::check_foreach_api_restrictions(tensors, scalars);             \
    return !at::native::can_use_fast_route(tensors, scalars, DIV_OP);         \
  }                                                                           \
                                                                              \
  bool _foreach_##NAME##__check_slow_path(at::TensorList tensors,             \
                                          at::ArrayRef<at::Scalar> scalars) { \
    at::native::check_foreach_api_restrictions(tensors, scalars);             \
    return !at::native::can_use_fast_route(tensors, scalars, DIV_OP);         \
  }

#define FOREACH_BINARY_OP_SCALAR_TENSOR_CHECK_SLOW_PATH(NAME, DIVISION_OP)  \
  bool _foreach_##NAME##_check_slow_path(at::TensorList tensors,            \
                                         const at::Tensor& scalar) {        \
    at::native::check_foreach_api_restrictions(tensors);                    \
    return !(at::native::can_use_fast_route(                                \
                 at::ArrayRef<at::TensorList>{tensors}, {}, DIVISION_OP) && \
             tensors[0].scalar_type() == scalar.scalar_type());             \
  }                                                                         \
                                                                            \
  bool _foreach_##NAME##__check_slow_path(at::TensorList tensors,           \
                                          const at::Tensor& scalar) {       \
    at::native::check_foreach_api_restrictions(tensors);                    \
    return !(at::native::can_use_fast_route(                                \
                 at::ArrayRef<at::TensorList>{tensors}, {}, DIVISION_OP) && \
             tensors[0].scalar_type() == scalar.scalar_type());             \
  }

#define FOREACH_POINTWISE_OP_SCALAR_CHECK_SLOW_PATH(NAME)                     \
  bool _foreach_##NAME##_check_slow_path(                                     \
      at::TensorList input, at::TensorList tensors1, at::TensorList tensors2, \
      const at::Scalar& scalar) {                                             \
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2);    \
                                                                              \
    return (!at::native::can_use_fast_route({input, tensors1, tensors2},      \
                                            scalar) ||                        \
            at::native::has_integral_tensor(input, /* includeBool */ true));  \
  }                                                                           \
                                                                              \
  bool _foreach_##NAME##__check_slow_path(                                    \
      at::TensorList input, at::TensorList tensors1, at::TensorList tensors2, \
      const at::Scalar& scalar) {                                             \
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2);    \
                                                                              \
    return (!at::native::can_use_fast_route({input, tensors1, tensors2},      \
                                            scalar) ||                        \
            at::native::has_integral_tensor(input, /* includeBool */ true));  \
  }

#define FOREACH_POINTWISE_OP_SCALARLIST_CHECK_SLOW_PATH(NAME)                 \
  bool _foreach_##NAME##_check_slow_path(                                     \
      at::TensorList input, at::TensorList tensors1, at::TensorList tensors2, \
      at::ArrayRef<at::Scalar> scalars) {                                     \
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2,     \
                                               scalars);                      \
                                                                              \
    return (!at::native::can_use_fast_route({input, tensors1, tensors2},      \
                                            scalars) ||                       \
            at::native::has_integral_tensor(input, /* includeBool */ true));  \
  }                                                                           \
                                                                              \
  bool _foreach_##NAME##__check_slow_path(                                    \
      at::TensorList input, at::TensorList tensors1, at::TensorList tensors2, \
      at::ArrayRef<at::Scalar> scalars) {                                     \
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2,     \
                                               scalars);                      \
                                                                              \
    return (!at::native::can_use_fast_route({input, tensors1, tensors2},      \
                                            scalars) ||                       \
            at::native::has_integral_tensor(input, /* includeBool */ true));  \
  }

#define FOREACH_TERNARY_OP_CHECK_SLOW_PATH(NAME)                              \
  bool _foreach_##NAME##_check_slow_path(at::TensorList tensors1,             \
                                         at::TensorList tensors2,             \
                                         at::TensorList tensors3) {           \
    at::native::check_foreach_api_restrictions(tensors1, tensors2, tensors3); \
    return !at::native::can_use_fast_route({tensors1, tensors2, tensors3});   \
  }                                                                           \
                                                                              \
  bool _foreach_##NAME##__check_slow_path(at::TensorList tensors1,            \
                                          at::TensorList tensors2,            \
                                          at::TensorList tensors3) {          \
    at::native::check_foreach_api_restrictions(tensors1, tensors2, tensors3); \
    return !at::native::can_use_fast_route({tensors1, tensors2, tensors3});   \
  }

#define FOREACH_TERNARY_OP_SCALAR_CHECK_SLOW_PATH(NAME)               \
  bool _foreach_##NAME##_check_slow_path(at::TensorList tensors1,     \
                                         at::TensorList tensors2,     \
                                         const at::Scalar& weight) {  \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);   \
    return !at::native::can_use_fast_route({tensors1, tensors2});     \
  }                                                                   \
                                                                      \
  bool _foreach_##NAME##__check_slow_path(at::TensorList tensors1,    \
                                          at::TensorList tensors2,    \
                                          const at::Scalar& weight) { \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);   \
    return !at::native::can_use_fast_route({tensors1, tensors2});     \
  }

#define FOREACH_UNARY_OP_CHECK_SLOW_PATH(NAME)                                 \
  bool _foreach_##NAME##_check_slow_path(at::TensorList tensors) {             \
    at::native::check_foreach_api_restrictions(tensors);                       \
    return (!at::native::can_use_fast_route(tensors) ||                        \
            at::native::has_integral_tensor(tensors, /* includeBool */ true)); \
  }                                                                            \
                                                                               \
  bool _foreach_##NAME##__check_slow_path(at::TensorList tensors) {            \
    at::native::check_foreach_api_restrictions(tensors);                       \
    return (!at::native::can_use_fast_route(tensors) ||                        \
            at::native::has_integral_tensor(tensors, /* includeBool */ true)); \
  }

FOREACH_UNARY_OP_CHECK_SLOW_PATH(atan)
FOREACH_UNARY_OP_CHECK_SLOW_PATH(ceil)
FOREACH_UNARY_OP_CHECK_SLOW_PATH(cos)
FOREACH_UNARY_OP_CHECK_SLOW_PATH(cosh)
FOREACH_UNARY_OP_CHECK_SLOW_PATH(sqrt)
FOREACH_UNARY_OP_CHECK_SLOW_PATH(expm1)
FOREACH_UNARY_OP_CHECK_SLOW_PATH(erf)
FOREACH_UNARY_OP_CHECK_SLOW_PATH(erfc)
FOREACH_UNARY_OP_CHECK_SLOW_PATH(floor)
FOREACH_UNARY_OP_CHECK_SLOW_PATH(asin)
FOREACH_UNARY_OP_CHECK_SLOW_PATH(log2)
FOREACH_UNARY_OP_CHECK_SLOW_PATH(acos)
FOREACH_UNARY_OP_CHECK_SLOW_PATH(exp)

FOREACH_BINARY_OP_LIST_ALPHA_CHECK_SLOW_PATH(add);
FOREACH_BINARY_OP_SCALAR_CHECK_SLOW_PATH(add, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST_CHECK_SLOW_PATH(add, /*div_op*/ false);

FOREACH_BINARY_OP_LIST_ALPHA_CHECK_SLOW_PATH(sub);
FOREACH_BINARY_OP_SCALAR_CHECK_SLOW_PATH(sub, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST_CHECK_SLOW_PATH(sub, /*div_op*/ false);

FOREACH_BINARY_OP_LIST_CHECK_SLOW_PATH(div, true)
FOREACH_BINARY_OP_SCALAR_CHECK_SLOW_PATH(div, true)
FOREACH_BINARY_OP_SCALARLIST_CHECK_SLOW_PATH(div, true)
FOREACH_BINARY_OP_SCALAR_TENSOR_CHECK_SLOW_PATH(div, true)

FOREACH_BINARY_OP_LIST_CHECK_SLOW_PATH(mul, false)
FOREACH_BINARY_OP_SCALAR_CHECK_SLOW_PATH(mul, false)
FOREACH_BINARY_OP_SCALARLIST_CHECK_SLOW_PATH(mul, false)
FOREACH_BINARY_OP_SCALAR_TENSOR_CHECK_SLOW_PATH(mul, false)

FOREACH_BINARY_OP_LIST_CHECK_SLOW_PATH(clamp_max, /*division_op*/ true)
FOREACH_BINARY_OP_SCALAR_CHECK_SLOW_PATH(clamp_max, /*division_op*/ true)
FOREACH_BINARY_OP_SCALARLIST_CHECK_SLOW_PATH(clamp_max, /*division_op*/ true)

FOREACH_BINARY_OP_LIST_CHECK_SLOW_PATH(clamp_min, /*division_op*/ true)
FOREACH_BINARY_OP_SCALAR_CHECK_SLOW_PATH(clamp_min, /*division_op*/ true)
FOREACH_BINARY_OP_SCALARLIST_CHECK_SLOW_PATH(clamp_min, /*division_op*/ true)

FOREACH_BINARY_OP_LIST_CHECK_SLOW_PATH(minimum, /*division_op*/ true)
FOREACH_BINARY_OP_SCALAR_CHECK_SLOW_PATH(minimum, /*division_op*/ true)
FOREACH_BINARY_OP_SCALARLIST_CHECK_SLOW_PATH(minimum, /*division_op*/ true)

FOREACH_BINARY_OP_LIST_CHECK_SLOW_PATH(maximum, /*division_op*/ true)
FOREACH_BINARY_OP_SCALAR_CHECK_SLOW_PATH(maximum, /*division_op*/ true)
FOREACH_BINARY_OP_SCALARLIST_CHECK_SLOW_PATH(maximum, /*division_op*/ true)

FOREACH_POINTWISE_OP_SCALAR_CHECK_SLOW_PATH(addcdiv)
FOREACH_POINTWISE_OP_SCALARLIST_CHECK_SLOW_PATH(addcdiv)

FOREACH_POINTWISE_OP_SCALAR_CHECK_SLOW_PATH(addcmul)
FOREACH_POINTWISE_OP_SCALARLIST_CHECK_SLOW_PATH(addcmul)

FOREACH_TERNARY_OP_CHECK_SLOW_PATH(lerp)
FOREACH_TERNARY_OP_SCALAR_CHECK_SLOW_PATH(lerp)

bool _foreach_norm_check_slow_path(at::TensorList self, const at::Scalar& ord,
                                   std::optional<c10::ScalarType> dtype) {
  double p;
  if (ord.isIntegral(false)) {
    p = ord.to<int64_t>();
  } else if (ord.isFloatingPoint()) {
    p = ord.to<double>();
  } else {
    TORCH_CHECK(false, "_foreach_norm expects ord to be integer or float");
  }
  at::native::check_foreach_api_restrictions(self);
  return (!at::native::can_use_fast_route(self) ||
          !(p == static_cast<double>(1) || p == static_cast<double>(2)));
}

/********************** slow path functions **********************/
// NOTE: check_foreach_api_restrictions already called in check_slow_path
// functions, so wo don't need to call it again.
#define FOREACH_BINARY_OP_TENSOR_SLOW_PATH(OP)                    \
  void _foreach_##OP##__slow_path(at::TensorList tensors,         \
                                  const at::Tensor& scalar) {     \
    TORCH_CHECK(scalar.dim() == 0 && scalar.numel() == 1,         \
                "scalar tensor expected to be 0 dim but it has ", \
                scalar.dim(), " dimensions and ", scalar.numel(), \
                " elements.");                                    \
    for (auto& t : tensors) {                                     \
      t.OP##_(scalar);                                            \
    }                                                             \
  }                                                               \
                                                                  \
  std::vector<at::Tensor> _foreach_##OP##_slow_path(              \
      at::TensorList tensors, const at::Tensor& scalar) {         \
    TORCH_CHECK(scalar.dim() == 0 && scalar.numel() == 1,         \
                "scalar tensor expected to be 0 dim but it has ", \
                scalar.dim(), " dimensions and ", scalar.numel(), \
                " elements.");                                    \
    std::vector<at::Tensor> result;                               \
    result.reserve(tensors.size());                               \
    for (const auto& t : tensors) {                               \
      result.emplace_back(t.OP(scalar));                          \
    }                                                             \
                                                                  \
    return result;                                                \
  }

#define FOREACH_BINARY_OP_TENSOR_ALPHA_SLOW_PATH(OP)                           \
  void _foreach_##OP##__slow_path(at::TensorList tensors,                      \
                                  const at::Tensor& scalar,                    \
                                  const at::Scalar& alpha) {                   \
    TORCH_CHECK(scalar.dim() == 0 && scalar.numel() == 1,                      \
                "scalar tensor expected to be 0 dim but it has ",              \
                scalar.dim(), " dimensions and ", scalar.numel(),              \
                " elements.");                                                 \
    for (auto& t : tensors) {                                                  \
      t.OP##_(scalar, alpha);                                                  \
    }                                                                          \
  }                                                                            \
                                                                               \
  std::vector<at::Tensor> _foreach_##OP##_slow_path(at::TensorList tensors,    \
                                                    const at::Tensor& scalar,  \
                                                    const at::Scalar& alpha) { \
    TORCH_CHECK(scalar.dim() == 0 && scalar.numel() == 1,                      \
                "scalar tensor expected to be 0 dim but it has ",              \
                scalar.dim(), " dimensions and ", scalar.numel(),              \
                " elements.");                                                 \
    std::vector<at::Tensor> result;                                            \
    result.reserve(tensors.size());                                            \
    for (const auto& t : tensors) {                                            \
      result.emplace_back(t.OP(scalar, alpha));                                \
    }                                                                          \
                                                                               \
    return result;                                                             \
  }

#define FOREACH_BINARY_OP_SCALAR_SLOW_PATH(OP) \
  FOREACH_BINARY_OP_SCALAR_SLOW_PATH_REALOP(OP, OP)

#define FOREACH_BINARY_OP_SCALAR_SLOW_PATH_REALOP(OP, REALOP) \
  void _foreach_##OP##__slow_path(at::TensorList tensors,     \
                                  const at::Scalar& scalar) { \
    for (auto& t : tensors) {                                 \
      t.REALOP##_(scalar);                                    \
    }                                                         \
  }                                                           \
                                                              \
  std::vector<at::Tensor> _foreach_##OP##_slow_path(          \
      at::TensorList tensors, const at::Scalar& scalar) {     \
    std::vector<at::Tensor> result;                           \
    result.reserve(tensors.size());                           \
    for (const auto& t : tensors) {                           \
      result.emplace_back(t.REALOP(scalar));                  \
    }                                                         \
                                                              \
    return result;                                            \
  }

#define FOREACH_BINARY_OP_SCALARLIST_SLOW_PATH(OP) \
  FOREACH_BINARY_OP_SCALARLIST_SLOW_PATH_REALOP(OP, OP)

#define FOREACH_BINARY_OP_SCALARLIST_SLOW_PATH_REALOP(OP, REALOP)     \
  void _foreach_##OP##__slow_path(at::TensorList tensors,             \
                                  at::ArrayRef<at::Scalar> scalars) { \
    for (const auto i : c10::irange(tensors.size())) {                \
      tensors[i].REALOP##_(scalars[i]);                               \
    }                                                                 \
  }                                                                   \
                                                                      \
  std::vector<at::Tensor> _foreach_##OP##_slow_path(                  \
      at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {     \
    std::vector<at::Tensor> result;                                   \
    result.reserve(tensors.size());                                   \
    for (const auto i : c10::irange(tensors.size())) {                \
      result.emplace_back(tensors[i].REALOP(scalars[i]));             \
    }                                                                 \
                                                                      \
    return result;                                                    \
  }

#define FOREACH_BINARY_OP_LIST_SLOW_PATH(OP) \
  FOREACH_BINARY_OP_LIST_SLOW_PATH_REALOP(OP, OP)

#define FOREACH_BINARY_OP_LIST_SLOW_PATH_REALOP(OP, REALOP)                    \
  std::vector<at::Tensor> _foreach_##OP##_slow_path(at::TensorList tensors1,   \
                                                    at::TensorList tensors2) { \
    std::vector<at::Tensor> result;                                            \
    result.reserve(tensors1.size());                                           \
    for (const auto i : c10::irange(tensors1.size())) {                        \
      result.emplace_back(tensors1[i].REALOP(tensors2[i]));                    \
    }                                                                          \
                                                                               \
    return result;                                                             \
  }                                                                            \
                                                                               \
  void _foreach_##OP##__slow_path(at::TensorList tensors1,                     \
                                  at::TensorList tensors2) {                   \
    for (const auto i : c10::irange(tensors1.size())) {                        \
      tensors1[i].REALOP##_(tensors2[i]);                                      \
    }                                                                          \
  }

#define FOREACH_BINARY_OP_LIST_ALPHA_SLOW_PATH(OP)                             \
  std::vector<at::Tensor> _foreach_##OP##_slow_path(at::TensorList tensors1,   \
                                                    at::TensorList tensors2,   \
                                                    const at::Scalar& alpha) { \
    std::vector<at::Tensor> result;                                            \
    result.reserve(tensors1.size());                                           \
    for (const auto i : c10::irange(tensors1.size())) {                        \
      result.emplace_back(tensors1[i].OP(tensors2[i], alpha));                 \
    }                                                                          \
                                                                               \
    return result;                                                             \
  }                                                                            \
                                                                               \
  void _foreach_##OP##__slow_path(at::TensorList tensors1,                     \
                                  at::TensorList tensors2,                     \
                                  const at::Scalar& alpha) {                   \
    for (const auto i : c10::irange(tensors1.size())) {                        \
      tensors1[i].OP##_(tensors2[i], alpha);                                   \
    }                                                                          \
  }

#define FOREACH_UNARY_OP_SLOW_PATH(OP)                                        \
  std::vector<at::Tensor> _foreach_##OP##_slow_path(at::TensorList tensors) { \
    std::vector<at::Tensor> result;                                           \
    result.reserve(tensors.size());                                           \
    for (const auto& t : tensors) {                                           \
      result.emplace_back(t.OP());                                            \
    }                                                                         \
                                                                              \
    return result;                                                            \
  }                                                                           \
                                                                              \
  void _foreach_##OP##__slow_path(at::TensorList tensors) {                   \
    for (auto& t : tensors) {                                                 \
      t.OP##_();                                                              \
    }                                                                         \
  }

#define FOREACH_POINTWISE_OP_SCALAR_SLOW_PATH(OP)                             \
  std::vector<at::Tensor> _foreach_##OP##_slow_path(                          \
      at::TensorList input, at::TensorList tensors1, at::TensorList tensors2, \
      const at::Scalar& scalar) {                                             \
    std::vector<at::Tensor> result;                                           \
    for (const auto i : c10::irange(input.size())) {                          \
      result.emplace_back(input[i].OP(tensors1[i], tensors2[i], scalar));     \
    }                                                                         \
                                                                              \
    return result;                                                            \
  }                                                                           \
                                                                              \
  void _foreach_##OP##__slow_path(                                            \
      at::TensorList input, at::TensorList tensors1, at::TensorList tensors2, \
      const at::Scalar& scalar) {                                             \
    for (const auto i : c10::irange(input.size())) {                          \
      input[i].OP##_(tensors1[i], tensors2[i], scalar);                       \
    }                                                                         \
  }

#define FOREACH_POINTWISE_OP_SCALARLIST_SLOW_PATH(OP)                         \
  std::vector<at::Tensor> _foreach_##OP##_slow_path(                          \
      at::TensorList input, at::TensorList tensors1, at::TensorList tensors2, \
      at::ArrayRef<at::Scalar> scalars) {                                     \
    std::vector<at::Tensor> result;                                           \
    for (const auto i : c10::irange(input.size())) {                          \
      result.emplace_back(input[i].OP(tensors1[i], tensors2[i], scalars[i])); \
    }                                                                         \
                                                                              \
    return result;                                                            \
  }                                                                           \
                                                                              \
  void _foreach_##OP##__slow_path(                                            \
      at::TensorList input, at::TensorList tensors1, at::TensorList tensors2, \
      at::ArrayRef<at::Scalar> scalars) {                                     \
    for (const auto i : c10::irange(input.size())) {                          \
      input[i].OP##_(tensors1[i], tensors2[i], scalars[i]);                   \
    }                                                                         \
  }

#define FOREACH_POINTWISE_OP_TENSOR_SLOW_PATH(OP)                             \
  std::vector<at::Tensor> _foreach_##OP##_slow_path(                          \
      at::TensorList input, at::TensorList tensors1, at::TensorList tensors2, \
      const at::Tensor& scalars_) {                                           \
    auto scalars =                                                            \
        at::native::convert_tensor_to_scalar_list(scalars_, input.size());    \
    return _foreach_##OP##_slow_path(input, tensors1, tensors2, scalars);     \
  }                                                                           \
                                                                              \
  void _foreach_##OP##__slow_path(                                            \
      at::TensorList input, at::TensorList tensors1, at::TensorList tensors2, \
      const at::Tensor& scalars_) {                                           \
    auto scalars =                                                            \
        at::native::convert_tensor_to_scalar_list(scalars_, input.size());    \
    _foreach_##OP##__slow_path(input, tensors1, tensors2, scalars);           \
  }

FOREACH_BINARY_OP_LIST_ALPHA_SLOW_PATH(add);
FOREACH_BINARY_OP_LIST_ALPHA_SLOW_PATH(sub);
FOREACH_BINARY_OP_LIST_ALPHA_SLOW_PATH(lerp);

FOREACH_BINARY_OP_TENSOR_ALPHA_SLOW_PATH(add);
FOREACH_BINARY_OP_TENSOR_ALPHA_SLOW_PATH(sub);
FOREACH_BINARY_OP_TENSOR_SLOW_PATH(mul);
FOREACH_BINARY_OP_TENSOR_SLOW_PATH(div);

FOREACH_BINARY_OP_SCALAR_SLOW_PATH(add);
FOREACH_BINARY_OP_SCALAR_SLOW_PATH(sub);
FOREACH_BINARY_OP_SCALAR_SLOW_PATH(mul);
FOREACH_BINARY_OP_SCALAR_SLOW_PATH(div);
FOREACH_BINARY_OP_SCALAR_SLOW_PATH(clamp_min);
FOREACH_BINARY_OP_SCALAR_SLOW_PATH(clamp_max);
FOREACH_BINARY_OP_SCALAR_SLOW_PATH(pow);
FOREACH_BINARY_OP_SCALAR_SLOW_PATH_REALOP(minimum, clamp_max);
FOREACH_BINARY_OP_SCALAR_SLOW_PATH_REALOP(maximum, clamp_min);

FOREACH_BINARY_OP_SCALARLIST_SLOW_PATH(add);
FOREACH_BINARY_OP_SCALARLIST_SLOW_PATH(sub);
FOREACH_BINARY_OP_SCALARLIST_SLOW_PATH(mul);
FOREACH_BINARY_OP_SCALARLIST_SLOW_PATH(div);
FOREACH_BINARY_OP_SCALARLIST_SLOW_PATH(clamp_min);
FOREACH_BINARY_OP_SCALARLIST_SLOW_PATH(clamp_max);
FOREACH_BINARY_OP_SCALARLIST_SLOW_PATH(pow);
FOREACH_BINARY_OP_SCALARLIST_SLOW_PATH_REALOP(minimum, clamp_max);
FOREACH_BINARY_OP_SCALARLIST_SLOW_PATH_REALOP(maximum, clamp_min);

FOREACH_BINARY_OP_LIST_SLOW_PATH(mul);
FOREACH_BINARY_OP_LIST_SLOW_PATH(div);
FOREACH_BINARY_OP_LIST_SLOW_PATH(clamp_min);
FOREACH_BINARY_OP_LIST_SLOW_PATH(clamp_max);
FOREACH_BINARY_OP_LIST_SLOW_PATH(pow);
FOREACH_BINARY_OP_LIST_SLOW_PATH_REALOP(minimum, clamp_max);
FOREACH_BINARY_OP_LIST_SLOW_PATH_REALOP(maximum, clamp_min);
// _foreach_copy_
void foreach_tensor_copy_list_kernel_slow_(at::TensorList self,
                                           at::TensorList src,
                                           const bool non_blocking) {
  for (const auto i : c10::irange(self.size())) {
    self[i].copy_(src[i], non_blocking);
  }
}

FOREACH_UNARY_OP_SLOW_PATH(sqrt);
FOREACH_UNARY_OP_SLOW_PATH(exp);
FOREACH_UNARY_OP_SLOW_PATH(abs);
FOREACH_UNARY_OP_SLOW_PATH(acos);
FOREACH_UNARY_OP_SLOW_PATH(asin);
FOREACH_UNARY_OP_SLOW_PATH(atan);
FOREACH_UNARY_OP_SLOW_PATH(ceil);
FOREACH_UNARY_OP_SLOW_PATH(cos);
FOREACH_UNARY_OP_SLOW_PATH(cosh);
FOREACH_UNARY_OP_SLOW_PATH(erf);
FOREACH_UNARY_OP_SLOW_PATH(erfc);
FOREACH_UNARY_OP_SLOW_PATH(expm1);
FOREACH_UNARY_OP_SLOW_PATH(floor);
FOREACH_UNARY_OP_SLOW_PATH(log);
FOREACH_UNARY_OP_SLOW_PATH(log10);
FOREACH_UNARY_OP_SLOW_PATH(log1p);
FOREACH_UNARY_OP_SLOW_PATH(log2);
FOREACH_UNARY_OP_SLOW_PATH(neg);
FOREACH_UNARY_OP_SLOW_PATH(tan);
FOREACH_UNARY_OP_SLOW_PATH(tanh);
FOREACH_UNARY_OP_SLOW_PATH(sin);
FOREACH_UNARY_OP_SLOW_PATH(sinh);
FOREACH_UNARY_OP_SLOW_PATH(round);
FOREACH_UNARY_OP_SLOW_PATH(lgamma);
FOREACH_UNARY_OP_SLOW_PATH(frac);
FOREACH_UNARY_OP_SLOW_PATH(trunc);
FOREACH_UNARY_OP_SLOW_PATH(reciprocal);
FOREACH_UNARY_OP_SLOW_PATH(sigmoid);
FOREACH_UNARY_OP_SLOW_PATH(sign);

FOREACH_POINTWISE_OP_SCALAR_SLOW_PATH(addcdiv);
FOREACH_POINTWISE_OP_SCALAR_SLOW_PATH(addcmul);

FOREACH_POINTWISE_OP_SCALARLIST_SLOW_PATH(addcdiv);
FOREACH_POINTWISE_OP_SCALARLIST_SLOW_PATH(addcmul);

FOREACH_POINTWISE_OP_TENSOR_SLOW_PATH(addcdiv);
FOREACH_POINTWISE_OP_TENSOR_SLOW_PATH(addcmul);

#define FOREACH_TERNARY_OP_SLOW_PATH(OP)                                       \
  std::vector<at::Tensor> _foreach_##OP##_slow_path(at::TensorList tensors1,   \
                                                    at::TensorList tensors2,   \
                                                    at::TensorList tensors3) { \
    std::vector<at::Tensor> result;                                            \
    for (const auto i : c10::irange(tensors1.size())) {                        \
      result.emplace_back(tensors1[i].OP(tensors2[i], tensors3[i]));           \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
                                                                               \
  void _foreach_##OP##__slow_path(at::TensorList tensors1,                     \
                                  at::TensorList tensors2,                     \
                                  at::TensorList tensors3) {                   \
    for (const auto i : c10::irange(tensors1.size())) {                        \
      tensors1[i].OP##_(tensors2[i], tensors3[i]);                             \
    }                                                                          \
  }

FOREACH_TERNARY_OP_SLOW_PATH(lerp);

void _foreach_zero__slow_path(at::TensorList tensors) {
  for (auto& t : tensors) {
    t.zero_();
  }
}

std::vector<at::Tensor> _foreach_norm_slow_path(
    at::TensorList tensors, const at::Scalar& ord,
    std::optional<at::ScalarType> dtype) {
  std::vector<at::Tensor> result;
  for (const auto& t : tensors) {
    result.emplace_back(at::linalg_vector_norm(t, ord, {}, false, dtype));
  }
  return result;
}

std::vector<at::Tensor> _foreach_pow_slow_path(const at::Scalar& self,
                                               at::TensorList exponent) {
  std::vector<at::Tensor> result;
  result.reserve(exponent.size());
  for (const auto& t : exponent) {
    result.emplace_back(at::pow(self, t));
  }
  return result;
}

}  // namespace aotops
}  // namespace torch_gcu
