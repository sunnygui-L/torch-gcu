#pragma once

// ${generated_comment}

#ifdef TORCH_ASSERT_NO_OPERATORS
#error This change adds a dependency on native_functions.yaml,             \
  meaning the file will need to be re-compiled every time an operator      \
  is changed or added. Consider if your change would be better placed in   \
  another file, or if a more specific header might achieve the same goal.  \
  See NOTE: [Tensor vs. TensorBase]
#endif

#if defined(AT_PER_OPERATOR_HEADERS) && defined(TORCH_ASSERT_ONLY_METHOD_OPERATORS)
#error This change adds a dependency on all pytorch operators, meaning the     \
  file will need to be re-compiled every time an operator is changed or added. \
  Consider including a specific operator from <ATen/ops/{my_operator}_ops.h>   \
  and see NOTE [TORCH_ASSERT_ONLY_METHOD_OPERATORS].
#endif

#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/Scalar.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/QScheme.h>
#include <c10/util/OptionalArrayRef.h>
#include <tuple>
#include <vector>

${Operators_includes}

// Separately, XXXX_OP(op) and XXXX_OP2(op, overload) define a class containing compile-time
// metadata about a given xxxx operator.
// Notable data on the class includes:
// - XXXX_OP2(add, Tensor)::name // returns the string name: "add"
// - XXXX_OP2(add, Tensor)::overload_name // returns the string overload name: "Tensor"
// - XXXX_OP2(add, Tensor)::schema // returns the C++ schema type: at::Tensor (const at::Tensor &, const at::Tensor &, const at::Scalar &)
// - XXXX_OP2(add, Tensor)::schema_str // returns the string jit type: "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"

#define ${namespace_defn}_OP2(op_name, overload) ${namespace}::_ops::op_name##_##overload
#define ${namespace_defn}_OP(op_name) ${namespace}::_ops::op_name

namespace ${namespace} {
namespace _ops {
${Operators_declarations}
} // namespace _ops
} // namespace ${namespace}
