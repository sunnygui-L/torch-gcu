
/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#include "aten/aot_ops/gcu_op_check.h"

#include <ATen/core/Array.h>

#include <string>

#include "aten/aten_cpu_fallback.h"
namespace torch_gcu {

namespace {

constexpr auto f2 = c10::ScalarType::Half;
constexpr auto f4 = c10::ScalarType::Float;
constexpr auto f8 = c10::ScalarType::Double;
constexpr auto c2 = c10::ScalarType::ComplexHalf;
constexpr auto c4 = c10::ScalarType::ComplexFloat;
constexpr auto c8 = c10::ScalarType::ComplexDouble;
constexpr auto bf = c10::ScalarType::BFloat16;

constexpr std::array<c10::ScalarType, 7> index2dtype{f2, f4, f8, c2,
                                                     c4, c8, bf};

constexpr std::array<int64_t, static_cast<size_t>(c10::ScalarType::NumOptions)>
calculate_dtype2index() {
  std::array<int64_t, static_cast<size_t>(c10::ScalarType::NumOptions)>
      inverse = {};
  for (int64_t i = 0; i < static_cast<int64_t>(c10::ScalarType::NumOptions);
       i++) {
    inverse[i] = -1;
  }
  for (int64_t i = 0; i < static_cast<int64_t>(index2dtype.size()); i++) {
    inverse[static_cast<int64_t>(index2dtype[i])] = i;
  }
  return inverse;
}

constexpr auto dtype2index = calculate_dtype2index();

}  // namespace

at::ScalarType get_lower_scalar_type(at::ScalarType type_a,
                                     at::ScalarType type_b) {
  if (at::isFloatingType(type_a) || at::isFloatingType(type_b)) {
    auto ix_a = dtype2index[static_cast<int64_t>(type_a)];
    TORCH_INTERNAL_ASSERT(ix_a != -1);
    auto ix_b = dtype2index[static_cast<int64_t>(type_b)];
    TORCH_INTERNAL_ASSERT(ix_b != -1);
    // This table axes must be consistent with index2dtype
    // clang-format off
    static constexpr
    std::array<std::array<c10::ScalarType, index2dtype.size()>, index2dtype.size()>
        _lowerTypesLookup = {{
        /*        f2  f4  f8  c2  c4  c8  bf*/
        /* f2 */ {f2, f2, f2, f2, f2, f2, bf},
        /* f4 */ {f2, f4, f4, c2, f4, f4, bf},
        /* f8 */ {f2, f4, f8, c2, c4, f8, bf},
        /* c2 */ {f2, c2, c2, c2, c2, c2, bf},
        /* c4 */ {f2, f4, c4, c2, c4, c4, bf},
        /* c8 */ {f2, f4, f8, c2, c4, c8, bf},
        /* bf */ {bf, bf, bf, bf, bf, bf, bf},
    }};
    // clang-format on
    return _lowerTypesLookup[ix_a][ix_b];
  }
  return type_a;
}

OpCheckResult result_check_func(const at::Tensor& out1, const at::Tensor& out2,
                                const std::string& op_name) {
  if (!out1.defined() && !out2.defined()) {
    return {true, op_name + " acc check pass.\n"};
  } else if (!out1.defined() || !out2.defined()) {
    return {false, op_name +
                       " acc check failed. \nOne of cpu or gcu contains "
                       "undefined tensor\n"};
  }
  auto gcu_out = out1.detach().cpu();
  auto cpu_out = out2.detach().cpu();
  at::ScalarType dtype = gcu_out.scalar_type();
  if (gcu_out.scalar_type() != cpu_out.scalar_type()) {
    TORCH_WARN("Output dtype of cpu and gcu is different, op: ", op_name,
               "cpu type: ", cpu_out.scalar_type(),
               " gcu type: ", gcu_out.scalar_type(),
               " convert cpu dtype to gcu");
    dtype = get_lower_scalar_type(gcu_out.scalar_type(), cpu_out.scalar_type());
    cpu_out = cpu_out.to(gcu_out.scalar_type());
  }
  at::Tensor mask;
  auto& cfg = OpDebugConfig::GetInstance();
  switch (dtype) {
    case at::ScalarType::Double:
    case at::ScalarType::Float: {
      double rtol = cfg.getFP32Rtol();
      double atol = cfg.getFP32Atol();
      mask = gcu_out.isclose(cpu_out, rtol, atol, /*equal_nan=*/true);
      break;
    }
    case at::ScalarType::Half: {
      double rtol = cfg.getFP16Rtol();
      double atol = cfg.getFP16Atol();
      mask = gcu_out.isclose(cpu_out, rtol, atol, /*equal_nan=*/true);
      break;
    }
    case at::ScalarType::BFloat16: {
      double rtol = cfg.getBF16Rtol();
      double atol = cfg.getBF16Atol();
      mask = gcu_out.isclose(cpu_out, rtol, atol, /*equal_nan=*/true);
      break;
    }
    case at::ScalarType::Bool:
    case at::ScalarType::Byte:
    case at::ScalarType::Char:
    case at::ScalarType::Int:
    case at::ScalarType::Long:
    case at::ScalarType::Short:
      mask = gcu_out.eq(cpu_out);
      break;
    default:
      PTDLOG(TORCH_GCU) << "Unsupport dtype (" << c10::toString(dtype)
                        << ") for op check use fp32 check mode";
      double rtol = cfg.getFP32Rtol();
      double atol = cfg.getFP32Atol();
      mask = gcu_out.isclose(cpu_out, rtol, atol, false);
      break;
  }
  bool acc_pass = mask.all().item().to<uint8_t>();
  std::stringstream ss;
  if (acc_pass) {
    ss << op_name << " acc check pass.\n";
  } else {
    mask = mask.eq(0);
    const int w = 25;
    auto diff_out1 = gcu_out.masked_select(mask).slice(0, 0, 100);
    auto diff_out2 = cpu_out.masked_select(mask).slice(0, 0, 100);
    // clang-format off
    ss << op_name << " acc check failed.\n"
       << "with acc info: {"
       << "\n"
       << std::setw(w) << std::left << out1.device().str() + "(A): "
       << std::setw(w) << std::left << out2.device().str() + "(B): "
       << std::setw(w) << std::left << "|A - B|"
       << std::setw(w) << std::left << "|A - B|/|B|"
       << "\n"
       << std::setw(w) << std::left
       << std::string(c10::toString(out1.scalar_type())) + "(" + std::to_string(diff_out1.numel()) + ")"
       << std::setw(w) << std::left
       << std::string(c10::toString(out2.scalar_type())) + "(" + std::to_string(diff_out2.numel()) + ")";

    // clang-format on
    if (diff_out1.scalar_type() == c10::ScalarType::Bool) {
      for (size_t i = 0; i < (size_t)diff_out1.numel(); ++i) {
        auto abs_ab = (diff_out1[i] == diff_out2[i]).item();
        ss << "\n"
           << std::setw(w) << std::left << diff_out1[i].item() << std::setw(w)
           << std::left << diff_out2[i].item() << std::setw(w) << std::left
           << abs_ab << std::setw(w) << std::left << abs_ab;
      }
    } else {
      for (size_t i = 0; i < (size_t)diff_out1.numel(); ++i) {
        auto abs_ab = at::native::abs(diff_out1[i] - diff_out2[i]).item();
        ss << "\n"
           << std::setw(w) << std::left << diff_out1[i].item() << std::setw(w)
           << std::left << diff_out2[i].item() << std::setw(w) << std::left
           << abs_ab << std::setw(w) << std::left
           << std::to_string(100 * abs_ab.toDouble() /
                             std::abs(diff_out2[i].item().toDouble())) +
                  "%";
      }
    }
    ss << "\n}\n";
  }
  return {acc_pass, ss.str()};
}

OpCheckResult result_check_func(const int64_t v1, const int64_t v2,
                                const std::string& op_name) {
  bool acc_pass = true;
  std::stringstream ss;
  if (v1 != v2) {
    // clang-format off
    ss << op_name << " acc check failed.\n"
       << "with acc info: {\n"
       << "result is " << v1 << " but expect " << v2 << "\n"
       << "}\n";
    // clang-format on
  } else {
    ss << op_name << " acc check pass.\n";
  }
  return {acc_pass, ss.str()};
}

OpCheckResult result_check_func(const c10::SymInt& v1, const c10::SymInt& v2,
                                const std::string& op_name) {
  return result_check_func(v1.expect_int(), v2.expect_int(), op_name);
}

void gcu_opcheck_run(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const c10::Device& xdevice = c10::Device(c10::DeviceType::CPU);
  auto& schema_args = op.schema().arguments();
  const auto num_arguments = schema_args.size();

  // Call the underlying xdevice implementation of the operator
  PTDLOG(OP) << op.schema() << " run acc check with " << xdevice;
  if (xdevice.is_cpu()) {
    torch::jit::Stack stack_backup;
    for (size_t i = 0; i < num_arguments; ++i) {
      stack_backup.push_back(stack->at(i));
    }

    try {
      op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);
    } catch (const c10::Error& error) {
      TORCH_CHECK(
          std::string(error.what_without_backtrace())
                  .find("not implemented for 'Half'") != std::string::npos,
          std::string(error.what_without_backtrace()));
      TORCH_WARN("cpu_check: ", c10::toString(op.operator_name()),
                 " not support Half, convert Half tensor to Float.");

      xdevice_cpu_fallback_with_convert(op, stack_backup, stack, true);
    }
  } else {
    PTCHECK(0) << "gcu_opcheck_run not support with " << xdevice;
  }
}

}  // namespace torch_gcu
