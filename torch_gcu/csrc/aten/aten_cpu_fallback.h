/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <ATen/Functions.h>
#include <ATen/Operators.h>
#include <ATen/native/CPUFallback.h>

#include "gcu/gcu_macros.h"

namespace torch_gcu {

/**
 * Performs CPU fallback and statistics for a given operator.
 *
 * This function checks if statistics are enabled and if so, it uses the
 * `op_record_mutable` function to get the operator's output information from
 * the `op_record`. This avoids repetitive computation by reusing the stack
 * from `op_record_mutable`. If statistics are not enabled, it falls back to
 * the `gcu_cpu_fallback` function.
 *
 * @param op The operator handle.
 * @param stack The stack containing the input and output tensors.
 */
void gcu_cpu_fallback_and_statistics(const c10::OperatorHandle& op,
                                     torch::jit::Stack* stack);

/**
 * @brief Fallback function for executing an operator on CPU when it is not
 * supported in torch_gcu.
 *
 * This function is called when an operator is not supported in torch_gcu and
 * needs to be executed on CPU. It converts optional TensorList arguments to CPU
 * and then calls the corresponding CPU fallback function. After the CPU
 * fallback function is executed, the results are copied back to the original
 * device. If the operator throws a "not implemented for 'Half'" error, it
 * converts Half tensors to Float before executing the CPU fallback.
 *
 * @param op The operator handle for the unsupported operator.
 * @param stack The stack containing the input arguments and output results.
 */
void gcu_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

/**
 * @brief Fallback function to CPU when cpu op not support f16
 *
 * This function is called when the cpu fallback error because of f16 type
 * not support. It firstly convert f16 tensor to f32 and run cpu op.
 * After return the f32 result, this function convert the result back to the
 * dtype that f16 op return by use meta backend.
 *
 * @param op The operator handle for the unsupported operator.
 * @param stack_backup the backup for the input arguments, because of the stack
 * will change after call once op.
 * @param stack The stack containing the input arguments and output results.
 * @param is_cpu_cpu_fallback Whether xdevice is cpu.
 */
void xdevice_cpu_fallback_with_convert(const c10::OperatorHandle& op,
                                       const torch::jit::Stack& stack_backup,
                                       torch::jit::Stack* stack,
                                       bool is_cpu_cpu_fallback = false);
}  // namespace torch_gcu
