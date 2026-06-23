#!/usr/bin/env python
#
# Copyright 2023-2025 Enflame. All Rights Reserved.
#
__all__ = ["GcuOpTest", "unittest", "copy", "torch", "random_tensor"]
import torch

try:
    import torchvision
except Exception as e:
    print(e)

dev = None
if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    import torch_gcu
    dev = torch.device("gcu")

import torch._prims_common as utils
import unittest
import copy
from typing import List

def random_tensor(shape, dtype):
    if utils.is_integer_dtype(dtype):
        return torch.randint(-10, 10, shape).to(dtype)
    elif utils.is_float_dtype(dtype) or utils.is_complex_dtype(dtype):
        return torch.randn(shape).to(dtype)
    elif utils.is_boolean_dtype(dtype):
        return torch.randn(shape) > 0
    else:
        raise NotImplementedError("does not support {}".format(dtype))


class GcuOpTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atol = 1e-5

    def setUp(self) -> None:
        torch.manual_seed(1 << 30)
        return super().setUp()

    def set_atol(self, atol=1e-5):
        self.atol = atol

    def run_op_test(self, op, *args, **kwargs):
        # for inplace op, so we need copy
        copy_args = copy.deepcopy(args)
        copy_kwargs = copy.deepcopy(kwargs)
        # for inplace op with gcu
        kwargs_args_map = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                for i, arg in enumerate(args):
                    if id(arg) == id(value):
                        kwargs_args_map[key] = i

        cpu_out = op(*args, **kwargs)

        gcu_args = []
        for input in copy_args:
            if isinstance(input, torch.Tensor):
                gcu_args.append(input.to(dev))
            elif isinstance(input, List):
                if len(input) > 0 and isinstance(input[0], torch.Tensor):
                    gcu_tensors = []
                    for t in input:
                        gcu_tensors.append(t.to(dev))
                    gcu_args.append(gcu_tensors)
                else:
                    gcu_args.append(input)
            else:
                gcu_args.append(input)

        gcu_kwargs = {}
        for key, value in copy_kwargs.items():
            if isinstance(value, torch.Tensor):
                if key in kwargs_args_map.keys():
                    gcu_kwargs[key] = gcu_args[kwargs_args_map[key]]
                else:
                    gcu_kwargs[key] = value.to(dev)
            else:
                gcu_kwargs[key] = value

        gcu_out = op(*gcu_args, **gcu_kwargs)

        def var_info(t):
            info = ""
            if isinstance(t, torch.Tensor):
                info = f"Tensor:[{t.shape}, {t.dtype}, {t.device}]"
            elif isinstance(t, tuple):
                info = f"Tuple:[{', '.join([var_info(x) for x in t])}]"
            else:
                info = f"{type(t)}:{t}"
            return info

        def blue(msg):
            return f"\033[34m\033[1m{msg}\033[0m"

        def error_info(cpu_outs, gcu_outs):
            info = ""
            info += blue("args:\n")
            n = '\n'
            t = '\t\t\t'
            info += f"{n.join([var_info(arg) for arg in args])}\n"
            if len(kwargs) > 0:
                info += blue("kwargs:\n")
                info += f"{n.join(str(key)+t+var_info(value) for key, value in kwargs.items())}\n"
            info += blue("cpu_out:\n")
            info += f"{[c_out for c_out in cpu_outs] if isinstance(cpu_outs, tuple) else cpu_outs}\n"
            info += blue("gcu_out:\n")
            info += f"{[g_out for g_out in gcu_outs] if isinstance(gcu_outs, tuple) else gcu_outs}\n"
            return info

        def check_result(lhs, rhs):
            self.assertTrue(
                torch.allclose(lhs, rhs, atol=self.atol),
                msg=
                f"run {op.__name__} op test fail, error info\n{error_info(lhs, rhs)}."
            )

        # check result
        if isinstance(cpu_out, tuple) and isinstance(gcu_out, tuple):
            assert len(cpu_out) == len(
                gcu_out
            ), "run {} op test fail: cpu_out {} vs gcu_out {}.".format(
                op.__name__, len(cpu_out), len(gcu_out))
            for i in range(len(cpu_out)):
                check_result(cpu_out[i], gcu_out[i].cpu())
        else:
            check_result(cpu_out, gcu_out.cpu())
        print("run {} op test success.".format(op.__name__))
