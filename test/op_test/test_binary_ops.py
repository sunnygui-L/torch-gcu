#!/usr/bin/env python
#
# Copyright 2023-2025 Enflame. All Rights Reserved.
#
import os
os.environ["ENFLAME_LOG_DEBUG_MOD"] = "TORCH_GCU/OP,FLAME"

import torch
import torch._prims_common as utils
import unittest

from gcu_op_test import *

dev = "gcu:0"


class BinaryOpsTest(GcuOpTest):

    def setUp(self) -> None:
        torch.manual_seed(1 << 11)

    def test_binary_add_sub(self):
        vec_params = [
            [((3, 8), torch.float32), 20.0, 0.8, ((0, ), torch.float32)],
            [10, ((3, 6), torch.float16), 0.8, ((0, ), torch.float16)],
            # [((3, 8), torch.float32), ((3, 8), torch.float32), 0.1],
            # [((1, 8), torch.float16), ((3, 1), torch.float16), 0.6],
        ]
        for idx, params in enumerate(vec_params):
            nps = len(params)
            assert nps == 3 or nps == 4, "unsupported  params {}".format(params)
            op_params = {
                "lhs": params[0],
                "rhs": params[1],
                "alpha": params[2]
            }
            if nps == 4:
                op_params["out"] = params[-1]
            self.binary_run(torch.ops.aten.add, op_params, dev)
            self.binary_run(torch.ops.aten.sub, op_params, dev)
            if idx < 1:
                self.binary_run(torch.ops.aten.rsub, op_params, dev)

    def test_binary_mul_div(self):
        vec_params = [
            [((3, 8), torch.float32), 20.0],
            [10, ((3, 6), torch.float16)],
            # [((3, 8), torch.float32), ((3, 8), torch.float32)],
            # [((1, 8), torch.float16), ((3, 1), torch.float16)],
        ]
        for _, params in enumerate(vec_params):
            nps = len(params)
            assert nps == 2 or nps == 3, "unsupported  params {}".format(params)
            op_params = {"lhs": params[0], "rhs": params[1]}
            if nps == 3:
                op_params["out"] = params[-1]
            self.binary_run(torch.ops.aten.mul, op_params, dev)
            self.binary_run(torch.ops.aten.div, op_params, dev, equal_nan=True)

    def test_binary_comparison(self):
        vec_params = [
            [((1024, 8, 12), torch.float32), 1],
            [((1024, 8, 12), torch.float32),
             float('inf')],
            # [((1024, 8), torch.float32), ((1, 8), torch.float32)],
            # [((1024, 8, 12), torch.int16), 1],
        ]
        for _, params in enumerate(vec_params):
            nps = len(params)
            assert nps == 2 or nps == 3, "unsupported  params {}".format(params)
            op_params = {"lhs": params[0], "rhs": params[1]}
            if nps == 3:
                op_params["out"] = params[-1]
            self.binary_run(torch.ops.aten.eq, op_params, dev)
            self.binary_run(torch.ops.aten.ne, op_params, dev)
            self.binary_run(torch.ops.aten.gt, op_params, dev)
            self.binary_run(torch.ops.aten.ge, op_params, dev)
            self.binary_run(torch.ops.aten.lt, op_params, dev)
            self.binary_run(torch.ops.aten.le, op_params, dev)

    def test_binary_bitwise(self):
        vec_params = [
            [((1024, 8, 12), torch.int32), 1],
            [10, ((1024, 8, 12), torch.int16), ((1024, 8, 12), torch.int16)],
        ]
        for _, params in enumerate(vec_params):
            nps = len(params)
            assert nps == 2 or nps == 3, "unsupported  params {}".format(params)
            op_params = {"lhs": params[0], "rhs": params[1]}
            if nps == 3:
                op_params["out"] = params[-1]
            self.binary_run(torch.ops.aten.bitwise_and, op_params, dev)

    def test_binary_bitwise_or(self):
        vec_params = [
            [((1024, 8, 12), torch.int32), 1],
            [10, ((1024, 8, 12), torch.int16), ((1024, 8, 12), torch.int16)],
        ]
        for _, params in enumerate(vec_params):
            nps = len(params)
            assert nps == 2 or nps == 3, "unsupported  params {}".format(params)
            op_params = {"lhs": params[0], "rhs": params[1]}
            if nps == 3:
                op_params["out"] = params[-1]
            self.binary_run(torch.ops.aten.bitwise_or, op_params, dev)

    def test_binary_pow(self):
        vec_params = [
            [((1024, 8, 12), torch.float), 2.0],
            [3.0, ((8, 12), torch.float16), ((0, ), torch.float16)],
        ]
        for _, params in enumerate(vec_params):
            nps = len(params)
            assert nps == 2 or nps == 3, "unsupported  params {}".format(params)
            op_params = {"lhs": params[0], "rhs": params[1]}
            if nps == 3:
                op_params["out"] = params[-1]
            self.binary_run(torch.ops.aten.pow, op_params, dev)

    def test_binary_copysign(self):
        vec_params = [
            [((4, 4), torch.float), ((4, ), torch.float)],
            [((4, 4), torch.float), 1.0],
        ]
        for _, params in enumerate(vec_params):
            nps = len(params)
            assert nps == 2 or nps == 3, "unsupported  params {}".format(params)
            op_params = {"lhs": params[0], "rhs": params[1]}
            if nps == 3:
                op_params["out"] = params[-1]
            self.binary_run(torch.ops.aten.copysign, op_params, dev)

    def test_binary_div_mode(self):
        vec_params = [
            [((3, 8), torch.float32), 20.0],
            [10, ((3, 6), torch.float16)],
        ]
        for _, params in enumerate(vec_params):
            nps = len(params)
            assert nps == 2 or nps == 3, "unsupported  params {}".format(params)
            op_params = {
                "lhs": params[0],
                "rhs": params[1],
                "rounding_mode": "floor"
            }
            if nps == 4:
                op_params["out"] = params[-1]
            self.binary_run(torch.ops.aten.div, op_params, dev, equal_nan=True)

    def test_binary_floor_divide(self):
        vec_params = [[((3, 8), torch.float32), 20.0],
                      [1.0, ((3, 6), torch.float16)],
                      [0.5, ((8, 12), torch.float16), ((0, ), torch.float16)]]

        for _, params in enumerate(vec_params):
            nps = len(params)
            assert nps == 2 or nps == 3, "unsupported  params {}".format(params)
            op_params = {"lhs": params[0], "rhs": params[1]}
            if nps == 3:
                op_params["out"] = params[-1]
            self.binary_run(torch.ops.aten.floor_divide,
                            op_params,
                            dev,
                            equal_nan=True)

        vec_params_inplace = [[((3, 8), torch.float32), 20.0],
                              [((3, 8), torch.float16),
                               ((8, ), torch.float16)],
                              [((8, 12), torch.float16), 3.0]]

        torch.set_printoptions(precision=8)
        for _, params in enumerate(vec_params_inplace):
            nps = len(params)
            assert nps == 2, "unsupported  params {}".format(params)
            op_params = {"lhs": params[0], "rhs": params[1]}
            self.binary_run(torch.ops.aten.floor_divide_,
                            op_params,
                            dev,
                            equal_nan=True)
        torch.set_printoptions(precision=4)

    def binary_run(
        self,
        op,
        params,
        dev,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    ):
        lhs_param = params["lhs"]
        rhs_param = params["rhs"]
        c_lhs_clone, c_rhs_clone = None, None
        if isinstance(lhs_param, (list, tuple)):
            c_lhs = self.tensor_create(*lhs_param)
            c_lhs_clone = c_lhs.detach().clone()
            d_lhs = c_lhs.detach().to(dev)
        else:
            c_lhs = d_lhs = c_lhs_clone = lhs_param
        if isinstance(rhs_param, (list, tuple)):
            c_rhs = self.tensor_create(*rhs_param)
            c_rhs_clone = c_rhs.detach().clone()
            d_rhs = c_rhs.detach().to(dev)
        else:
            c_rhs = d_rhs = c_rhs_clone = rhs_param
        cpu_params = {}
        gcu_params = {}
        for key, v in params.items():
            if key != "lhs" and key != "rhs":
                if isinstance(v, (list, tuple)):
                    tmp = self.tensor_create(*v)
                    cpu_params[key] = tmp
                    gcu_params[key] = tmp.detach().to(dev)
                else:
                    cpu_params[key] = v
                    gcu_params[key] = v
        c_y = op(c_lhs, c_rhs, **cpu_params)
        d_y = op(d_lhs, d_rhs, **gcu_params)
        c_y_cpu = c_y.detach().cpu()
        d_y_cpu = d_y.detach().cpu()
        acc_pass, mask = self.tensor_check(c_y_cpu, d_y_cpu, rtol, atol,
                                           equal_nan)

        outs = [c_y_cpu, d_y_cpu, c_lhs_clone, c_rhs_clone]

        if mask is not None:
            outs = self.tensor_mask_select(outs, mask)

        self.assertTrue(
            acc_pass,
            msg=
            "op_name {}, params {}, \ncpu {} vs \ngcu {}, \nwith inputs: \nlhs {}, \nrhs {}"
            .format(op.__name__, params, *outs),
        )

    def tensor_mask_select(self, tensors, mask, length=20):
        x1, x2, lhs, rhs = tensors
        mask_x1 = x1.masked_select(mask)[:length]
        mask_x2 = x2.masked_select(mask)[:length]
        mask_lhs, mask_rhs = lhs, rhs
        if isinstance(lhs, torch.Tensor) and lhs.numel() > 1:
            mask_lhs = lhs.expand_as(mask).masked_select(mask)[:length]
        if isinstance(rhs, torch.Tensor) and rhs.numel() > 1:
            mask_rhs = rhs.expand_as(mask).masked_select(mask)[:length]
        return mask_x1, mask_x2, mask_lhs, mask_rhs

    def tensor_check(self, x1, x2, rtol, atol, equal_nan):
        mask = torch.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)
        acc_pass = mask.all().item()
        if acc_pass:
            return acc_pass, None
        else:
            mask = mask == 0
            return acc_pass, mask

    def tensor_create(self, shape, dtype):
        if utils.is_integer_dtype(dtype):
            return torch.randint(-1024, 1024, shape).to(dtype)
        elif utils.is_float_dtype(dtype):
            return torch.randn(shape).to(dtype)
        elif utils.is_boolean_dtype(dtype):
            return torch.randn(shape) > 0
        else:
            raise NotImplementedError("does not support {}".format(dtype))

    def test_lshift_scalar(self):
        c_input = torch.randint(1, 10, (3, 4))
        d_input = c_input.gcu()

        c_out = c_input << 2
        d_out = d_input << 2
        d_out = d_out.cpu()

        self.assertTrue(
            torch.allclose(c_out, d_out, atol=1e-4),
            msg="run {} op test fail: value cpu {} vs gcu {}.".format(
                "lshift", c_out, d_out),
        )
        pass

    def test_lshift_tensor(self):
        c_input = torch.randint(1, 10, (3, 4))
        c_shift = torch.tensor(2)
        d_input = c_input.gcu()
        d_shift = c_shift.gcu()

        c_out = c_input << c_shift
        d_out = d_input << d_shift
        d_out = d_out.cpu()

        self.assertTrue(
            torch.allclose(c_out, d_out, atol=1e-4),
            msg="run {} op test fail: value cpu {} vs gcu {}.".format(
                "lshift", c_out, d_out),
        )
        pass

    def test_gcd(self):
        import numpy as np
        type_list = (torch.uint8, torch.int8, torch.int16, torch.int32,
                     torch.int64)
        device = 'gcu'
        for dtype in type_list:
            t1 = torch.tensor([0, 10, 0], dtype=dtype, device=device)
            t2 = torch.tensor([0, 0, 10], dtype=dtype, device=device)
            actual = torch.gcd(t1, t2).cpu()
            expected = torch.from_numpy(np.gcd([0, 10, 0],
                                               [0, 0, 10])).to(actual.dtype)
            self.assertTrue(
                torch.allclose(actual, expected, atol=1e-4),
                msg="run {} op test fail: value cpu {} vs gcu {}.".format(
                    "gcd", expected, actual),
            )

            if dtype == torch.uint8:
                # Test unsigned integers with potential sign issues (i.e., uint8 with value >= 128)
                a = torch.tensor([190, 210], device=device, dtype=dtype)
                b = torch.tensor([190, 220], device=device, dtype=dtype)
                actual = torch.gcd(a, b)
                expected = torch.tensor([190, 10], device=device, dtype=dtype)
                self.assertTrue(
                    torch.allclose(actual, expected, atol=1e-4),
                    msg="run {} op test fail: value cpu {} vs gcu {}.".format(
                        "gcd", expected, actual),
                )
            else:
                # Compares with NumPy
                a = torch.randint(-20,
                                  20, (1024, ),
                                  device=device,
                                  dtype=dtype)
                b = torch.randint(-20,
                                  20, (1024, ),
                                  device=device,
                                  dtype=dtype)
                actual = torch.gcd(a, b).cpu()
                expected = torch.from_numpy(
                    np.gcd(a.cpu().numpy(),
                           b.cpu().numpy())).to(actual.dtype)
                self.assertTrue(
                    torch.allclose(actual, expected, atol=1e-4),
                    msg="run {} op test fail: value cpu {} vs gcu {}.".format(
                        "gcd", expected, actual),
                )

    def test_atan2(self):
        op = torch.ops.aten.atan2
        shapes = [[[3], [3]], [[3, 4], [3, 4]], [[3, 4, 5], [3, 4, 5]],
                  [[3, 1], [1, 3]]]
        dtypes = [torch.float32, torch.half, torch.int32]
        for shape in shapes:
            for dtype in dtypes:
                lhs = random_tensor(shape[0], dtype)
                rhs = random_tensor(shape[1], dtype)
                self.run_op_test(op, lhs, rhs)

    def test_fmax(self):
        op = torch.ops.aten.fmax
        input = torch.rand(3, 4)
        other = torch.rand(3, 4)
        self.run_op_test(op, input, other)

    def test_fmin(self):
        op = torch.ops.aten.fmin
        input = torch.rand(3, 4)
        other = torch.rand(3, 4)
        self.run_op_test(op, input, other)


if __name__ == "__main__":
    unittest.main()
