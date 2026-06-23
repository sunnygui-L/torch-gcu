#!/usr/bin/env python
#
# Copyright 2023-2025 Enflame. All Rights Reserved.
#
import os
os.environ["ENFLAME_LOG_DEBUG_MOD"] = "TORCH_GCU/OP,FLAME"

import unittest
import torch
import torch.nn as nn
from gcu_op_test import *


class LinearTest(GcuOpTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        torch.manual_seed(11)

    def test_linear(self):
        fn = torch.ops.aten.linear

        input_cpu = torch.randn(2, 5)
        input_gcu = input_cpu.clone().detach().gcu()

        weight_cpu = torch.randn(3, 5)
        weight_gcu = weight_cpu.clone().detach().gcu()

        bias_cpu = torch.randn(3)
        bias_gcu = bias_cpu.clone().detach().gcu()

        output_cpu = fn(input_cpu, weight_cpu, bias_cpu)
        output_gcu = fn(input_gcu, weight_gcu, bias_gcu)

        self.assertTrue(torch.allclose(output_cpu, output_gcu.cpu()),
                        msg='test_linear_withbias fail: value cpu {} vs gcu {}.'.format(output_cpu, output_gcu.cpu()))

    def test_linear_nobias(self):
        fn = torch.ops.aten.linear

        input_cpu = torch.randn(2, 5)
        input_gcu = input_cpu.clone().detach().gcu()

        weight_cpu = torch.randn(3, 5)
        weight_gcu = weight_cpu.clone().detach().gcu()

        output_cpu = fn(input_cpu, weight_cpu)
        output_gcu = fn(input_gcu, weight_gcu)

        self.assertTrue(torch.allclose(output_cpu, output_gcu.cpu()),
                        msg='test_linear_nobias fail: value cpu {} vs gcu {}.'.format(output_cpu, output_gcu.cpu()))

    def test_addmm(self):
        fn = torch.ops.aten.addmm

        input_cpu = torch.randn(3)
        input_gcu = input_cpu.clone().detach().gcu()

        mat1_cpu = torch.randn(2, 5)
        mat1_gcu = mat1_cpu.clone().detach().gcu()

        mat2_cpu = torch.randn(5, 3)
        mat2_gcu = mat2_cpu.clone().detach().gcu()

        output_cpu = fn(input_cpu, mat1_cpu, mat2_cpu)
        output_gcu = fn(input_gcu, mat1_gcu, mat2_gcu)
        self.assertTrue(torch.allclose(output_cpu, output_gcu.cpu()),
                        msg='test_addmm fail: value cpu {} vs gcu {}.'.format(output_cpu, output_gcu.cpu()))

    def test_mm(self):
        fn = torch.ops.aten.mm

        mat1_cpu = torch.randn(2, 5)
        mat1_gcu = mat1_cpu.clone().detach().gcu()

        mat2_cpu = torch.randn(5, 3)
        mat2_gcu = mat2_cpu.clone().detach().gcu()

        output_cpu = fn(mat1_cpu, mat2_cpu)
        output_gcu = fn(mat1_gcu, mat2_gcu)
        self.assertTrue(torch.allclose(output_cpu, output_gcu.cpu()),
                        msg='test_mm fail: value cpu {} vs gcu {}.'.format(output_cpu, output_gcu.cpu()))

    def test_bmm(self):
        fn = torch.ops.aten.bmm

        mat1_cpu = torch.randn(4, 2, 5)
        mat1_gcu = mat1_cpu.clone().detach().gcu()

        mat2_cpu = torch.randn(4, 5, 3)
        mat2_gcu = mat2_cpu.clone().detach().gcu()

        output_cpu = fn(mat1_cpu, mat2_cpu)
        output_gcu = fn(mat1_gcu, mat2_gcu)
        self.assertTrue(torch.allclose(output_cpu, output_gcu.cpu()),
                        msg='test_bmm fail: value cpu {} vs gcu {}.'.format(output_cpu, output_gcu.cpu()))

    def test_matmul(self):
        fn = torch.ops.aten.matmul

        mat1_cpu = torch.randn(4, 2, 5)
        mat1_gcu = mat1_cpu.clone().detach().gcu()

        mat2_cpu = torch.randn(4, 5, 3)
        mat2_gcu = mat2_cpu.clone().detach().gcu()

        output_cpu = fn(mat1_cpu, mat2_cpu)
        output_gcu = fn(mat1_gcu, mat2_gcu)
        self.assertTrue(torch.allclose(output_cpu, output_gcu.cpu()),
                        msg='test_matmul fail: value cpu {} vs gcu {}.'.format(output_cpu, output_gcu.cpu()))

    def test_baddbmm(self):
        fn = torch.ops.aten.baddbmm

        in_cpu = torch.randn(4, 2, 5)
        in_gcu = in_cpu.clone().detach().gcu()

        batch1_cpu = torch.randn(4, 2, 6)
        batch1_gcu = batch1_cpu.clone().detach().gcu()

        batch2_cpu = torch.randn(4, 6, 5)
        batch2_gcu = batch2_cpu.clone().detach().gcu()

        beta = -0.3
        alpha = 0.3

        output_cpu = fn(in_cpu, batch1_cpu, batch2_cpu, beta=beta, alpha=alpha)
        output_gcu = fn(in_gcu, batch1_gcu, batch2_gcu, beta=beta, alpha=alpha)

        self.assertTrue(torch.allclose(output_cpu, output_gcu.cpu()),
                        msg='test_baddbmm fail: value cpu {} vs gcu {}.'.format(output_cpu, output_gcu.cpu()))

    def test_addmv(self):
        op = torch.ops.aten.addmv
        input_shapes = [[[2], [2, 3], [3]], [[], [5, 3], [3]]]
        for input_shape in input_shapes:
            input = torch.randn(input_shape[0])
            mat = torch.randn(input_shape[1])
            vec = torch.randn(input_shape[2])
            for beta in [0, 1, 2]:
                for alpha in [0, 1, 2]:
                    self.run_op_test(op,
                                     input,
                                     mat,
                                     vec,
                                     beta=beta,
                                     alpha=alpha)



if __name__ == '__main__':
    unittest.main()
