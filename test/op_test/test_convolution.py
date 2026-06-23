#!/usr/bin/env python
#
# Copyright 2023-2025 Enflame. All Rights Reserved.
#
import os

os.environ["ENFLAME_LOG_DEBUG_MOD"] = "TORCH_GCU/OP,FLAME"

import unittest
from copy import deepcopy

import torch
import torch.nn as nn
import torch_gcu


class ConvTest(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        torch.manual_seed(11)

    def test_conv1d(self):
        module_cpu = nn.Conv1d(
            in_channels=2, out_channels=16, kernel_size=3, bias=False, padding=1
        ).to(memory_format=torch.channels_last)
        module_gcu = deepcopy(module_cpu).gcu()

        input_cpu = torch.randn(2, 2, 4)
        input_gcu = input_cpu.clone().detach().gcu()

        output_cpu = module_cpu(input_cpu)
        output_gcu = module_gcu(input_gcu)
        self.assertTrue(
            torch.allclose(output_cpu, output_gcu.cpu(), atol=1e-4),
            msg="test_conv1d fail: value cpu {} vs gcu {}.".format(
                output_cpu, output_gcu.cpu()
            ),
        )

    def test_conv1d_with_bias(self):
        fn = torch.nn.functional.conv1d

        input_cpu = torch.randn(230, 3, 135)
        input_gcu = input_cpu.clone().detach().gcu()

        weight_cpu = torch.randn(4, 3, 3)
        weight_gcu = weight_cpu.clone().detach().gcu()

        bias_cpu = torch.randn(4)
        bias_gcu = bias_cpu.clone().detach().gcu()

        output_cpu = fn(
            input_cpu, weight_cpu, bias_cpu, stride=1, padding=0, dilation=1, groups=1
        )
        output_gcu = fn(
            input_gcu, weight_gcu, bias_gcu, stride=1, padding=0, dilation=1, groups=1
        )
        self.assertTrue(
            torch.allclose(output_cpu, output_gcu.cpu(), atol=1e-4),
            msg="test_conv1d_with_bias fail: value cpu {} vs gcu {}.".format(
                output_cpu, output_gcu.cpu()
            ),
        )

    def test_conv2d(self):
        module_cpu = nn.Conv2d(
            in_channels=2, out_channels=16, kernel_size=3, bias=False, padding=1
        ).to(memory_format=torch.channels_last)
        module_gcu = deepcopy(module_cpu).gcu()

        input_cpu = torch.randn(3, 2, 4, 4).to(memory_format=torch.channels_last)
        input_gcu = input_cpu.clone().detach().gcu()

        output_cpu = module_cpu(input_cpu)
        output_gcu = module_gcu(input_gcu)
        self.assertTrue(
            torch.allclose(output_cpu, output_gcu.cpu(), atol=1e-4),
            msg="test_conv2d fail: value cpu {} vs gcu {}.".format(
                output_cpu, output_gcu.cpu()
            ),
        )

    def test_conv2d_input_contiguous(self):
        module_cpu = nn.Conv2d(
            in_channels=2, out_channels=16, kernel_size=3, bias=False, padding=1
        )
        module_gcu = deepcopy(module_cpu).gcu()

        input_cpu = torch.randn(3, 2, 4, 4).to(memory_format=torch.channels_last)
        input_gcu = input_cpu.clone().detach().gcu()

        output_cpu = module_cpu(input_cpu)
        output_gcu = module_gcu(input_gcu)
        self.assertTrue(
            torch.allclose(output_cpu, output_gcu.cpu(), atol=1e-4),
            msg="test_conv2d_input_contiguous fail: value cpu {} vs gcu {}.".format(
                output_cpu, output_gcu.cpu()
            ),
        )

    def test_conv2d_with_bias(self):
        fn = torch.nn.functional.conv2d

        input_cpu = torch.randn(2, 4, 240, 135).to(memory_format=torch.channels_last)
        input_gcu = input_cpu.clone().detach().gcu()

        weight_cpu = torch.randn(320, 4, 3, 3).to(memory_format=torch.channels_last)
        weight_gcu = weight_cpu.clone().detach().gcu()

        bias_cpu = torch.randn(320)
        bias_gcu = bias_cpu.clone().detach().gcu()

        output_cpu = fn(
            input_cpu, weight_cpu, bias_cpu, stride=1, padding=0, dilation=1, groups=1
        )
        output_gcu = fn(
            input_gcu, weight_gcu, bias_gcu, stride=1, padding=0, dilation=1, groups=1
        )
        self.assertTrue(
            torch.allclose(output_cpu, output_gcu.cpu(), atol=1e-4),
            msg="test_conv2d_with_bias fail: value cpu {} vs gcu {}.".format(
                output_cpu, output_gcu.cpu()
            ),
        )

    def test_conv2d_with_bias2(self):
        from torch.nn import functional as F

        input_cpu = torch.rand(1, 4, 65, 128)
        final_padding = [1, 1, 1, 0]
        input_cpu = F.pad(input_cpu, final_padding, mode="constant")
        input_gcu = input_cpu.clone().detach().gcu()

        weight_cpu = torch.rand(320, 4, 3, 3)
        weight_gcu = weight_cpu.clone().detach().gcu()

        bias_cpu = torch.rand(320)
        bias_gcu = bias_cpu.clone().detach().gcu()

        output_cpu = F.conv2d(
            input_cpu, weight_cpu, bias_cpu, padding="valid"
        )
        output_gcu = F.conv2d(
            input_gcu, weight_gcu, bias_gcu, padding="valid"
        )
        self.assertTrue(
            torch.allclose(output_cpu, output_gcu.cpu(), atol=1e-4),
            msg="test_conv2d_with_bias2 fail: value cpu {} vs gcu {}.".format(
                output_cpu, output_gcu.cpu()
            ),
        )


    def test_conv2d_with_bias2_deterministic_mode_setting(self):
        from torch.nn import functional as F
        torch.use_deterministic_algorithms(True, warn_only=False)

        input_cpu = torch.rand(1, 4, 65, 128)
        final_padding = [1, 1, 1, 0]
        input_cpu = F.pad(input_cpu, final_padding, mode="constant")
        input_gcu = input_cpu.clone().detach().gcu()

        weight_cpu = torch.rand(320, 4, 3, 3)
        weight_gcu = weight_cpu.clone().detach().gcu()

        bias_cpu = torch.rand(320)
        bias_gcu = bias_cpu.clone().detach().gcu()

        output_cpu = F.conv2d(
            input_cpu, weight_cpu, bias_cpu, padding="valid"
        )
        output_gcu = F.conv2d(
            input_gcu, weight_gcu, bias_gcu, padding="valid"
        )
        self.assertTrue(
            torch.allclose(output_cpu, output_gcu.cpu(), atol=1e-4),
            msg="test_conv2d_with_bias2_deterministic_mode_setting fail: value cpu {} vs gcu {}.".format(
                output_cpu, output_gcu.cpu()
            ),
        )
        torch.use_deterministic_algorithms(False, warn_only=False)

    def test_convolution_deterministic_mode_setting(self):
        """Test that deterministic mode synchronization is called correctly"""
        fn = torch.nn.functional.conv2d

        input_gcu = torch.randn(2, 4, 240, 135).gcu()
        weight_gcu = torch.randn(320, 4, 3, 3).gcu()
        bias_gcu = torch.randn(320).gcu()

        # Test with deterministic mode enabled
        torch.use_deterministic_algorithms(True, warn_only=False)
        output_gcu = fn(
            input_gcu, weight_gcu, bias_gcu, stride=1, padding=0, dilation=1, groups=1
        )
        torch.use_deterministic_algorithms(False, warn_only=False)

if __name__ == "__main__":
    unittest.main()
