import os

os.environ["ENFLAME_LOG_DEBUG_MOD"] = "TORCH_GCU/OP,FLAME"

import torch
import torch_gcu

from torch_gcu import transfer_to_gcu

import unittest

import array
import numpy as np


class BaseAPITest(unittest.TestCase):

    def setUp(self):
        super().setUp()

    def _check_value(self, expect, got):
        self.assertTrue(torch.allclose(expect, got.cpu()),
                        f"result mismatch !!! expect {expect} but got {got}")

    def _check_shape_device(self, expect, got):
        self.assertTrue(
            got.is_gcu == True,
            f"result device mismatch !!! expect gcu but got {got.device}")
        self.assertTrue(
            expect.shape == got.shape,
            f"result shape mismatch !!! expect {expect.shape} but got {got.shape}"
        )

    def _torch_fn_test_base(self, fn, check_func, *args, **kwargs):
        cuda_out1 = fn(*args, **kwargs, device="cuda")
        cuda_out2 = fn(*args, **kwargs, device=0)
        cuda_out3 = fn(*args, **kwargs, device="cuda:0")
        cuda_out4 = fn(*args, **kwargs, device=torch.device("cuda"))
        cuda_out5 = fn(*args, **kwargs, device=torch.device(0))
        cuda_out6 = fn(*args, **kwargs, device=torch.device("cuda:0"))

        cpu_out = fn(*args, **kwargs, device="cpu")

        check_func(cpu_out, cuda_out1)
        check_func(cpu_out, cuda_out2)
        check_func(cpu_out, cuda_out3)
        check_func(cpu_out, cuda_out4)
        check_func(cpu_out, cuda_out5)
        check_func(cpu_out, cuda_out5)
        check_func(cpu_out, cuda_out6)

    def _torch_fn_test_check_value(self, fn, *args, **kwargs):
        self._torch_fn_test_base(fn, self._check_value, *args, **kwargs)

    def _torch_fn_test_check_shape_device(self, fn, *args, **kwargs):
        self._torch_fn_test_base(fn, self._check_shape_device, *args, **kwargs)

    def test_torch_cuda_fn(self):

        # get_device_properties
        print("get_device_properties: ",
              torch.cuda.get_device_properties("cuda"))
        # get_device_name
        self.assertEqual(torch.cuda.get_device_name(), "GCU")
        # get_device_capability
        print("get_device_capability: ", torch.cuda.get_device_capability())

        # TODO: support torch.cuda.list_gpu_processes
        # list_gpu_processes
        # print("list_gpu_processes: ", torch.cuda.list_gpu_processes())

        # set_device
        torch.cuda.set_device(0)

        # synchronize
        torch.cuda.synchronize()

        # mem_get_info
        print("mem_get_info: ", torch.cuda.mem_get_info())

        # memory_stats
        print("memory_stats: ", torch.cuda.memory_stats())

        # memory_summary
        print("memory_summary: ", torch.cuda.memory_summary())

        # memory_allocated
        print("memory_allocated: ", torch.cuda.memory_allocated())

        # max_memory_allocated
        print("max_memory_allocated: ", torch.cuda.max_memory_allocated())

        # reset_max_memory_allocated
        torch.cuda.reset_max_memory_allocated()

        # memory_reserved
        print("memory_reserved: ", torch.cuda.memory_reserved())

        # max_memory_reserved
        print("max_memory_reserved: ", torch.cuda.max_memory_reserved())

        # reset_max_memory_cached
        torch.cuda.reset_max_memory_cached()

        # torch.cuda.is_bf16_supported
        self.assertEqual(torch.cuda.is_bf16_supported(), True)

        # StreamContext
        self.assertEqual(torch.cuda.StreamContext, torch.gcu.StreamContext)

        # init
        self.assertEqual(torch.cuda.init, torch.gcu.init)
        self.assertEqual(torch.cuda._lazy_call, torch.gcu._lazy_call)
        self.assertEqual(torch.cuda._lazy_init, torch.gcu._lazy_init)

        # _tls
        self.assertEqual(torch.cuda._tls, torch.gcu._tls)

        # _initialization_lock
        self.assertEqual(torch.cuda._initialization_lock,
                         torch.gcu._initialization_lock)

        # _queued_calls
        self.assertEqual(torch.cuda._queued_calls, torch.gcu._queued_calls)

        # _is_in_bad_fork
        self.assertEqual(torch.cuda._is_in_bad_fork, torch.gcu._is_in_bad_fork)

        # _LazySeedTracker
        self.assertEqual(torch.cuda._LazySeedTracker,
                         torch.gcu._LazySeedTracker)

        # _lazy_seed_tracker
        self.assertEqual(torch.cuda._lazy_seed_tracker,
                         torch.gcu._lazy_seed_tracker)

        # reset_peak_memory_stats
        torch.cuda.reset_peak_memory_stats()

        # device
        self.assertEqual(torch.cuda.device, torch.gcu.device)

        # CUDAPluggableAllocator
        self.assertEqual(torch.cuda.memory.CUDAPluggableAllocator,
                         torch.gcu.memory.GCUPluggableAllocator)

    def test_tensor_type(self):

        def run_test(fn, dtype):
            label = torch.tensor([[1, 2, 3], [4, 5, 6]],
                                 dtype=dtype,
                                 device="cuda")
            val = fn([[1, 2, 3], [4, 5, 6]])
            self.assertTrue(torch.allclose(val.cpu(), label.cpu()))
            self.assertTrue(val.dtype == label.dtype)
            self.assertTrue(val.device == label.device)

        fn_list = [
            torch.cuda.DoubleTensor, torch.cuda.FloatTensor,
            torch.cuda.HalfTensor, torch.cuda.BFloat16Tensor,
            torch.cuda.LongTensor, torch.cuda.IntTensor,
            torch.cuda.ShortTensor, torch.cuda.BoolTensor,
            torch.cuda.CharTensor, torch.cuda.ByteTensor
        ]

        dtype_list = [
            torch.double, torch.float, torch.half, torch.bfloat16, torch.long,
            torch.int, torch.short, torch.bool, torch.int8, torch.uint8
        ]

        for fn, dtype in zip(fn_list, dtype_list):
            run_test(fn, dtype)

        def test_double():
            DOUBLE_MAX = 1.7976931348623158E+308  # 64 bit floating point max value
            DOUBLE_MIN = 2.2250738585072014E-308  # 64 bit floating point max value
            FLOAT_MAX = 3.402823466E+38  # 32 bit floating point max value
            FLOAT_MIN = 1.175494351E-38  # 32 bit floating point max value
            data = [[FLOAT_MIN, 0, FLOAT_MAX], [DOUBLE_MIN, 10.0, DOUBLE_MAX]]
            label = torch.tensor(data, dtype=torch.double, device="cuda")
            val = torch.cuda.DoubleTensor(data)
            self.assertTrue(torch.allclose(val.cpu(), label.cpu()))
            self.assertTrue(val.dtype == label.dtype)
            self.assertTrue(val.device == label.device)

        test_double()

    def test_cuda_random(self):
        torch.cuda.manual_seed(10)
        torch.cuda.manual_seed_all(41)
        torch.cuda.seed()
        torch.cuda.seed_all()
        torch.cuda.initial_seed()
        torch.cuda.get_rng_state()
        torch.cuda.get_rng_state()
        torch.cuda.get_rng_state_all()
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        torch.cuda.set_rng_state_all(torch.cuda.get_rng_state_all())
        self.assertEqual(torch.cuda.default_generators,
                         torch.gcu.default_generators)
        self.assertTrue(len(torch.cuda.default_generators) > 0)

    def test_cuda_nvtx(self):
        # Just making sure we can see the symbols
        torch.cuda.nvtx.range_push("foo")
        torch.cuda.nvtx.mark("bar")
        torch.cuda.nvtx.range_pop()
        range_handle = torch.cuda.nvtx.range_start("range_start")
        torch.cuda.nvtx.range_end(range_handle)
        torch.cuda.nvtx.range("foo_bar")

    def test_memory_snapshot(self):
        self.assertEqual(torch.cuda.memory_snapshot, torch.gcu.memory_snapshot)
        self.assertEqual(torch.cuda.memory._record_memory_history,
                         torch.gcu.memory._record_memory_history)
        self.assertEqual(torch.cuda.memory._record_memory_history_legacy,
                         torch.gcu.memory._record_memory_history_legacy)
        self.assertEqual(torch.cuda.memory._record_memory_history_impl,
                         torch.gcu.memory._record_memory_history_impl)
        self.assertEqual(torch.cuda.memory._snapshot,
                         torch.gcu.memory._snapshot)
        self.assertEqual(torch.cuda.memory._dump_snapshot,
                         torch.gcu.memory._dump_snapshot)

    def test_graph(self):
        self.assertEqual(torch.cuda.is_current_stream_capturing,
                         torch.gcu.is_current_stream_capturing)
        self.assertEqual(torch.cuda.graph_pool_handle,
                         torch.gcu.graph_pool_handle)
        self.assertEqual(torch.cuda.CUDAGraph, torch.gcu.GCUGraph)
        self.assertEqual(torch.cuda.graph, torch.gcu.graph)
        self.assertEqual(torch.cuda.make_graphed_callables,
                         torch.gcu.make_graphed_callables)

        # Ensure torch._C graph APIs are patched for GCU even if they don't follow
        # the "cuda" -> "gcu" naming convention (e.g. `_graph_pool_handle`).
        self.assertTrue(hasattr(torch._C, "_graph_pool_handle"))
        self.assertTrue(hasattr(torch_gcu._C, "_graph_pool_handle"))
        self.assertEqual(torch._C._graph_pool_handle,
                         torch_gcu._C._graph_pool_handle)

class DeviceTest(unittest.TestCase):

    def setUp(self):
        super().setUp()

    def test_device(self):
        self.assertEqual(torch.device("cuda").type, "gcu")

    def test_with_device(self):
        with torch.device("cuda"):
            data = torch.randn(3, 3)
            self.assertTrue(data.is_gcu)


class TunableTest(unittest.TestCase):

    def setUp(self):
        super().setUp()

    def test_api_replacement(self):
        self.assertEqual(torch.cuda.tunable.enable, torch.gcu.tunable.enable)
        self.assertEqual(torch.cuda.tunable.is_enabled,
                         torch.gcu.tunable.is_enabled)
        self.assertEqual(torch.cuda.tunable.tuning_enable,
                         torch.gcu.tunable.tuning_enable)
        self.assertEqual(torch.cuda.tunable.tuning_is_enabled,
                         torch.gcu.tunable.tuning_is_enabled)
        self.assertEqual(torch.cuda.tunable.set_max_tuning_duration,
                         torch.gcu.tunable.set_max_tuning_duration)
        self.assertEqual(torch.cuda.tunable.get_max_tuning_duration,
                         torch.gcu.tunable.get_max_tuning_duration)
        self.assertEqual(torch.cuda.tunable.set_max_tuning_iterations,
                         torch.gcu.tunable.set_max_tuning_iterations)
        self.assertEqual(torch.cuda.tunable.get_max_tuning_iterations,
                         torch.gcu.tunable.get_max_tuning_iterations)
        self.assertEqual(torch.cuda.tunable.set_filename,
                         torch.gcu.tunable.set_filename)
        self.assertEqual(torch.cuda.tunable.get_filename,
                         torch.gcu.tunable.get_filename)
        self.assertEqual(torch.cuda.tunable.get_results,
                         torch.gcu.tunable.get_results)
        self.assertEqual(torch.cuda.tunable.get_validators,
                         torch.gcu.tunable.get_validators)
        self.assertEqual(torch.cuda.tunable.write_file_on_exit,
                         torch.gcu.tunable.write_file_on_exit)
        self.assertEqual(torch.cuda.tunable.write_file,
                         torch.gcu.tunable.write_file)
        self.assertEqual(torch.cuda.tunable.read_file,
                         torch.gcu.tunable.read_file)

    def test_api(self):
        self.assertFalse(torch.cuda.tunable.is_enabled())
        self.assertFalse(torch.cuda.tunable.tuning_is_enabled())
        self.assertEqual(torch.cuda.tunable.get_max_tuning_duration(), 0)
        self.assertEqual(torch.cuda.tunable.get_max_tuning_iterations(), 0)
        self.assertEqual(torch.cuda.tunable.get_filename(), "")
        self.assertEqual(torch.cuda.tunable.get_results(), ("", "", "", 0.0))
        self.assertEqual(torch.cuda.tunable.get_validators(), ("", ""))

    def test_efml(self):
        torch.cuda.device_memory_used()


class ProfilerTest(unittest.TestCase):

    def setUp(self):
        super().setUp()

    def test_api_replacement(self):
        self.assertEqual(torch.profiler, torch_gcu.gcu.profiler)
        self.assertEqual(torch.profiler.ProfilerActivity.CUDA,
                         torch.profiler.ProfilerActivity.GCU)
        self.assertEqual(torch.profiler.ProfilerActivity.PrivateUse1,
                         torch.profiler.ProfilerActivity.GCU)


if __name__ == "__main__":
    unittest.main()
