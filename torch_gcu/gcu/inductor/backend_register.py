import os
import torch
import functools
import logging

logger = logging.getLogger(__name__)
# mark monkey patch applied or not
_IS_INDUCTOR_PATCHED = False
_gcu_interface_cls = None
_gcu_device_op_overrides = None

def apply_gcu_inductor_monkey_patch():
    global _IS_INDUCTOR_PATCHED, _gcu_interface_cls, _gcu_device_op_overrides
    if _IS_INDUCTOR_PATCHED:
        return

    _IS_INDUCTOR_PATCHED = True

    import torch
    from torch._inductor.codegen.common import get_scheduling_for_device, register_backend_for_device
    from torch._inductor.triton_bundler import TritonBundler
    from torch.types import Device as _device_t
    from torch._dynamo.device_interface import DeviceInterface, register_interface_for_device
    from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
    from torch._inductor.codegen.common import (DeviceOpOverrides,
                                                register_device_op_overrides,
                                                InplacedBuffer, RemovedArg,
                                                ArgName, ConstexprArg)

    caching_worker_device_properties: Dict[str, Any] = {}
    caching_worker_current_devices: Dict[str, int] = {}

    from torch_gcu._C import _gcu_getCurrentRawStream as get_gcu_stream

    class GcuInterface(DeviceInterface):
        device = torch.gcu.device
        Event = torch.gcu.Event
        # Stream = torch.gcu.Stream

        class Worker:

            @staticmethod
            def set_device(device: int):
                caching_worker_current_devices["gcu"] = device

            @staticmethod
            def current_device() -> int:
                if "gcu" in caching_worker_current_devices:
                    return caching_worker_current_devices["gcu"]
                return torch.gcu.current_device()

            @staticmethod
            def get_device_properties(device: _device_t = None):
                if device is not None:
                    if isinstance(device, str):
                        device = torch.device(device)
                        assert device.type == "gcu"
                    if isinstance(device, torch.device):
                        device = device.index
                if device is None:
                    device = GcuInterface.Worker.current_device()

                if "gcu" not in caching_worker_device_properties:
                    device_prop = [
                        torch.gcu.get_device_properties(i)
                        for i in range(torch.gcu.device_count())
                    ]
                    caching_worker_device_properties["gcu"] = device_prop

                return caching_worker_device_properties["gcu"][device]

        current_device = staticmethod(torch.gcu.current_device)
        set_device = staticmethod(torch.gcu.set_device)
        device_count = staticmethod(torch.gcu.device_count)
        stream = staticmethod(torch.gcu.stream)  # type: ignore[assignment]
        current_stream = staticmethod(torch.gcu.current_stream)
        set_stream = staticmethod(torch.gcu.set_stream)  # type: ignore[assignment]
        _set_stream_by_id = staticmethod(
            torch.gcu._set_stream_by_id)  # type: ignore[assignment]
        synchronize = staticmethod(torch.gcu.synchronize)
        get_device_properties = staticmethod(
            torch.gcu.get_device_properties)  # type: ignore[assignment]
        get_raw_stream = staticmethod(
            get_gcu_stream)  # type: ignore[assignment, arg-type]
        exchange_device = staticmethod(
            torch.gcu._exchange_device)  # type: ignore[arg-type]
        maybe_exchange_device = staticmethod(
            torch.gcu._maybe_exchange_device)  # type: ignore[arg-type]

        # Can be mock patched by @patch decorator.
        @staticmethod
        def is_available() -> bool:
            return torch.gcu.is_available()

        @staticmethod
        def get_compute_capability(device: _device_t = None):
            cc = torch.gcu.get_device_capability(device)
            return cc

        @staticmethod
        def is_bf16_supported(including_emulation: bool = False) -> bool:
            return torch.gcu.is_bf16_supported()

    _gcu_interface_cls = GcuInterface

    class GCUDeviceOpOverrides(DeviceOpOverrides):

        def import_get_raw_stream_as(self, name):
            return f"from torch_gcu._C import _gcu_getCurrentRawStream as {name}"

        def set_device(self, device_idx):
            return f"torch.gcu.set_device({device_idx})"

        def synchronize(self):
            return "torch.gcu.synchronize()"

        def device_guard(self, device_idx):
            return f"torch.gcu._DeviceGuard({device_idx})"

    _gcu_device_op_overrides = GCUDeviceOpOverrides

    register_device_op_overrides("gcu", GCUDeviceOpOverrides())

    def gcu_get_device_op_overrides(device: str) -> DeviceOpOverrides:
        from torch._inductor.codegen.common import device_op_overrides_dict
        assert isinstance(device, str)

        # if not device_op_overrides_dict:
        from torch._inductor.codegen import cpu_device_op_overrides, mps_device_op_overrides  # noqa: F401
        from torch._inductor.codegen.cuda import device_op_overrides  # noqa: F401
        from torch._inductor.codegen.xpu import device_op_overrides as xpu_op_overrides  # noqa: F401

        return device_op_overrides_dict[device]

    from torch._inductor.codegen.wrapper import PythonWrapperCodegen

    def gcu_init_wrapper_code(
        self,
        is_subgraph: bool = False,
        subgraph_name: Optional[str] = None,
        parent_wrapper_code: Optional[PythonWrapperCodegen] = None,
    ) -> None:
        from torch._inductor.codegen.common import get_wrapper_codegen_for_device

        device_types = self.device_types.copy()
        device_types.discard("cpu")
        device_types.discard("meta")
        # TODO(Eikan): Only support mixing cpu and other device now.
        assert len(device_types) <= 1, "Does not support mixing {}".format(
            "+".join(device_types))
        only_cpu = len(device_types) == 0
        self.device_type = "cpu" if only_cpu else device_types.pop()

        if self.cpp_wrapper:
            self.validate_can_generate_cpp_wrapper()

        self.device_ops = gcu_get_device_op_overrides(self.device_type)
        wrapper_code_gen_cls = get_wrapper_codegen_for_device(
            self.device_type, self.cpp_wrapper)
        assert (wrapper_code_gen_cls
                is not None), f"Device {self.device_type} not supported"
        self.wrapper_code = wrapper_code_gen_cls.create(
            is_subgraph, subgraph_name, parent_wrapper_code)

        if self.const_module:
            # If we have const module, we could reuse the kernels
            # This could avoid duplication and save time on doing recompilation (if Triton.)
            self.wrapper_code._names_iter = self.const_module.wrapper_code._names_iter
            self.wrapper_code.src_to_kernel = (
                self.const_module.wrapper_code.src_to_kernel)

    if get_scheduling_for_device("gcu") is None:
        from torch._inductor.codegen.triton import TritonScheduling
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen

        register_backend_for_device("gcu", TritonScheduling,
                                    PythonWrapperCodegen)

    register_interface_for_device("gcu", GcuInterface)
    for i in range(torch.gcu.device_count()):
        register_interface_for_device(f"gcu:{i}", GcuInterface)

    from functools import lru_cache
    from torch._inductor.codegen.triton import gen_attr_descriptor_import
    from torch._inductor.utils import IndentedBuffer, is_pointwise_use

    try:
        import triton_gcu.triton
    except ImportError:
        pass

    @lru_cache(None)
    def gen_common_triton_imports():
        imports = IndentedBuffer()
        imports.splice("""
            import torch_gcu
            import triton
            import triton_gcu.triton
            import triton.language as tl
            """)
        if attr_desc := gen_attr_descriptor_import():
            imports.writeline(attr_desc)

        imports.splice("""
            from torch._inductor.runtime import triton_helpers, triton_heuristics
            from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
            from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, AttrsDescriptorWrapper, DeviceProperties
            """)
        return imports.getvalue()

    def is_gpu_or_gcu(device: str):
        assert isinstance(device, str) or device is None, device
        return device in ["cuda", "xpu", "gcu"]


    import torch._inductor.runtime.triton_heuristics

    def get_max_y_grid():
        return MAX_GRID_SIZE_GCU["Y"]

    torch._inductor.runtime.runtime_utils.get_max_y_grid = get_max_y_grid

    from torch._inductor.runtime.runtime_utils import get_max_y_grid

    def should_prefer_unfused_addmm(match):
        inp = match.kwargs["inp"]
        if not (inp.meta["val"].is_cuda or inp.meta["val"].is_gcu):
            return False

        output = match.output_node()
        return all(is_pointwise_use(use) for use in output.users)

    from torch._inductor.pattern_matcher import (
        Arg,
        CallFunction,
        KeywordArg,
        Match,
        register_graph_pattern,
    )

    aten = torch.ops.aten
    import torch._inductor.fx_passes.post_grad

    pass_patterns = torch._inductor.fx_passes.post_grad.pass_patterns

    @register_graph_pattern(
        CallFunction(aten.addmm, KeywordArg("inp"), Arg(), Arg()),
        pass_dict=pass_patterns[2],
        extra_check=should_prefer_unfused_addmm,
    )
    def unfuse_bias_add_to_pointwise(match: Match, mat1, mat2, *, inp):

        def repl(inp, x1, x2):
            return x1 @ x2 + inp

        match.replace_by_example(repl, [inp, mat1, mat2])

    from torch._inductor.ir import get_device_type

    def is_triton(x: object) -> bool:
        dtype = get_device_type(x)
        return bool(dtype and is_gpu_or_gcu(dtype))

    # Replace the torch._inductor.graph.GraphLowering.compile_to_module interface for
    # conversions that do not support the int64 data type
    from torch._inductor.codecache import output_code_log
    from torch._inductor.graph import GraphLowering
    from torch._logging import trace_structured

    def _compile_to_module(self):
        from torch._inductor.codecache import PyCodeCache

        wrapper_code, _ = (self.codegen_with_cpp_wrapper()
                           if self.cpp_wrapper else self.codegen())
        if config.triton.autotune_at_compile_time:
            tuning_code = ('"""\n' + "Compile-time auto-tuning block: \n" +
                           self.wrapper_code.kernel_autotune_defs.getvalue() +
                           self.wrapper_code.kernel_autotune_calls.getvalue() +
                           '"""\n')
            wrapper_code.value = tuning_code + wrapper_code.value
        if GraphLowering.save_output_code is not None:
            GraphLowering.save_output_code(wrapper_code.value)
        output_code_log.debug("Output code: \n%s", wrapper_code.value)

        from datetime import datetime
        cur_time = datetime.now()
        time_stamp = cur_time.strftime('%Y%m%d%H%M%S')

        wrapper_code.value = wrapper_code.value.replace("int64", "int32")
        wrapper_code.value = wrapper_code.value.replace("i64", "i32")

        from torch._inductor.runtime import autotune_cache
        from torch._inductor.runtime.autotune_cache import AutotuneCacheBundler

        inductor_meta = autotune_cache.inductor_meta_from_config()
        AutotuneCacheBundler.begin_compile(inductor_meta,
                                           code=wrapper_code.value)

        try:
            linemap = [
                (line_no, node.stack_trace)  # type: ignore[attr-defined]
                for line_no, node in wrapper_code.line_map
            ]
            key, path = PyCodeCache.write(wrapper_code.value)
            output_code_log.debug("Output code written to: %s", path)
        except Exception:
            trace_structured(
                "inductor_output_code",
                # Just omit the filename, I still want the code though!
                payload_fn=lambda: wrapper_code.value,
            )
            raise
        else:
            trace_structured(
                "inductor_output_code",
                lambda: {"filename": path},
                payload_fn=lambda: wrapper_code.value,
            )
        with dynamo_timed("PyCodeCache.load_by_key_path",
                          log_pt2_compile_event=True):
            mod = PyCodeCache.load_by_key_path(
                key,
                path,
                linemap=linemap,  # type: ignore[arg-type]
                attrs={
                    **self.constants,
                    **self.torchbind_constants
                },
            )
        self.cache_key = key
        self.cache_path = path
        self.cache_linemap = linemap  # type: ignore[assignment]

        if config.benchmark_harness and config.profile_bandwidth_output:
            # run the inputs code gen to get the bandwidth info
            mod.benchmark_compiled_module(times=1, repeat=1)
        # Logged twice as per https://github.com/pytorch/pytorch/pull/99038#discussion_r1167826029
        # TODO. Revisit this once the logging API is more mature
        assert mod.__file__ is not None

        import os

        # log_module_code(mod.__file__)
        # log.debug("Output code written to: %s", mod.__file__)
        output_code_log.info("Output code written to: %s", mod.__file__)
        if config.benchmark_kernel:
            print(f"Compiled module path: {mod.__file__}", file=sys.stderr)
        V.debug.output_code(mod.__file__)
        V.debug.copy(os.path.splitext(mod.__file__)[0] + ".debug")
        return mod

    def patched_compile_to_module(self):
        with dynamo_timed(
                "GraphLowering.compile_to_module",
                phase_name="code_gen",
                log_pt2_compile_event=True,
                dynamo_compile_column_us=
                "inductor_code_gen_cumulative_compile_time_us",
        ):
            return _compile_to_module(self)

    from functools import cached_property
    from typing import Any, Callable
    from typing_extensions import Self

    import math
    from torch._inductor.runtime.runtime_utils import (
        ceildiv,
        conditional_product,
        create_bandwidth_info_str,
        dynamo_timed,
        get_first_attr,
        get_max_y_grid,
        get_num_bytes,
        next_power_of_2,
        triton_cache_dir,
        triton_config_to_hashable,
        triton_hash_to_path_key,
        validate_triton_config,
    )

    try:
        import triton
    except ImportError:
        triton = None

    if triton is not None:
        from triton import Config
        from triton.compiler import CompiledKernel
        from triton.runtime.autotuner import OutOfResources
        from triton.runtime.jit import KernelInterface

        try:
            from triton.compiler.compiler import ASTSource
        except ImportError:
            ASTSource = None

        try:
            from triton.backends.compiler import GPUTarget
        except ImportError:
            GPUTarget = None
    else:
        Config = object
        KernelInterface = object
        OutOfResources = object
        ASTSource = None
        GPUTarget = None

    TRITON_MAX_BLOCK_GCU = {
        "X": 4096,
        "Y": 1024,
        "Z": 1024,
        "R0_": 4096 * 16,  # * 16 is multi-kernel only
        "R1_": 2048 * 16,
    }

    # TODO(GCU) update grid size according to different chip
    MAX_GRID_SIZE_GCU = {
        "X": 65535,
        "Y": 255,
        "Z": 255,
    }
    from torch._inductor.runtime.triton_heuristics import _num_warps, _check_max_grid_x, check_config
    from torch._inductor.runtime.hints import _NUM_THREADS_PER_WARP

    def triton_config_gcu(
        size_hints,
        x,
        y=None,
        z=None,
        num_stages=1,
        num_elements_per_warp=256,
        min_elem_per_thread=0,
    ) -> Config:
        """
        Construct a pointwise triton config with some adjustment heuristics
        based on size_hints. Size_hints is a tuple of numels in each tile
        dimension and will be rounded up to the nearest power of 2.

        num_elements_per_warp is a suggestion for controlling how many warps
        the triton config should contain. e.g.: if x=16, y=8, z=4 then
        num_elements = 16*8*4 = 512. Then if we set num_elements_per_warp=128,
        we'll launch 512 (elem) / 128 (elem/warp) = 4 warps. Note that it's
        just a suggestion, and sometimes other adjustment heuristics will
        override the num_elements_per_warp.

        min_elem_per_thread controls the minimum number of elements
        processed by each thread. It's always enforced.
        """
        # Ideally we want to read this from some device config

        # maxGridSize = [2147483647, 65535, 65535]
        # update maxGridSize to GCU setting by MAX_GRID_SIZE_GCU
        # maxGridSize = [65535, 255, 255]

        target = conditional_product(x, y, z)
        if conditional_product(*size_hints.values()) < target:
            target //= 8

        # shrink sizes to size hints
        x = min(x, size_hints["x"])
        if y:
            y = min(y, size_hints["y"])
        if z:
            z = min(z, size_hints["z"])

        # if we are below original block size, scale up where we can;
        # or if the calculated grid size is larger than the limit, we bump up the corresponding dimension
        while x < min(size_hints["x"], TRITON_MAX_BLOCK_GCU["X"]) and (
                x * MAX_GRID_SIZE_GCU["X"] < size_hints["x"]
                or conditional_product(x, y, z) < target):
            x *= 2
        while (y and y < min(size_hints["y"], TRITON_MAX_BLOCK_GCU["Y"])
               and (y * MAX_GRID_SIZE_GCU["Y"] < size_hints["y"]
                    or conditional_product(x, y, z) < target)):
            y *= 2
        while (z and z < min(size_hints["z"], TRITON_MAX_BLOCK_GCU["Z"])
               and (z * MAX_GRID_SIZE_GCU["Z"] < size_hints["z"]
                    or conditional_product(x, y, z) < target)):
            z *= 2

        num_warps = _num_warps(conditional_product(x, y, z) //
                               num_elements_per_warp,
                               max_num_warps=4,
                               min_num_warps=1)
        # we are going to arrive at 2 warps only if bs was too small due to
        # numel being too small. However to workaround some ptx bugs we still
        # want at least 4 warps if there's enough elements per thread
        # given that this is a rare situation, don't expect this to affect perf
        # in general
        # see https://github.com/pytorch/pytorch/pull/97950
        if conditional_product(x, y, z) >= 128 and not torch.version.hip:
            num_warps = max(num_warps, 4)
        xnumel = size_hints["x"]
        ynumel = size_hints["y"] if y else None
        znumel = size_hints["z"] if z else None

        # Increase x to satisfy min_elem_per_thread requirements.
        block_size = max(
            conditional_product(x, y, z),
            min_elem_per_thread * _NUM_THREADS_PER_WARP * num_warps,
        )
        x *= math.ceil(block_size / conditional_product(x, y, z))

        x, _num_blocks = _check_max_grid_x(size_hints, x, num_warps)

        cfg = {"XBLOCK": x}
        if y:
            cfg["YBLOCK"] = y
        if z:
            cfg["ZBLOCK"] = z
        assert x <= TRITON_MAX_BLOCK_GCU[
            "X"], f"increase TRITON_MAX_BLOCK['X'] to {x}"
        check_config(cfg, xnumel=xnumel, ynumel=ynumel, znumel=znumel)
        return Config(cfg, num_warps=num_warps, num_stages=num_stages)

    from torch._inductor.runtime.hints import ReductionHint, HeuristicType
    from torch._inductor.runtime.triton_heuristics import (
        disable_pointwise_autotuning, cached_autotune,
        _get_nd_reduction_numels, _get_config, check_max_block, GridExpr,
        config_to_dict)

    def triton_config_reduction_gcu(size_hints,
                                    x,
                                    r,
                                    num_stages=1,
                                    num_warps=None,
                                    register_intensive=False) -> Config:
        """
        Construct a reduction triton config with some adjustment heuristics
        based on size_hints. Size_hints is a tuple of numels in each tile
        dimension and will be rounded up to the nearest power of 2.
        """
        # Convert the linear reduction numel into a multi-dimensional block.
        rnumels = _get_nd_reduction_numels(r, size_hints)

        # shrink sizes to size hints
        x = min(x, size_hints["x"])

        def total_numel() -> int:
            return conditional_product(x, *rnumels.values())

        target = total_numel()
        if conditional_product(*size_hints.values()) < target:
            target //= 8

        # if we are below original block size, scale up where we can
        while x < size_hints["x"] and total_numel() < target:
            x *= 2
        for prefix in sorted(rnumels):
            while rnumels[prefix] < size_hints[prefix] and total_numel(
            ) < target:
                rnumels[prefix] *= 2

        if num_warps is None:
            num_warps = total_numel() // 128
        num_warps = _num_warps(num_warps,
                               max_num_warps=4,
                               register_intensive=register_intensive)

        x, _num_blocks = _check_max_grid_x(size_hints, x, num_warps)

        for prefix in sorted(rnumels):
            while total_numel() > target:
                if rnumels[prefix] == 1:
                    break
                rnumels[prefix] //= 2

        cfg = _get_config({"x": x, **rnumels})
        check_max_block(cfg)
        check_config(cfg, xnumel=size_hints["x"])
        assert x <= TRITON_MAX_BLOCK_GCU[
            "X"], f"increase TRITON_MAX_BLOCK['X'] to {x}"
        assert r <= TRITON_MAX_BLOCK_GCU[
            "R0_"], f"increase TRITON_MAX_BLOCK['r'] to {r}"
        return Config(cfg, num_warps=num_warps, num_stages=num_stages)

    from torch._inductor.runtime.triton_heuristics import get_total_reduction_numel

    def _reduction_configs_gcu(*, size_hints: dict[str, int],
                               inductor_meta: dict[str, Any]) -> list[Config]:
        reduction_hint = inductor_meta.get("reduction_hint", None)

        # Convert reductions to 1D, to simplify heuristics.
        rnumel = get_total_reduction_numel(size_hints)

        register_intensive = False
        MAX_R0_BLOCK = 2048
        if (size_hints["x"] >= 1024 and inductor_meta.get("num_load", 0) +
                inductor_meta.get("num_reduction", 0) >= 10):
            # A heuristics to reduce R0_BLOCK if a kernel potentially need many registers.
            # Consider load and reduction since load need move data into registers and
            # reduction needs an accumulator.
            #
            # The magic numbers are a bit arbitrary.
            #
            # We cannot rely on dynamically scaling down R0_BLOCK later, since sometimes
            # triton makes it to use less registers with worse perf. Check:
            # https://github.com/pytorch/pytorch/issues/126463
            #
            # The heuristic is a very simple one since registers can be reused. But
            # hopefully it can be a good enough indicator.
            MAX_R0_BLOCK = 1024
            register_intensive = True

        contiguous_config = triton_config_reduction_gcu(
            size_hints,
            1,
            (rnumel if 256 <= rnumel < MAX_R0_BLOCK else MAX_R0_BLOCK),
            register_intensive=register_intensive,
        )
        outer_config = triton_config_reduction_gcu(
            size_hints, 64, 8, register_intensive=register_intensive)
        tiny_config = triton_config_reduction_gcu(
            size_hints,
            2 * (256 // rnumel) if rnumel <= 256 else 1,
            min(rnumel, MAX_R0_BLOCK),
            register_intensive=register_intensive,
        )
        if inductor_meta.get("max_autotune") or inductor_meta.get(
                "max_autotune_pointwise"):
            pass  # skip all these cases
        elif reduction_hint == ReductionHint.INNER:
            return [contiguous_config]
        elif reduction_hint == ReductionHint.OUTER:
            return [outer_config]
        elif reduction_hint == ReductionHint.OUTER_TINY:
            return [tiny_config]
        if disable_pointwise_autotuning(inductor_meta):
            return [triton_config_reduction_gcu(size_hints, 32, 128)]
        return [
            contiguous_config,
            outer_config,
            tiny_config,
            triton_config_reduction_gcu(size_hints, 64, 64),
            triton_config_reduction_gcu(size_hints, 8, 512),
            # halve the XBLOCK/RBLOCK compared to outer_config
            # TODO: this may only be beneficial when each iteration of the reduction
            # is quite heavy. E.g. https://gist.github.com/shunting314/189a8ef69f90db9d614a823385147a72
            triton_config_reduction_gcu(size_hints, 64, 4, num_warps=8),
        ]

    def _persistent_reduction_configs_gcu(
        size_hints,
        reduction_hint=False,
        inductor_meta=None,
    ):
        xnumel = size_hints["x"]
        rnumel = get_total_reduction_numel(size_hints)

        # logger.info(f"++++++++++++_persistent_reduction_configs_gcu, xnumel: {xnumel}, rnumel: {rnumel}")

        xblock_max = 1
        for xblock in (1, 2, 4, 8, 16, 32, 64, 128):
            if rnumel * xblock <= 4096 and xblock <= xnumel:
                xblock_max = xblock
        # logger.info(f"++++++++++++_persistent_reduction_configs_gcu, xblock_max: {xblock_max}")
        configs = [
            triton_config_reduction_gcu(size_hints, xblock_max, rnumel, register_intensive=True)
        ]

        # TODO(jansel): we should be able to improve these heuristics
        if reduction_hint == ReductionHint.INNER and rnumel >= 256:
            configs = configs[:1]
        elif reduction_hint == ReductionHint.OUTER:
            configs = configs[-1:]
        elif reduction_hint == ReductionHint.OUTER_TINY:
            configs = [
                triton_config_reduction_gcu(
                    size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, rnumel
                )
            ]
        for c in configs:
            # we don't need Rn_BLOCK for persistent reduction
            for prefix in size_hints:
                if prefix_is_reduction(prefix):
                    c.kwargs.pop(f"{prefix.upper()}BLOCK")

        if disable_pointwise_autotuning(inductor_meta):
            configs = configs[:1]

        return configs

    import sympy
    import logging
    from typing import (
        Any,
        Callable,
        cast,
        Dict,
        Iterable,
        List,
        Optional,
        Tuple,
        TYPE_CHECKING,
        Union,
    )
    #if TYPE_CHECKING:
    #    from torch._inductor.ir import IRNode
    from torch._inductor import config, ir
    from torch._inductor.ir import IRNode
    from torch.utils._ordered_set import OrderedSet
    from torch.utils._sympy.symbol import free_symbol_is_type, prefix_str, symbol_is_type, SymT
    from torch._inductor.runtime.hints import AutotuneHint, DeviceProperties
    from torch._inductor.virtualized import _ops as ops, OpsHandler, ReductionType, StoreMode, V
    from torch._inductor.codegen.common import (
        BackendFeature,
        CSE,
        CSEVariable,
        DeferredLine,
        IndentedBuffer,
        OpOverrides,
        PythonPrinter,
        SizeArg,
        TensorArg,
        WorkspaceArg,
        WorkspaceZeroMode,
    )
    from torch._inductor.codegen.simd import (
        constant_repr,
        IterationRangesEntry,
        IterationRangesRoot,
        pexpr,
        prefix_is_reduction,
        SIMDKernel,
        SIMDScheduling,
    )

    from torch._inductor.codegen.triton import (
        BlockPtrOptions,
        TritonKernelOverrides,
        HelperFunctions,
        BlockParameters,
        texpr,
    )

    from torch._inductor.codegen.triton_utils import (
        config_of,
        should_unwrap_unspec_arg,
        signature_of,
        signature_to_meta,
        non_constexpr_signature,
        equal_1_arg_indices,
    )

    from torch._inductor.utils import (
        cache_on_self,
        get_bounds_index_expr,
        get_fused_kernel_name,
        get_kernel_metadata,
        is_welford_reduction,
        Placeholder,
        sympy_dot,
        sympy_subs,
        triton_version_uses_attrs_dict,
    )

    log = logging.getLogger(__name__)

    def iteration_ranges_codegen_header_gcu(self, entry, code):
        x = entry.prefix
        if entry.is_loop:
            code.writeline(f"{entry.name} = {x}offset + {x}base")
        elif entry.grid_dim is None:
            # no need to "{x}offset = "
            code.writeline(
                f"{entry.name} = {self.iteration_ranges_ranges_code(entry)}")
            code.writeline(f"{x}offset = 0")
        else:
            if entry.tensor_dim is not None:
                line = f"{x}offset + {self.iteration_ranges_ranges_code(entry)}"
            else:
                line = self.iteration_ranges_scalar_code(entry, f"{x}offset")
            # NOTE(GCU) Add loop itera code for pointwise 1d and persistent_reduction case
            if (self._get_heuristic() == "pointwise"
                    and len(self.numels) == 2) or self.persistent_reduction:
                code.writeline(f"{x}offset = {x}loop * {x.upper()}BLOCK")
            else:
                code.writeline(
                    f"{x}offset = {self.iteration_ranges_get_pid(entry)} * {x.upper()}BLOCK"
                )

            code.writeline(f"{entry.name} = {line}")

        if self._has_constant_mask(entry):
            sizes = self.dense_size_str()
            code.writeline(f"{x}mask = tl.full({sizes}, True, tl.int1)")
        else:
            code.writeline(f"{x}mask = {entry.name} < {x}numel")

    def codegen_kernel_gcu(self, name=None):
        code = IndentedBuffer()

        size_hints = {}
        for prefix, numel in self.numels.items():
            if prefix_is_reduction(prefix) and not self.inside_reduction:
                continue

            numel_hint = V.graph.sizevars.symbolic_hint(numel)
            if not isinstance(numel_hint, (int, sympy.Integer)):
                # This default heuristic hint was picked carefully: it is
                # large, to ensure that we don't shrink the block size (since
                # if you don't have many elements, it'd be wasteful to pick a
                # large block size).  Since we don't know how many elements we
                # might have, we should be OK with some inefficiency to make
                # sure we handle the large case well.  8192 is the largest
                # block size we support, so we pick that.
                #
                # If we have a better hint for unbacked SymInts (e.g., because
                # a user told us, or we are tracking upper bounds) we could
                # use that here.
                size_hint = 8192
            else:
                size_hint = next_power_of_2(int(numel_hint))
            size_hints[prefix] = size_hint
        heuristics = self._get_heuristic()
        if name is None:
            code.splice(gen_common_triton_imports())
            device_type = V.graph.get_current_device_or_throw().type
            if device_type == "cpu":
                code.splice("triton_helpers.set_driver_to_cpu()")
            else:
                code.splice("triton_helpers.set_driver_to_gpu()")

            if config.benchmark_kernel:
                code.splice(self.imports_for_benchmark_kernel())

        argdefs, _, signature, _ = self.args.python_argdefs()
        # maps actual expression to SizeArg if it is in sizevars replacements
        for i, arg in enumerate(signature):
            if isinstance(arg, SizeArg):
                # mypy is unhappy about the sympy.Expr
                # type for the key of the dict below
                symbol = cast(sympy.Symbol, arg.expr)
                if symbol in V.graph.sizevars.inv_precomputed_replacements:
                    signature[i] = SizeArg(
                        arg.name,
                        V.graph.sizevars.inv_precomputed_replacements[symbol])

        mutated_args = OrderedSet[str]()
        for mutation in self.mutations:
            if mutation in self.args.input_buffers:
                mutated_args.add(self.args.input_buffers[mutation])
            if (mutation in self.args.inplace_buffers
                    and mutation not in V.graph.removed_buffers
                    and mutation not in self.removed_buffers):
                mutated_args.add(
                    cast(InplacedBuffer,
                         self.args.inplace_buffers[mutation]).inner_name)
            if mutation in self.args.output_buffers:
                mutation_arg = self.args.output_buffers[mutation]
                assert not isinstance(mutation_arg, RemovedArg)
                mutated_args.add(mutation_arg)

        # Note: [Workspace Mutation]
        # workspace arguments are mutated, but are not marked as mutations in self.mutations
        # because their buffers are added during codegen, and aren't tracked during
        # lowering/scheduling. So we add them as mutated_args explicitly below.
        #
        # In the logic below, we only mark the workspaces a mutated if they are marked with
        # zero_fill: that's because, if we don't expect the buffer to be pre-filled with
        # zeros, then, although we still mutate the data, we don't care about those
        # mutations because we don't make any assumptions about the contents of the
        # workspace buffer.  Similarly, ZERO_PER_GRAPH requires the kernel to return
        # the buffer back to its original state.
        for argname, arg in zip(argdefs, signature):
            if (isinstance(arg, WorkspaceArg)
                    and arg.zero_mode == WorkspaceZeroMode.ZERO_ON_CALL):
                mutated_args.add(argname.name)

        mutated_args = sorted(mutated_args)

        for tree in self.active_range_trees():
            sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
            signature.append(sizearg)
            argdefs.append(ArgName(sizearg.name))
            # constexpr version causes issues, see
            # https://github.com/pytorch/torchdynamo/pull/1362
            # triton_meta["constants"][len(argdefs)] = V.graph.sizevars.size_hint(
            #     tree.numel
            # )
            # argdefs.append(f"{tree.prefix}numel: tl.constexpr")

        def add_constexpr_arg(arg_name):
            # new versions (but not old versions) of Triton need constexprs included in the signature
            if triton_version_uses_attrs_dict():
                signature.append(ConstexprArg(arg_name))
            argdefs.append(ArgName(arg_name, is_constexpr=True))

        for tree in self.range_trees:
            if tree.is_reduction and self.persistent_reduction:
                # Rn_BLOCK for persistent_reduction is defined in codegen_static_numels
                continue
            if tree.tensor_dim is None:
                continue

            add_constexpr_arg(f"{tree.prefix.upper()}BLOCK")

        if self.cooperative_reduction:
            add_constexpr_arg("RSPLIT")

        triton_meta_signature = signature_to_meta(signature,
                                                  size_dtype=self.index_dtype,
                                                  argdefs=argdefs)
        triton_meta: dict[str, Any] = {
            "signature":
            triton_meta_signature,
            "device":
            DeviceProperties.create(V.graph.get_current_device_or_throw()),
            "constants": {},
        }

        # Skip memory optimization for forward of the training loop where we expect
        # every new node will increase the peak memory and our greedy approach would
        # introduce a lot of unnecessary cpu copies.
        optimize_mem = V.graph.is_inference or V.graph.is_backward

        inductor_meta = {
            # Triton will not accept an OrderedSet for autotune_hints
            "grid_type": self._get_grid_type().__name__,
            "autotune_hints": set(self.autotune_hints),  # noqa: set_linter
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,
            "optimize_mem": optimize_mem,
            "no_x_dim": self.no_x_dim,
            "num_load": self.num_load,
            "num_reduction": self.num_reduction,
            **self.inductor_meta_common(),
        }
        if self.cooperative_reduction:
            inductor_meta["persistent_reduction"] = self.persistent_reduction

        num_gb = None
        if config.benchmark_kernel or config.profile_bandwidth:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb

        triton_meta["configs"] = [config_of(signature)]

        # Triton compiler includes equal_to_1 args into constants even
        # when they are not constexpr. otherwise there may be a segfault
        # during launching the Inductor-compiled Triton kernel.
        # https://github.com/pytorch/pytorch/issues/120478#issuecomment-1962822307
        # https://github.com/openai/triton/blob/231efe9ed2d200be0f69a07c298e4342b08efe3d/python/triton/runtime/jit.py#L384
        for arg_num in equal_1_arg_indices(signature):  # type: ignore[index]
            triton_meta["constants"][
                signature[arg_num].name] = 1  # type: ignore[index,union-attr]

        self.triton_meta = triton_meta

        self.codegen_body()

        for helper in self.helper_functions:
            code.writeline("")
            code.splice(helper)

        if self.fixed_config:
            heuristics_line = f"""
                @triton_heuristics.{self._get_heuristic()}(
                    config={self.fixed_config.config!r},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        elif self.inside_reduction:
            reduction_hint = self.features.get_reduction_hint()
            heuristics_line = f"""
                @triton_heuristics.{self._get_heuristic()}(
                    size_hints={size_hints!r},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        else:
            tile_hint = ""
            if len(size_hints) == 2:
                if (len(non_constexpr_signature(signature)) == 4
                    ):  # input, output and 2 args
                    tile_hint = "tile_hint=TileHint.SQUARE,"
                else:
                    tile_hint = "tile_hint=TileHint.DEFAULT,"
            heuristics_line = f"""
                @triton_heuristics.{self._get_heuristic()}(
                    size_hints={size_hints!r}, {tile_hint}
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r},
                    min_elem_per_thread={self.min_elem_per_thread}
                )
                @triton.jit
            """
        code.splice(heuristics_line)
        code.writeline(
            f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(x.full_name() for x in argdefs)}):"
        )
        with code.indent():
            self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            # NOTE(GCU) Add loop control code for pointwise 1d and persistent_reduction case
            if (heuristics == "pointwise"
                    and len(self.numels) == 2) or self.persistent_reduction:
                # add gcu loop at this stage only for pointwise XBLOCK
                code.writeline(
                    "# GCU loop support pointwise and persistent_reduction among XBLOCK"
                )
                code.writeline(f"gridsize = tl.num_programs(0)")
                code.writeline(f"xloop_num = (xnumel + XBLOCK - 1) // XBLOCK")
                code.writeline(
                    f"for xloop in range(tl.program_id(0), xloop_num, gridsize):"
                )
                with code.indent():
                    code.splice(self.body)
            else:
                code.splice(self.body)

        if config.benchmark_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb))

        return code.getvalue()

    def want_no_x_dim_gcu(self):
        # NOTE(GCU) GCU No X dim is not supported
        # TODO(GCU) support No X dim NEED OR NOT NEED?
        # if self.persistent_reduction and len(self.numels) == 2:
        #     if self.fixed_config:
        #         return self.fixed_config["XBLOCK"] == 1
        #     return V.choices.want_no_x_dim(self.features)
        return False

    def ceildiv(numel: Union[str, int], block: Union[None, int, str]) -> Union[str, int]:
        mode = "python"
        if block is None or block == 1:
            return numel
        if isinstance(numel, int) and isinstance(block, int):
            return ceildiv(numel, block)  # constant fold
        if mode == "python":
            return f"-(({numel}) // -({block}))"
        # trick above doesn't work in C++ due to rounding differences
        return f"(({numel} + ({block} - 1)) / ({block}))"

    def gcu_get_grid_dim(numel: Union[str, int], block: Union[None, int, str]):
        if numel is None:
            return 1
        if block is None:
            return numel
        return ceildiv(numel, block)

    def gcu_Grid1D_generate(self, meta: dict[str, int]):
        _num_warps = meta.get("num_warps", 1)
        self.x_grid = 24 // _num_warps if meta.get(
            "R0_BLOCK", None) is None and meta.get(
                "XBLOCK", 1) > 24 else gcu_get_grid_dim(
                    "xnumel", meta.get("XBLOCK", 1))
        return self.x_grid

    def gcu_Grid2D_generate(self, meta: dict[str, int]):
        self.x_grid = gcu_Grid1D_generate(self, meta)
        self.y_grid = gcu_get_grid_dim("ynumel", meta.get("YBLOCK", None))
        max_y_grid = get_max_y_grid()
        div = ceildiv(self.y_grid, max_y_grid)
        self.y_grid = ceildiv(self.y_grid, div)
        return self.x_grid, self.y_grid

    def gcu_Grid3D_generate(self, meta: dict[str, int]):
        self.x_grid, self.y_grid = gcu_Grid2D_generate(self, meta)
        self.z_grid = self.ceildiv("znumel", meta.get("ZBLOCK"))

    def triton_gcu_make_launcher(self):
        """
        Launching triton kernels is performance sensitive, we compile
        a custom Python function get the grid() and reorder the args to
        the underlying wrapper.
        """
        cfg = self.config
        compile_meta = self.compile_meta
        binary = self.kernel
        fn = binary.src.fn
        binary._init_handles()
        """
        https://github.com/pytorch/pytorch/issues/115344

        self.fn.constexprs doesn't properly deal with None args, so when we filter out
        an arg in UserDefinedTritonKernel.codegen, we need to filter it here as well.
        We also don't want to modify self.fn.

        We know that we removed something from the signature if:
            1. It's in compile_meta["constants"]
            2. It isn't a constant we already know about
                Note: The value of interest has already been added to compile_meta['constants'],
                    so we use self.fn.constexprs instead.
            3. It isn't in the compile_meta signature
        """
        known_constants = OrderedSet(arg for i, arg in enumerate(fn.arg_names)
                                     if i in fn.constexprs)
        none_args = OrderedSet(k for k, v in compile_meta["constants"].items()
                               if v is None and k not in known_constants)
        none_args = none_args.difference(
            OrderedSet(compile_meta["signature"].keys()))

        if triton_version_uses_attrs_dict():
            call_args = fn.arg_names
            def_args = fn.arg_names
            if ("num_warps" in compile_meta["constants"]
                    or "num_stages" in compile_meta["constants"]):
                # num_warps/num_stages are special implicit args that are not in the signature
                # see test_triton_kernel_special_params
                def_args = [
                    arg for arg in def_args
                    if arg not in ("num_warps", "num_stages")
                ]
                repl = {
                    k: str(compile_meta["constants"].get(k))
                    for k in ("num_warps", "num_stages")
                }
                call_args = [repl.get(arg, arg) for arg in call_args]
        else:
            call_args = [
                arg for i, arg in enumerate(fn.arg_names)
                if i not in fn.constexprs and arg not in none_args
            ]
            cfg_dict = config_to_dict(cfg)
            def_args = [
                name for name in fn.arg_names
                if name not in cfg_dict and name not in none_args
            ]

        binary_shared = (binary.shared if hasattr(binary, "shared") else
                         binary.metadata.shared)

        scope = {
            "grid_meta":
            cfg.kwargs,
            "bin":
            binary,
            "launch_enter_hook":
            binary.__class__.launch_enter_hook,
            "launch_exit_hook":
            binary.__class__.launch_exit_hook,
            "metadata": (binary.packed_metadata if hasattr(
                binary, "packed_metadata") else binary.metadata),
            "shared":
            binary_shared,
            "num_warps": (binary.num_warps if hasattr(binary, "num_warps") else
                          binary.metadata.num_warps),
            "cta_args":
            ((
                binary.num_ctas,
                *get_first_attr(binary, "cluster_dims", "clusterDims"),
            ) if hasattr(binary, "num_ctas") else
             ((binary.metadata.num_ctas, *binary.metadata.cluster_dims)
              if hasattr(binary, "metadata") else ())),
            "function":
            get_first_attr(binary, "function", "cu_function"),
            "runner":
            get_first_attr(binary, "run", "c_wrapper"),
        }

        if not hasattr(binary, "launch_metadata"):
            # launch args before CompiledKernel.launch_metadata is added.
            # TODO(jansel): delete this branch in mid-2025
            runner_args = [
                "grid_0",
                "grid_1",
                "grid_2",
                "num_warps",
                "*cta_args",
                "shared",
                "stream",
                "function",
                "launch_enter_hook",
                "launch_exit_hook",
                "metadata",
                *call_args,
            ]
        else:  # args after CompiledKernel.launch_metadata: https://github.com/openai/triton/pull/3492
            # Getting the kernel launch args is extremely perf-sensitive.  Evaluating
            # `bin.launch_metadata` is relatively expensive, and returns None unless a
            # `launch_enter_hook` is installed.  So if we don't have that hook installed,
            # we want to burn None in to the launch args with zero overhead.
            # See https://github.com/pytorch/pytorch/issues/123597
            if binary.__class__.launch_enter_hook:
                launch_metadata = f"bin.launch_metadata((grid_0, grid_1, grid_2), stream, {', '.join(call_args)})"
            else:
                launch_metadata = "None"
            runner_args = [
                "grid_0",
                "grid_1",
                "grid_2",
                "stream",
                "function",
                "metadata",
                launch_metadata,
                "launch_enter_hook",
                "launch_exit_hook",
                *call_args,
            ]

        if "extra_launcher_args" in self.inductor_meta:
            def_args = [*def_args, *self.inductor_meta["extra_launcher_args"]]

        grid = GridExpr.from_meta(self.inductor_meta, cfg)

        # grid.prefix is usually empty, grid.x_grid is something like `-(xnumel//-1024)`
        lines = [
            f"def launcher({', '.join(def_args)}, stream):",
            *[f"    {line}" for line in grid.prefix],
            f"    grid_0 = {grid.x_grid}",
            f"    grid_1 = {grid.y_grid}",
            f"    grid_2 = {grid.z_grid}",
            f"    runner({', '.join(runner_args)})",
        ]
        exec("\n".join(lines), scope)

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = binary_shared
        launcher.store_cubin = self.inductor_meta.get("store_cubin", False)
        # store this global variable to avoid the high overhead of reading it when calling run
        if launcher.store_cubin:
            launcher.fn = fn
            launcher.bin = binary
            if triton_version_uses_attrs_dict():
                # arg filtering wasn't done above
                cfg_dict = config_to_dict(cfg)
                def_args = [x for x in def_args if x not in cfg_dict]
                call_args = [
                    x for x in call_args if compile_meta["signature"].get(
                        x, "constexpr") != "constexpr" and x not in none_args
                ]
            launcher.def_args = def_args
            launcher.call_args = call_args
        return launcher

    @cache_on_self
    def write_triton_header_once(self) -> None:
        from torch._inductor.runtime import triton_heuristics
        # NOTE(GCU): particular for triton_gcu
        import_str = f"""
            import triton
            import triton.language as tl
            import triton_gcu.triton
            from {triton_heuristics.__name__} import start_graph, end_graph
            """
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.splice(import_str)
            self.kernel_autotune_calls.writeline(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream"))
        if not V.graph.cpp_wrapper:
            self.imports.splice(import_str, strip=True)
            self.imports.writeline(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream"))

    def get_warpsmax(self):
        # NOTE(GCU): for cuda:
        # return 1024 // 32,
        # but for gcu just return 4
        return 4

    def print_performance(fn,
                          args=(),
                          times=10,
                          repeat=10,
                          baseline=1.0,
                          device: str = "gcu"):
        from torch._inductor.utils import timed
        device = 'gcu'
        timings = torch.tensor(
            [timed(fn, args, times, device) for _ in range(repeat)])
        took = torch.median(timings) / times
        print(f"{took / baseline:.6f}")
        return took

    @staticmethod
    @functools.lru_cache(None)
    def gcu_codecache_get_system():
        import hashlib
        import json

        try:
            from triton.compiler.compiler import triton_key

            # Use triton_key instead of triton.__version__ as the version
            # is not updated with each code change
            triton_version = triton_key()
        except ModuleNotFoundError:
            triton_version = None

        try:
            system: Dict[str, Any] = {
                "device": {
                    "name": None
                },
                "version": {
                    "triton": triton_version,
                },
            }
            device_properties = torch.gcu.get_device_properties(
                torch.gcu.current_device())

            # NOTE(GCU): just keep only the cuda branch
            system["device"]["name"] = device_properties.name
            system["version"]["gcu"] = torch.version.cuda

        except (AssertionError, RuntimeError):
            # If cuda is not installed, none of the above config is relevant.
            system = {}

        system["hash"] = hashlib.sha256(
            json.dumps(system, sort_keys=True).encode("utf-8")).hexdigest()

        return system

    def gcu_L2_cache_size(self) -> int:
        """Get the L2 cache size, in bytes, of the current device."""
        L2_cache_size = 1024 * 1024
        return L2_cache_size

    # torch._inductor.runtime.triton_heuristics.CachingAutotuner._precompile_config = _precompile_config_gcu
    torch._inductor.runtime.triton_heuristics.Grid1D.generate = gcu_Grid1D_generate
    torch._inductor.runtime.triton_heuristics.Grid2D.generate = gcu_Grid2D_generate
    torch._inductor.runtime.triton_heuristics.Grid3D.generate = gcu_Grid3D_generate
    # torch._inductor.runtime.triton_heuristics.TritonCompileResult.make_launcher = triton_gcu_make_launcher

    torch._inductor.runtime.benchmarking.InductorBenchmarker.L2_cache_size = property(
        gcu_L2_cache_size)

    torch._inductor.codegen.triton.TritonKernel.iteration_ranges_codegen_header = iteration_ranges_codegen_header_gcu
    torch._inductor.codegen.triton.TritonKernel.codegen_kernel = codegen_kernel_gcu
    torch._inductor.codegen.triton.TritonKernel.want_no_x_dim = want_no_x_dim_gcu

    #torch._inductor.codegen.triton.TritonKernel = TritonGCUKernel
    #torch._inductor.select_algorithm.TritonKernel = torch._inductor.codegen.triton.TritonKernel
    #torch._inductor.codegen.triton_combo_kernel.TritonKernel = torch._inductor.codegen.triton.TritonKernel
    #torch._inductor.codegen.triton_split_scan.TritonKernel = torch._inductor.codegen.triton.TritonKernel
    #torch._inductor.codegen.wrapper.TritonKernel = torch._inductor.codegen.triton.TritonKernel

    import torch._inductor.runtime.triton_heuristics
    torch._inductor.runtime.triton_heuristics.benchmarker = torch._inductor.runtime.benchmarking.benchmarker

    torch._inductor.graph.GraphLowering.compile_to_module = patched_compile_to_module
    torch._inductor.graph.GraphLowering.init_wrapper_code = gcu_init_wrapper_code
    torch._inductor.runtime.triton_heuristics.TRITON_MAX_BLOCK = TRITON_MAX_BLOCK_GCU
    # torch._inductor.runtime.triton_heuristics.grid = new_grid
    torch._inductor.runtime.triton_heuristics.triton_config = triton_config_gcu
    torch._inductor.runtime.triton_heuristics._reduction_configs = _reduction_configs_gcu
    torch._inductor.runtime.triton_heuristics.triton_config_reduction = triton_config_reduction_gcu
    torch._inductor.runtime.triton_heuristics._persistent_reduction_configs = _persistent_reduction_configs_gcu
    torch._inductor.codegen.triton.gen_common_triton_imports = gen_common_triton_imports
    torch._inductor.utils.is_gpu = is_gpu_or_gcu
    torch._inductor.ir.is_triton = is_triton

    # NOTE(GCU): Write 'import triton_gcu.triton' in output to lazy effect of this monkey patch
    torch._inductor.codegen.wrapper.PythonWrapperCodegen.write_triton_header_once = write_triton_header_once

    # NOTE(GCU): to get gcu max warps
    torch._inductor.runtime.coordinate_descent_tuner.CoordescTuner.get_warpsmax = get_warpsmax

    # NOTE(GCU): for print performance default device is cuda, change to gcu
    torch._inductor.utils.print_performance = print_performance

    # NOTE(GCU): in torch._inductor.codecache.CacheBase.get_system, if torch.version.cuda is None,
    # it will use hip device properties of gcnArchName, but our device do not have.
    torch._inductor.codecache.CacheBase.get_system = gcu_codecache_get_system


###################################
# end of monkey patching
###################################
# disable inductor as default
gcu_inductor_enable: bool = os.environ.get(f"TORCHGCU_INDUCTOR_ENABLE", "0") == "1"

if gcu_inductor_enable:
    import importlib.util
    try:
        if importlib.util.find_spec("triton.backends.enflame") is None:
            import triton_gcu.triton
    except (ImportError, ModuleNotFoundError) as e:
        gcu_inductor_enable = False
        print(
            f"triton_gcu not installed, gcu inductor is disabled. Warning: {e}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to import triton_gcu due to internal error during import: {e}"
        ) from e

def get_gcu_interface():
    global _gcu_interface_cls,gcu_inductor_enable
    if gcu_inductor_enable and not _IS_INDUCTOR_PATCHED:
        apply_gcu_inductor_monkey_patch()
    return _gcu_interface_cls

def get_gcu_device_op_overrides():
    global _gcu_device_op_overrides,gcu_inductor_enable
    if gcu_inductor_enable and not _IS_INDUCTOR_PATCHED:
        apply_gcu_inductor_monkey_patch()
    return _gcu_device_op_overrides

if gcu_inductor_enable:
    # lazy use monkey patch
    ##################################
    # lazy method 1: if torch.compile is called, and backend='inductor', then apply monkey patch
    ##################################
    # get original torch.compile function
    _original_torch_compile = torch.compile

    @functools.wraps(_original_torch_compile)
    def _gcu_inductor_compile(model=None, backend="inductor", **kwargs):
        if model is None:
            def decorator(func_or_model):
                return _original_torch_compile(func_or_model, backend=backend, **kwargs)

            if backend == "inductor":
                apply_gcu_inductor_monkey_patch()
            return decorator

        if backend == "inductor":
            apply_gcu_inductor_monkey_patch()
        return _original_torch_compile(model, backend=backend, **kwargs)

    # Apply the monkey patch after torch.compile function
    torch.compile = _gcu_inductor_compile

    ##################################
    # lazy method 2: if triton_gcu is imported, then apply monkey patch
    ##################################
    import sys
    from importlib.abc import MetaPathFinder

    class CustomPatchFinder(MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "triton_gcu":
                global _IS_INDUCTOR_PATCHED
                if not _IS_INDUCTOR_PATCHED:
                    apply_gcu_inductor_monkey_patch()
            return None

    sys.meta_path.insert(0, CustomPatchFinder())
