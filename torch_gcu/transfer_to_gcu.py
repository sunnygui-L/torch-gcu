import os
import warnings
from functools import wraps
import torch
import torch.version
import torch_gcu.version
import torch_gcu
import re
import sys
from torch.utils._device import _device_constructors, DeviceContext
import torch.distributed.fsdp

warnings.filterwarnings(action="once")

torch_fn_white_list = [
    "logspace",
    "randint",
    "hann_window",
    "rand",
    "full_like",
    "ones_like",
    "rand_like",
    "randperm",
    "arange",
    "frombuffer",
    "normal",
    "_empty_per_channel_affine_quantized",
    "empty_strided",
    "empty_like",
    "scalar_tensor",
    "tril_indices",
    "bartlett_window",
    "ones",
    "sparse_coo_tensor",
    "randn",
    "kaiser_window",
    "tensor",
    "triu_indices",
    "as_tensor",
    "zeros",
    "randint_like",
    "full",
    "eye",
    "_sparse_csr_tensor_unsafe",
    "empty",
    "_sparse_coo_tensor_unsafe",
    "blackman_window",
    "zeros_like",
    "range",
    "sparse_csr_tensor",
    "randn_like",
    "from_file",
    "_cudnn_init_dropout_state",
    "_empty_affine_quantized",
    "linspace",
    "hamming_window",
    "empty_quantized",
    "_pin_memory",
    "autocast",
    "load",
]
torch_tensor_fn_white_list = [
    "new_empty",
    "new_empty_strided",
    "new_full",
    "new_ones",
    "new_tensor",
    "new_zeros",
    "to",
]
torch_module_fn_white_list = ["to", "to_empty"]
torch_cuda_fn_white_list = [
    "get_device_properties",
    "get_device_name",
    "get_device_capability",
    "list_gpu_processes",
    "set_device",
    "synchronize",
    "mem_get_info",
    "memory_stats",
    "memory_summary",
    "memory_allocated",
    "max_memory_allocated",
    "reset_max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "reset_max_memory_cached",
    "reset_peak_memory_stats",
    "device",
]
torch_distributed_fn_white_list = ["__init__"]
device_kwargs_list = ["device", "device_type", "map_location"]

# record "original-wrapped" function pairs
wrapped_func_map = {}

has_is_aten_op_or_tensor_method_patched = False


def wrapper_cuda(fn):

    @wraps(fn)
    def decorated(*args, **kwargs):
        replace_int = fn.__name__ in ["to", "to_empty"]
        if args:
            args_new = list(args)
            args = replace_cuda_to_gcu_in_list(args_new, replace_int)
        if kwargs:
            for device_arg in device_kwargs_list:
                device = kwargs.get(device_arg, None)
                if type(device) == str and "cuda" in device:
                    kwargs[device_arg] = device.replace("cuda", "gcu")
                if type(device) == torch.device and "cuda" in device.type:
                    device_info = (
                        "gcu:{}".format(device.index)
                        if device.index is not None
                        else "gcu"
                    )
                    kwargs[device_arg] = torch.device(device_info)
                if type(device) == int:
                    kwargs[device_arg] = f"gcu:{device}"
            device_ids = kwargs.get("device_ids", None)
            if type(device_ids) == list:
                device_ids = replace_cuda_to_gcu_in_list(device_ids, replace_int)
        if (str(fn) == "<method 'to' of 'torch._C.TensorBase' objects>") or (
            str(fn) == "<method 'new_empty' of 'torch._C.TensorBase' objects>"
        ):
            from torch._dynamo.trace_rules import is_aten_op_or_tensor_method

            global has_is_aten_op_or_tensor_method_patched

            if not has_is_aten_op_or_tensor_method_patched:

                def patched_is_aten_op_or_tensor_method(obj):
                    if obj.__name__ in ["new_empty", "to"]:
                        return True
                    return is_aten_op_or_tensor_method(obj)

                setattr(
                    torch._dynamo.trace_rules,
                    "is_aten_op_or_tensor_method",
                    patched_is_aten_op_or_tensor_method,
                )

                has_is_aten_op_or_tensor_method_patched = True

            return fn(*args, **kwargs)
        else:
            return fn(*args, **kwargs)

    return decorated


def replace_cuda_to_gcu_in_list(args_list, replace_int):
    for idx, arg in enumerate(args_list):
        if type(arg) == str and (re.match("^cuda(:\d+)?$", arg)):
            args_list[idx] = arg.replace("cuda", "gcu")
        if type(arg) == torch.device and "cuda" in arg.type:
            device_info = "gcu:{}".format(arg.index) if arg.index is not None else "gcu"
            args_list[idx] = torch.device(device_info)
        if replace_int and type(arg) != bool and type(arg) == int:
            args_list[idx] = f"gcu:{arg}"
    return args_list


def device_wrapper(enter_fn, white_list):
    for fn_name in white_list:
        fn = getattr(enter_fn, fn_name, None)
        if fn:
            wrapped_fn = wrapper_cuda(fn)
            # record the mappings
            wrapped_func_map[fn] = wrapped_fn
            setattr(enter_fn, fn_name, wrapped_fn)


def _gcu_DeviceContext___torch_function__(self, func, types, args=(), kwargs=None):
    kwargs = kwargs or {}
    wrapped_func = None
    if func in wrapped_func_map:
        wrapped_func = wrapped_func_map[func]
    device_constructors = _device_constructors()
    # NOTE: Cover the case when return values of _device_constructors() ​​are cached
    if (
        func in device_constructors or wrapped_func in device_constructors
    ) and kwargs.get("device") is None:
        kwargs["device"] = self.device
    return func(*args, **kwargs)


def wrapper_eccl(fn):

    @wraps(fn)
    def decorated(*args, **kwargs):
        if args:
            args_new = list(args)
            for idx, arg in enumerate(args_new):
                if type(arg) == str:
                    if "nccl" in arg:
                        args_new[idx] = arg.replace("nccl", "eccl")
                    elif "NCCL" in arg:
                        args_new[idx] = arg.replace("NCCL", "ECCL")
            args = args_new
        if kwargs:
            backend_arg = kwargs.get("backend", None)
            if type(backend_arg) == str:
                backend_arg = backend_arg.lower()
                if "nccl" in backend_arg:
                    backend_arg = backend_arg.replace("nccl", "eccl")
                    if "cuda" in backend_arg:
                        backend_arg = backend_arg.replace("cuda", "gcu")
                    kwargs["backend"] = backend_arg
        return fn(*args, **kwargs)

    return decorated


def wrapper_data_loader(fn):

    @wraps(fn)
    def decorated(*args, **kwargs):
        if kwargs:
            pin_memory = kwargs.get("pin_memory", False)
            pin_memory_device = kwargs.get("pin_memory_device", None)
            if pin_memory and not pin_memory_device:
                kwargs["pin_memory_device"] = "gcu"
            if (
                pin_memory
                and type(pin_memory_device) == str
                and "cuda" in pin_memory_device
            ):
                kwargs["pin_memory_device"] = pin_memory_device.replace("cuda", "gcu")
        return fn(*args, **kwargs)

    return decorated


def wrapper_profiler(fn):

    @wraps(fn)
    def decorated(*args, **kwargs):
        if kwargs:
            if "experimental_config" in kwargs.keys():
                warnings.warn(
                    "The parameter experimental_config of torch.profiler.profile has been deleted by the tool "
                    "because it can only be used in cuda, please manually modify the code "
                    "and use the experimental_config parameter adapted to gcu."
                )
                del kwargs["experimental_config"]
        return fn(*args, **kwargs)

    return decorated


def jit_script(obj, optimize=None, _frames_up=0, _rcb=None, example_inputs=None):
    msg = "torch.jit.script will be disabled by transfer_to_gcu, which currently does not support it."
    warnings.warn(msg, RuntimeWarning)
    return obj


def patch_cuda():
    patches = [
        ["cuda", torch_gcu.gcu],
        ["cuda.amp", torch_gcu.gcu.amp],
        ["cuda.random", torch_gcu.gcu.random],
        ["cuda.amp.autocast_mode", torch_gcu.gcu.amp.autocast_mode],
        ["cuda.amp.common", torch_gcu.gcu.amp.common],
        ["cuda.amp.grad_scaler", torch_gcu.gcu.amp.grad_scaler],
        ["cuda.tunable", torch_gcu.gcu.tunable],
    ]
    torch_gcu._apply_patches(patches)


def patch_prims():
    patches = [
        [
            "_prims.rng_prims.run_and_save_rng_state",
            torch_gcu.gcu._prims.rng_prims.run_and_save_rng_state,
        ],
        [
            "_prims.rng_prims.run_with_rng_state",
            torch_gcu.gcu._prims.rng_prims.run_with_rng_state,
        ],
    ]
    torch_gcu._apply_patches(patches)


def patch_checkpoint():
    torch.distributed.checkpoint.filesystem._OverlappingCpuLoader = (
        torch_gcu.distributed.checkpoint.filesystem._OverlappingCpuLoader
    )


def patch_profiler():
    torch.profiler.ProfilerActivity.CUDA = torch.profiler.ProfilerActivity.GCU


def patch_torch_c():
    for attr_name in dir(torch._C):
        if "cuda" in attr_name or "CUDA" in attr_name:
            torch_gcu_attr_name = attr_name.replace("cuda", "gcu").replace(
                "CUDA", "GCU"
            )
            if hasattr(torch_gcu._C, torch_gcu_attr_name):
                setattr(torch._C, attr_name, getattr(torch_gcu._C, torch_gcu_attr_name))

    # from torch._C import _graph_pool_handle and expect it to work with GCU graphs.
    if hasattr(torch._C, "_graph_pool_handle") and hasattr(
        torch_gcu._C, "_graph_pool_handle"
    ):
        setattr(
            torch._C, "_graph_pool_handle", getattr(torch_gcu._C, "_graph_pool_handle")
        )
    # for data parallel
    data_parallels = [
        "_broadcast_coalesced",
        "_broadcast",
        "_broadcast_out",
        "_scatter",
        "_scatter_out",
        "_gather",
        "_gather_out",
    ]
    for op_name in data_parallels:
        if hasattr(torch_gcu._C, op_name):
            #    print(f"++++++++++++++++++op_name: {op_name}")
            setattr(torch._C, op_name, getattr(torch_gcu._C, op_name))


def patch_amp_autocast():
    # to support "from torch.cuda.amp import autocast"
    torch.cuda.amp.amp_definitely_not_available = (
        torch_gcu.gcu.amp.amp_definitely_not_available
    )
    torch.cuda.amp.autocast = torch_gcu.gcu.amp.autocast
    torch.cuda.amp.custom_bwd = torch_gcu.gcu.amp.custom_bwd
    torch.cuda.amp.custom_fwd = torch_gcu.gcu.amp.custom_fwd
    torch.cuda.amp.GradScaler = torch_gcu.gcu.amp.GradScaler
    torch.cuda.amp.autocast_mode.autocast = torch_gcu.gcu.amp.autocast_mode.autocast
    torch.cuda.amp.autocast_mode.custom_bwd = torch_gcu.gcu.amp.autocast_mode.custom_bwd
    torch.cuda.amp.autocast_mode.custom_fwd = torch_gcu.gcu.amp.autocast_mode.custom_fwd
    torch.cuda.amp.common.amp_definitely_not_available = (
        torch_gcu.gcu.amp.common.amp_definitely_not_available
    )
    torch.cuda.amp.grad_scaler.GradScaler = torch_gcu.gcu.amp.grad_scaler.GradScaler

    # to support "torch.amp.autocast('cuda', ...)" -> "gcu"
    _orig_torch_amp_autocast = torch.amp.autocast

    def _amp_autocast(device_type, dtype=None, enabled=True, cache_enabled=None):
        if isinstance(device_type, torch.device):
            device_type = device_type.type
        if isinstance(device_type, str) and (
            "cuda" in device_type or device_type == "gcu"
        ):
            if dtype is None:
                dtype = torch.get_autocast_dtype("gcu")
            return torch_gcu.gcu.amp.autocast(
                enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
            )
        return _orig_torch_amp_autocast(
            device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled
        )

    torch.amp.autocast = _amp_autocast


def warning_fn(msg, rank0=True):
    # Suppress in worker processes spawned via multiprocessing.spawn to avoid
    # flooding the console with duplicate warnings.
    _init_pid = os.environ.get("_TORCH_GCU_MIGRATION_INIT_PID")
    if _init_pid is not None and _init_pid != str(os.getpid()):
        return

    is_distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )
    env_rank = os.getenv("RANK", None)

    if rank0 and is_distributed:
        if torch.distributed.get_rank() == 0:
            warnings.warn(msg, ImportWarning)
    elif rank0 and env_rank:
        if env_rank == "0":
            warnings.warn(msg, ImportWarning)
    else:
        warnings.warn(msg, ImportWarning)


def _del_nccl_device_backend_map():
    if hasattr(torch.distributed.Backend, "default_device_backend_map"):
        if "cuda" in torch.distributed.Backend.default_device_backend_map:
            del torch.distributed.Backend.default_device_backend_map["cuda"]


def init():
    warning_fn(
        """
    *************************************************************************************************************
    The torch.Tensor.cuda and torch.nn.Module.cuda are replaced with torch.Tensor.gcu and torch.nn.Module.gcu now..
    The backend in torch.distributed.init_process_group set to eccl now..
    The torch.cuda.* and torch.cuda.amp.* are replaced with torch.gcu.* and torch.gcu.amp.* now..
    The device parameters have been replaced with gcu in the function below:
    {}
    *************************************************************************************************************
    """.format(
            ", ".join(
                ["torch." + i for i in torch_fn_white_list]
                + ["torch.Tensor." + i for i in torch_tensor_fn_white_list]
                + ["torch.nn.Module." + i for i in torch_module_fn_white_list]
            )
        )
    )

    device_wrapper(torch.cuda, torch_cuda_fn_white_list)

    # torch.profiler.*
    if torch_gcu._C._support_kineto_gcu():
        patch_profiler()
        torch.profiler.profile = wrapper_profiler(torch.profiler.profile)

    # torch.*
    device_wrapper(torch, torch_fn_white_list)

    # torch.Tensor.*
    device_wrapper(torch.Tensor, torch_tensor_fn_white_list)
    torch.Tensor.cuda = torch.Tensor.gcu
    torch.Tensor.is_cuda = torch.Tensor.is_gcu
    # torch.cuda.DoubleTensor = torch.gcu.FloatTensor

    # torch.nn.Module.*
    device_wrapper(torch.nn.Module, torch_module_fn_white_list)
    torch.nn.Module.cuda = torch.nn.Module.gcu

    # torch.distributed.init_process_group
    torch.distributed.init_process_group = wrapper_eccl(
        torch.distributed.init_process_group
    )
    torch.distributed.is_nccl_available = torch.distributed.is_eccl_available
    torch.distributed.ProcessGroup._get_backend = wrapper_eccl(
        torch.distributed.ProcessGroup._get_backend
    )
    torch.distributed.new_group = wrapper_eccl(torch.distributed.new_group)
    torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel.__init__ = wrapper_cuda(
        torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel.__init__
    )
    if hasattr(torch.distributed, "init_device_mesh"):
        _del_nccl_device_backend_map()
        torch.distributed.init_device_mesh = wrapper_cuda(
            torch.distributed.init_device_mesh
        )

    # torch.nn.parallel.DistributedDataParallel
    device_wrapper(
        torch.nn.parallel.DistributedDataParallel, torch_distributed_fn_white_list
    )
    # torch.utils.data.DataLoader
    torch.utils.data.DataLoader.__init__ = wrapper_data_loader(
        torch.utils.data.DataLoader.__init__
    )

    # torch.cuda.nvtx
    torch.cuda.nvtx = torch.gcu.topstx

    torch.jit.script = jit_script

    # CUDAGraph -> GCUGraph
    torch.cuda.CUDAGraph = torch.gcu.GCUGraph

    # Disable torch.cuda function
    torch.cuda.ipc_collect = lambda *args, **kwargs: None
    torch.cuda.utilization = lambda *args, **kwargs: 0

    patch_amp_autocast()
    # torch.cuda.*
    patch_cuda()

    # torch._prims
    patch_prims()

    # torch._C
    patch_torch_c()

    torch.cuda._lazy_init = torch.gcu._lazy_init

    # torch.device('cuda')
    torch_gcu._C._replace_device_new_method()

    # torch.Generator(device="cuda")
    torch_gcu._C._replace_generator_new_method()

    # with torch.device('cuda')
    DeviceContext.__torch_function__ = _gcu_DeviceContext___torch_function__

    # torch.load("xx.pt", map_location=None)
    torch.serialization.default_restore_location = wrapper_cuda(
        torch.serialization.default_restore_location
    )
    torch_gcu.is_transfer_to_gcu = True

    torch.cuda.Stream.cuda_stream = torch.gcu.Stream.gcu_stream

    torch.cuda.memory.CUDAPluggableAllocator = torch.gcu.memory.GCUPluggableAllocator

    # torch.cuda._device_count_nvml -> torch.gcu._device_count_efml
    torch.cuda._device_count_nvml = torch.gcu._device_count_efml

    patch_checkpoint()


init()
