#
# Copyright 2021-2023 Enflame. All Rights Reserved.
#
import os,sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Disable autoloading before running 'import torch' to avoid circular dependencies
ORG_AUTOLOAD = os.getenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch

import torch_gcu
import torch_gcu._C
import torch_gcu.distributed
import torch_gcu.gcu
import torch_gcu.gcu.amp
import torch_gcu.gcu._prims
import torch_gcu.gcu.efficient
import types
import sys
import warnings
from torch_gcu.gcu import (
    apply_storage_patch,
    apply_reductions_patch,
    apply_narrow,
    generate_methods_for_tensor,
    get_gcu_data_ptr,
    get_gcu_device_ptr,
)
import torch_gcu.gcu.overrides

if torch_gcu._C._support_kineto_gcu():
    import torch_gcu.gcu.autograd
    import torch_gcu.gcu.profiler
    from torch_gcu.gcu.profiler import apply_profiler_patch
else:
    warnings.warn(
        "torch_gcu not compiled with KINETO_GCU enabled, GCU events will not be recorded in profiler"
    )

from torch_gcu.version import __version__, git_version


def _apply_patches(monkey_patches):

    def _getattr(module_list, root_module=torch):
        if len(module_list) <= 1:
            return root_module

        if hasattr(root_module, module_list[0]):
            return _getattr(module_list[1:], getattr(root_module, module_list[0]))
        else:
            empty_module_name = f"{root_module.__name__}.{module_list[0]}"
            sys.modules[empty_module_name] = types.ModuleType(empty_module_name)
            setattr(root_module, module_list[0], sys.modules.get(empty_module_name))
            return _getattr(module_list[1:], getattr(root_module, module_list[0]))

    for patch_pair in monkey_patches:
        dest, patch = patch_pair
        dest_module = _getattr(dest.split("."), root_module=torch)
        last_module_level = dest.split(".")[-1]
        if not isinstance(patch, types.ModuleType):
            setattr(dest_module, last_module_level, patch)
            continue

        if not hasattr(dest_module, last_module_level) or not hasattr(patch, "__all__"):
            setattr(dest_module, last_module_level, patch)
            sys.modules[f"{dest_module.__name__}.{last_module_level}"] = patch
            continue

        if not hasattr(patch, "__all__"):
            raise NotImplementedError("Patch module must have __all__ definition.")
        dest_module = getattr(dest_module, last_module_level)
        for attr in patch.__all__:
            setattr(dest_module, attr, getattr(patch, attr))


torch.utils.rename_privateuse1_backend("gcu")
torch._register_device_module("gcu", torch_gcu.gcu)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

torch.distributed.is_eccl_available = torch_gcu.distributed.is_available

generate_methods_for_tensor("gcu_data_ptr", get_gcu_data_ptr)
generate_methods_for_tensor("gcu_device_ptr", get_gcu_device_ptr)

apply_narrow()

apply_storage_patch()

# this must be placed at the end
torch_gcu._C._initGcuTensorType()

# init and register eccl backend
torch.distributed.Backend.register_backend(
    "eccl",
    lambda store, group_rank, group_size, timeout: torch_gcu._C._distributed_c10d.ProcessGroupECCL(
        store, group_rank, group_size, timeout
    ),
    devices=["gcu"],
)

apply_reductions_patch()
if torch_gcu._C._support_kineto_gcu():
    apply_profiler_patch()

# 导出topsLaunchHostFunc的Python接口
def launch_host_func(func, *args, **kwargs):
    torch_gcu._C.launch_host_func(func, *args, **kwargs)

is_transfer_to_gcu = False


# This function is an entrypoint called by PyTorch
# when running 'import torch'. There is no need to do anything.
def _autoload():
    # We should restore this switch as sub processes need to inherit its value
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = ORG_AUTOLOAD

from torch.gcu.inductor import backend_register
