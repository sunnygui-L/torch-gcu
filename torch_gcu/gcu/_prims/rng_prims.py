import torch
import torch_gcu
from torch._C import DispatchKey
from torch import _prims as prims


def get_device(args, kwargs):
    if kwargs.get("device"):
        device = kwargs.get("device")
        if isinstance(device, str):
            device = torch.device(device)
        return device.type

    devices = {arg.device.type for arg in args if isinstance(
        arg, torch.Tensor)}
    if any(dev == "cuda" for dev in devices):
        return "cuda"
    elif any(dev == "gcu" for dev in devices):
        return "gcu"
    elif any(dev == "cpu" for dev in devices):
        return "cpu"
    return None


def register_run_and_save_rng_state_op():
    run_and_save_rng_state = prims.rng_prims.run_and_save_rng_state

    @run_and_save_rng_state.py_impl(DispatchKey.PrivateUse1)
    def impl_gcu(op, *args, **kwargs):
        return torch.gcu.get_rng_state(), op(*args, **kwargs)

    if DispatchKey.BackendSelect in run_and_save_rng_state.py_kernels:
        del run_and_save_rng_state.py_kernels[DispatchKey.BackendSelect]

    @run_and_save_rng_state.py_impl(DispatchKey.BackendSelect)
    def impl_backend_select(op, *args, **kwargs):
        impl_map = {
            "cuda": run_and_save_rng_state.py_kernels[DispatchKey.CUDA],
            "cpu": run_and_save_rng_state.py_kernels[DispatchKey.CPU],
            "gcu": impl_gcu
        }
        device = get_device(args, kwargs)
        assert device in impl_map, f"Backend not supported for {device}"
        impl = impl_map[device]
        return impl(op, *args, **kwargs)

    return run_and_save_rng_state


def register_run_with_rng_state_op():
    run_with_rng_state = prims.rng_prims.run_with_rng_state

    @run_with_rng_state.py_impl(DispatchKey.PrivateUse1)
    def impl_gcu(rng_state, op, *args, **kwargs):
        current_state = torch.gcu.get_rng_state()
        torch.gcu.set_rng_state(rng_state.cpu())
        out = op(*args, **kwargs)
        torch.gcu.set_rng_state(current_state)
        return out

    if DispatchKey.BackendSelect in run_with_rng_state.py_kernels:
        del run_with_rng_state.py_kernels[DispatchKey.BackendSelect]

    @run_with_rng_state.py_impl(DispatchKey.BackendSelect)
    def impl_backend_select(rng_state, op, *args, **kwargs):
        impl_map = {
            "cuda": run_with_rng_state.py_kernels[DispatchKey.CUDA],
            "cpu": run_with_rng_state.py_kernels[DispatchKey.CPU],
            "gcu": impl_gcu
        }
        device = get_device(args, kwargs)
        assert device in impl_map, f"Backend not supported for {device}"
        impl = impl_map[device]
        return impl(rng_state, op, *args, **kwargs)

    return run_with_rng_state


run_and_save_rng_state = register_run_and_save_rng_state_op()
run_with_rng_state = register_run_with_rng_state_op()
