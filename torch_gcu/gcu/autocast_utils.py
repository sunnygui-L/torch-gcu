import torch
import torch_gcu

__all__ = ["get_amp_supported_dtype", "is_autocast_enabled", "set_autocast_enabled", "get_autocast_dtype",
           "set_autocast_dtype"]


def get_amp_supported_dtype():
    support_list = [
        torch.float32,
        torch.float,
        torch.half,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.short,
        torch.int32,
        torch.int,
    ]
    if torch.gcu.is_bf16_supported():
        support_list.extend([torch.bfloat16, torch.float16])
    else:
        support_list.append(torch.float16)
    return support_list


def is_autocast_enabled():
    return torch_gcu._C.is_autocast_enabled()


def set_autocast_enabled(enable):
    torch_gcu._C.set_autocast_enabled(enable)


def get_autocast_dtype():
    return torch_gcu._C.get_autocast_dtype()


def set_autocast_dtype(dtype):
    return torch_gcu._C.set_autocast_dtype(dtype)
