import torch_gcu
from torch_gcu.distributed.distributed_c10d import all_to_all_vd, all_reduce_outplace,get_unique_id,all_gather_into_tensor_v,reduce_scatter_tensor_v

from torch_gcu.distributed.checkpoint.filesystem import _OverlappingCpuLoader

def is_available():
    """
    Returns ``True`` if the distributed package is available. Otherwise,
    ``torch.distributed`` does not expose any other APIs.
    """
    return hasattr(torch_gcu._C, "_c10d_gcu_init")


if is_available() and not torch_gcu._C._c10d_gcu_init():
    raise RuntimeError("Failed to initialize torch_gcu.distributed")
