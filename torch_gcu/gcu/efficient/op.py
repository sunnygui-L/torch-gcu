from typing import (
    List,
    Optional,
)
from torch import Tensor
import torch_gcu


def gcu_index(self: Tensor, indices: Optional[List[Tensor]] ):
    """index op for int32 indices which will not convert int32 indices to int64.

    Args:
      self: tensor to be selected.
      indices: list of int32 indices.
    """
    return torch_gcu._C._int32_indices_index(self, indices)
