# mypy: allow-untyped-defs
"""
``torch.autograd`` provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions.

It requires minimal changes to the existing code - you only need to declare :class:`Tensor` s
for which gradients should be computed with the ``requires_grad=True`` keyword.
As of now, we only support autograd for floating point :class:`Tensor` types (
half, float, double and bfloat16) and complex :class:`Tensor` types (cfloat, cdouble).
"""

import torch_gcu
import torch_gcu._C

if not torch_gcu._C._autograd_init():
    raise RuntimeError("autograd initialization failed")

# Import all native method/classes
from torch_gcu._C._autograd import (
    _add_metadata_json,
    _disable_profiler,
    _enable_profiler,
    _kineto_step,
    _KinetoEvent,
    _prepare_profiler,
    _profiler_enabled,
    _ProfilerResult,
    _record_function_with_args_enter,
    _record_function_with_args_exit,
    _supported_activities,
    _toggle_collection_dynamic,
    kineto_available,
)
from torch._C._autograd import DeviceType
from torch_gcu._C._profiler import ProfilerConfig
from torch._C._profiler import ProfilerActivity, ProfilerState
from . import profiler