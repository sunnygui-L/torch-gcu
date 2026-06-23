# mypy: allow-untyped-defs
r"""
PyTorch Profiler is a tool that allows the collection of performance metrics during training and inference.
Profiler's context manager API can be used to better understand what model operators are the most expensive,
examine their input shapes and stack traces, study device kernel activity and visualize the execution trace.

.. note::
    An earlier version of the API in :mod:`torch.autograd` module is considered legacy and will be deprecated.

"""
import os
import torch
import torch_gcu
import torch_gcu._C
from torch_gcu._C._autograd import _supported_activities, kineto_available
from torch._C._autograd import DeviceType

from torch_gcu._C._profiler import _ExperimentalConfig
from torch._C._profiler import ProfilerActivity
from torch._C._profiler import RecordScope
from torch_gcu.gcu.autograd.profiler import KinetoStepTracker, record_function
from torch.optim.optimizer import register_optimizer_step_post_hook

from torch_gcu.gcu.profiler.profiler import (
    _KinetoProfile,
    ExecutionTraceObserver,
    profile,
    ProfilerAction,
    schedule,
    supported_activities,
    tensorboard_trace_handler,
)

__all__ = [
    "profile",
    "schedule",
    "supported_activities",
    "tensorboard_trace_handler",
    "ProfilerAction",
    "ProfilerActivity",
    "kineto_available",
    "DeviceType",
    "record_function",
    "ExecutionTraceObserver",
]


def _optimizer_post_hook(optimizer, args, kwargs):
    KinetoStepTracker.increment_step("Optimizer")


if os.environ.get("KINETO_USE_DAEMON", None):
    _ = register_optimizer_step_post_hook(_optimizer_post_hook)

def apply_profiler_patch():
    torch.profiler = torch_gcu.gcu.profiler
    setattr(ProfilerActivity, "GCU", ProfilerActivity.PrivateUse1)