import collections
import contextlib
import threading
import warnings
import pickle
import sys
from inspect import signature

from typing import Any, Dict, Tuple, Union, Optional

import torch
from torch.types import Device
import torch_gcu
from torch_gcu import _C
from ._utils import _get_device_index


__all__ = [
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "set_per_process_memory_fraction",
    "empty_cache",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "memory_cached",
    "max_memory_cached",
    "memory_snapshot",
    "memory_summary",
    "mem_get_info",
    "get_allocator_backend",
    "tops_malloc_host_accessible",
    "MemPool",
    "use_mem_pool",
]


@contextlib.contextmanager
def _free_mutex():
    _C._gcu_lock_mutex()
    try:
        yield
    finally:
        _C._gcu_unlock_mutex()


def caching_allocator_alloc(size, device: Union[Device, int] = None, stream=None):
    r"""Performs a memory allocation using the GCU memory allocator.

    Memory is allocated for a given device and a stream, this
    function is intended to be used for interoperability with other
    frameworks. Allocated memory is released through
    :func:`~torch.gcu.caching_allocator_delete`.

    Args:
        size (int): number of bytes to be allocated.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default GCU device is used.
        stream (torch.gcu.Stream or int, optional): selected stream. If is ``None`` then
            the default stream for the selected device is used.

    .. note::
        See :ref:`gcu-memory-management` for more details about GCU memory
        management.
    """
    if device is None:
        device = torch.gcu.current_device()
    device = _get_device_index(device)
    if stream is None:
        stream = torch.gcu.current_stream(device)
    if isinstance(stream, torch.gcu.streams.Stream):
        stream = stream.gcu_stream
    if not isinstance(stream, int):
        raise TypeError(
            "Invalid type for stream argument, must be "
            "`torch.gcu.Stream` or `int` representing a pointer "
            "to a existing stream"
        )
    with torch.gcu.device(device):
        return _C._gcu_gcuCachingAllocator_raw_alloc(size, stream)


def caching_allocator_delete(mem_ptr):
    r"""Deletes memory allocated using the GCU memory allocator.

    Memory allocated with :func:`~torch.gcu.caching_allocator_alloc`.
    is freed here. The associated device and stream are tracked inside
    the allocator.

    Args:
        mem_ptr (int): memory address to be freed by the allocator.

    .. note::
        See :ref:`gcu-memory-management` for more details about GCU memory
        management.
    """
    _C._gcu_gcuCachingAllocator_raw_delete(mem_ptr)


def set_per_process_memory_fraction(
    fraction, device: Union[Device, int] = None
) -> None:
    r"""Set memory fraction for a process.
    The fraction is used to limit an caching allocator to allocated memory on a GCU device.
    The allowed value equals the total visible memory multiplied fraction.
    If trying to allocate more than the allowed value in a process, will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~1. Allowed memory equals total_memory * fraction.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default GCU device is used.
    .. note::
        In general, the total available free memory is less than the total capacity.
    """
    torch_gcu.gcu._lazy_init()
    if device is None:
        device = torch.gcu.current_device()
    device = _get_device_index(device)
    if not isinstance(fraction, float):
        raise TypeError("Invalid type for fraction argument, must be `float`")
    if fraction < 0 or fraction > 1:
        raise ValueError(f"Invalid fraction value: {fraction}. Allowed range: 0~1")

    _C._gcu_setMemoryFraction(fraction, device)


def empty_cache() -> None:
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GCU application and visible in
    `enflame-smi`.

    .. note::
        :func:`~torch.gcu.empty_cache` doesn't increase the amount of GCU
        memory available for PyTorch. However, it may help reduce fragmentation
        of GCU memory in certain cases. See :ref:`gcu-memory-management` for
        more details about GCU memory management.
    """
    if torch_gcu.gcu.is_initialized():
        _C._gcu_emptyCache()


def memory_stats(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Returns a dictionary of GCU memory allocator statistics for a
    given device.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from ``topsMalloc()``.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of active memory blocks.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of inactive, non-releasable memory blocks.
    - ``"inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of inactive, non-releasable memory.

    For these core statistics, values are broken down as follows.

    Pool type:

    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool
      (as of October 2019, for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool
      (as of October 2019, for size < 1MB allocations).

    Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    In addition to the core statistics, we also provide some simple event
    counters:

    - ``"num_alloc_retries"``: number of failed ``topsMalloc`` calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.

    The caching allocator can be configured via ENV to not split blocks larger than a
    defined size (see Memory Management section of the Gcu Semantics documentation).
    This helps avoid memory fragmentation but may have a performance
    penalty. Additional outputs to assist with tuning and evaluating impact:

    - ``"max_split_size"``: blocks above this size will not be split.
    - ``"oversize_allocations.{current,peak,allocated,freed}"``:
      number of over-size allocation requests received by the memory allocator.
    - ``"oversize_segments.{current,peak,allocated,freed}"``:
      number of over-size reserved segments from ``topsMalloc()``.

    The caching allocator can be configured via ENV to round memory allocations in order
    to reduce fragmentation. Sometimes the overhead from rounding can be higher than
    the fragmentation it helps reduce. The following stat can be used to check if
    rounding adds too much overhead:

    - ``"requested_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      memory requested by client code, compare this with allocated_bytes to check if
      allocation rounding adds too much overhead.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistics for the current device, given by :func:`~torch.gcu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`gcu-memory-management` for more details about GCU memory
        management.

    .. note::
        With :ref:`backend:topsMallocAsync<gcu-memory-envvars>`, some stats are not
        meaningful, and are always reported as zero.
    """
    result = []

    def _recurse_add_to_result(prefix, obj):
        if isinstance(obj, dict):
            if len(prefix) > 0:
                prefix += "."
            for k, v in obj.items():
                _recurse_add_to_result(prefix + k, v)
        else:
            result.append((prefix, obj))

    stats = memory_stats_as_nested_dict(device=device)
    _recurse_add_to_result("", stats)
    result.sort()

    return collections.OrderedDict(result)


def memory_stats_as_nested_dict(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Returns the result of :func:`~torch.gcu.memory_stats` as a nested dictionary."""
    if not torch_gcu.gcu.is_initialized():
        return {}
    device = _get_device_index(device, optional=True)
    return _C._gcu_memoryStats(device)


def reset_accumulated_memory_stats(device: Union[Device, int] = None) -> None:
    r"""Resets the "accumulated" (historical) stats tracked by the GCU memory allocator.

    See :func:`~torch.gcu.memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict, as well as
    `"num_alloc_retries"` and `"num_ooms"`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.gcu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`gcu-memory-management` for more details about GCU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return _C._gcu_resetAccumulatedMemoryStats(device)


def reset_peak_memory_stats(device: Union[Device, int] = None) -> None:
    r"""Resets the "peak" stats tracked by the GCU memory allocator.

    See :func:`~torch.gcu.memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.gcu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`gcu-memory-management` for more details about GCU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return _C._gcu_resetPeakMemoryStats(device)


def reset_max_memory_allocated(device: Union[Device, int] = None) -> None:
    r"""Resets the starting point in tracking maximum GCU memory occupied by
    tensors for a given device.

    See :func:`~torch.gcu.max_memory_allocated` for details.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.gcu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch.gcu.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.

    .. note::
        See :ref:`gcu-memory-management` for more details about GCU memory
        management.
    """
    warnings.warn(
        "torch.gcu.reset_max_memory_allocated now calls torch.gcu.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
    )
    return reset_peak_memory_stats(device=device)


def reset_max_memory_cached(device: Union[Device, int] = None) -> None:
    r"""Resets the starting point in tracking maximum GCU memory managed by the
    caching allocator for a given device.

    See :func:`~torch.gcu.max_memory_cached` for details.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.gcu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch.gcu.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.

    .. note::
        See :ref:`gcu-memory-management` for more details about GCU memory
        management.
    """
    warnings.warn(
        "torch.gcu.reset_max_memory_cached now calls torch.gcu.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
    )
    return reset_peak_memory_stats(device=device)


def memory_allocated(device: Union[Device, int] = None) -> int:
    r"""Returns the current GCU memory occupied by tensors in bytes for a given
    device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.gcu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This is likely less than the amount shown in `enflame-smi` since some
        unused memory can be held by the caching allocator and some context
        needs to be created on GCU. See :ref:`gcu-memory-management` for more
        details about GCU memory management.
    """
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device: Union[Device, int] = None) -> int:
    r"""Returns the maximum GCU memory occupied by tensors in bytes for a given
    device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.gcu.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.gcu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`gcu-memory-management` for more details about GCU memory
        management.
    """
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)


def memory_reserved(device: Union[Device, int] = None) -> int:
    r"""Returns the current GCU memory managed by the caching allocator in bytes
    for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.gcu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`gcu-memory-management` for more details about GCU memory
        management.
    """
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: Union[Device, int] = None) -> int:
    r"""Returns the maximum GCU memory managed by the caching allocator in bytes
    for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.gcu.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.gcu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`gcu-memory-management` for more details about GCU memory
        management.
    """
    return memory_stats(device=device).get("reserved_bytes.all.peak", 0)


def memory_cached(device: Union[Device, int] = None) -> int:
    r"""Deprecated; see :func:`~torch.gcu.memory_reserved`."""
    warnings.warn(
        "torch.gcu.memory_cached has been renamed to torch.gcu.memory_reserved",
        FutureWarning,
    )
    return memory_reserved(device=device)


def max_memory_cached(device: Union[Device, int] = None) -> int:
    r"""Deprecated; see :func:`~torch.gcu.max_memory_reserved`."""
    warnings.warn(
        "torch.gcu.max_memory_cached has been renamed to torch.gcu.max_memory_reserved",
        FutureWarning,
    )
    return max_memory_reserved(device=device)


def memory_snapshot():
    r"""Returns a snapshot of the GCU memory allocator state across all devices.

    Interpreting the output of this function requires familiarity with the
    memory allocator internals.

    .. note::
        See :ref:`gcu-memory-management` for more details about GCU memory
        management.
    """
    return _C._gcu_memorySnapshot()["segments"]


def memory_summary(device: Union[Device, int] = None, abbreviated: bool = False) -> str:
    r"""Returns a human-readable printout of the current memory allocator
    statistics for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.gcu.current_device`,
            if :attr:`device` is ``None`` (default).
        abbreviated (bool, optional): whether to return an abbreviated summary
            (default: False).

    .. note::
        See :ref:`gcu-memory-management` for more details about GCU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    stats = memory_stats(device=device)

    def _format_size(sz, pref_sz):
        prefixes = ["B  ", "KiB", "MiB", "GiB", "TiB", "PiB"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_sz < 768 * 1024:
                break
            prefix = new_prefix
            sz //= 1024
            pref_sz /= 1024
        return f"{sz:6d} {prefix}"

    def _format_count(cnt, pref_cnt):
        prefixes = [" ", "K", "M"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_cnt < 750 * 1000:
                break
            prefix = new_prefix
            cnt //= 1000
            pref_cnt /= 1000
        return f"{cnt:7d} {prefix} "

    metrics_to_display = [
        ("allocated_bytes", "Allocated memory", _format_size),
        ("active_bytes", "Active memory", _format_size),
        ("requested_bytes", "Requested memory", _format_size),
        ("reserved_bytes", "GCU reserved memory", _format_size),
        ("inactive_split_bytes", "Non-releasable memory", _format_size),
        ("allocation", "Allocations", _format_count),
        ("active", "Active allocs", _format_count),
        ("segment", "GCU reserved segments", _format_count),
        ("inactive_split", "Non-releasable allocs", _format_count),
    ]

    lines = []
    lines.append("=" * 75)
    lines.append(" {_:16} PyTorch GCU memory summary, device ID {device:<17d} ")
    lines.append("-" * 75)
    lines.append(
        "  {_:9} GCU OOMs: {num_ooms:<12d} | {_:6} topsMalloc retries: {num_alloc_retries:<8d}  "
    )
    lines.append("=" * 75)
    lines.append(
        "        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  "
    )

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)
        submetrics = [("all", metric_name)]
        if not abbreviated:
            submetrics.append(("large_pool", "      from large pool"))
            submetrics.append(("small_pool", "      from small pool"))

        current_prefval, peak_prefval, allocated_prefval, freed_prefval = (
            None,
            None,
            None,
            None,
        )

        for submetric_key, submetric_name in submetrics:
            prefix = metric_key + "." + submetric_key + "."

            current = stats[prefix + "current"]
            peak = stats[prefix + "peak"]
            allocated = stats[prefix + "allocated"]
            freed = stats[prefix + "freed"]

            if current_prefval is None:
                current_prefval = current
                peak_prefval = peak
                allocated_prefval = allocated
                freed_prefval = freed

            lines.append(
                " {:<21} | {} | {} | {} | {} ".format(
                    submetric_name,
                    formatter(current, current_prefval),
                    formatter(peak, peak_prefval),
                    formatter(allocated, allocated_prefval),
                    formatter(freed, freed_prefval),
                ),
            )

    metrics_to_display = [
        ("oversize_allocations", "Oversize allocations", _format_count),
        ("oversize_segments", "Oversize GCU segments", _format_count),
    ]

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)

        prefix = metric_key + "."

        current = stats[prefix + "current"]
        peak = stats[prefix + "peak"]
        allocated = stats[prefix + "allocated"]
        freed = stats[prefix + "freed"]

        lines.append(
            " {:<21} | {} | {} | {} | {} ".format(
                metric_name,
                formatter(current, current),
                formatter(peak, peak),
                formatter(allocated, allocated),
                formatter(freed, freed),
            ),
        )

    lines.append("=" * 75)

    fmt_dict = {"_": "", "device": device}
    for k, v in stats.items():
        fmt_dict[k.replace(".", "-")] = v
    return "|" + "|\n|".join(lines).format(**fmt_dict) + "|\n"


def mem_get_info(device: Union[Device, int] = None) -> Tuple[int, int]:
    r"""Returns the global free and total GCU memory for a given
    device using topsMemGetInfo.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.gcu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`gcu-memory-management` for more
        details about GCU memory management.
    """
    if device is None:
        device = torch.gcu.current_device()
    device = _get_device_index(device)
    return torch.gcu.gcurt().topsMemGetInfo(device)


def _record_memory_history_legacy(
    enabled: bool,
    record_context=True,
    trace_alloc_max_entries=1,
    trace_alloc_record_context=False,
    device: Union[Device, int] = None,
    record_context_cpp=False,
):
    _C._gcu_record_memory_history_legacy(
        enabled,
        record_context,
        record_context_cpp,
        trace_alloc_max_entries,
        trace_alloc_record_context,
    )


def _record_memory_history(enabled="all", *args, **kwargs):
    """
    Enables recording of stack traces associated with memory
    allocations, so you can tell what allocated any piece of memory in
    :func:`torch.gcu.memory._snapshot()`.

    In addition too keeping stack traces with each current allocation and free,
    this will also enable recording of a history of all alloc/free events.

    Use :func:`torch.gcu.memory._snapshot()` to retrieve this information,
    and the tools in `_memory_viz.py` to visualize snapshots.

    The Python trace collection is fast (2us per trace), so you may consider
    enabling this on production jobs if you anticipate ever having to debug
    memory issues.

    C++ trace collection is also fast (~50ns/frame), which for many typical programs
    works out to ~2us per trace, but can vary depending on stack depth.

    Args:
        enabled (Literal[None, "state", "all"], optional):
            `None`, disable recording memory history.
            `"state"`, keep information for currenly allocated memory.
            `"all"`, additionally keep a history of all alloc/free calls.
            Defaults to "all".
        context (Literal[None, "state", "alloc", "all"], optional):
            `None`, Do not record any tracebacks.
            `"state"`, Record tracebacks for currently allocated memory.
            `"alloc"`, additionally keep tracebacks for alloc calls.
            `"all"`, additionally keep tracebacks for free calls.
            Defaults to "all".
        stacks (Literal["python", "all"], optional):
            `"python"`, include Python, TorchScript, and inductor frames in tracebacks
            `"all"`, additionally include C++ frames
            Defaults to "all".
        max_entries (int, optional): Keep a maximum of `max_entries`
            alloc/free events in the recorded history recorded.
    """

    if isinstance(enabled, bool):
        return _record_memory_history_legacy(enabled, *args, **kwargs)
    else:
        return _record_memory_history_impl(enabled, *args, **kwargs)


def _record_memory_history_impl(
    enabled: Optional[str] = "all",
    context: Optional[str] = "all",
    stacks: str = "all",
    max_entries: int = sys.maxsize,
    device: Union[Device, int] = None,
):
    _C._gcu_record_memory_history(enabled, context, stacks, max_entries)


_record_memory_history.__signature__ = signature(_record_memory_history_impl)  # type: ignore[attr-defined]


def _snapshot(device: Union[Device, int] = None):
    """Saves a snapshot of GCU memory state at the time it was called.
    The state is represented as a dictionary with the following structure.

    .. code-block:: python

        class Snapshot(TypedDict):
            segments : List[Segment]
            device_traces: List[List[TraceEntry]]

        class Segment(TypedDict):
            # Segments are memory returned from a topsMalloc call.
            # The size of reserved memory is the sum of all Segments.
            # Segments are cached and reused for future allocations.
            # If the reuse is smaller than the segment, the segment
            # is split into more then one Block.
            # empty_cache() frees Segments that are entirely inactive.
            address: int
            total_size: int #  topsMalloc'd size of segment
            stream: int
            segment_type: Literal['small', 'large'] # 'large' (>1MB)
            allocated_size: int # size of memory in use
            active_size: int # size of memory in use or in active_awaiting_free state
            blocks : List[Block]

        class Block(TypedDict):
            # A piece of memory returned from the allocator, or
            # current cached but inactive.
            size: int
            requested_size: int # size requested during malloc, may be smaller than
                                # size due to rounding
            address: int
            state: Literal['active_allocated', # used by a tensor
                        'active_awaiting_free', # waiting for another stream to finish using
                                                # this, then it will become free
                        'inactive',] # free for reuse
            frames: List[Frame] # stack trace from where the allocation occurred

        class Frame(TypedDict):
                filename: str
                line: int
                name: str

        class TraceEntry(TypedDict):
            # When `torch.gcu.memory._record_memory_history()` is enabled,
            # the snapshot will contain TraceEntry objects that record each
            # action the allocator took.
            action: Literal[
            'alloc'  # memory allocated
            'free_requested', # the allocated received a call to free memory
            'free_completed', # the memory that was requested to be freed is now
                            # able to be used in future allocation calls
            'segment_alloc', # the caching allocator ask topsMalloc for more memory
                            # and added it as a segment in its cache
            'segment_free',  # the caching allocator called topsFree to return memory
                            # to gcu possibly trying free up memory to
                            # allocate more segments or because empty_caches was called
            'oom',          # the allocator threw an OOM exception. 'size' is
                            # the requested number of bytes that did not succeed
            'snapshot'      # the allocator generated a memory snapshot
                            # useful to coorelate a previously taken
                            # snapshot with this trace
            ]
            addr: int # not present for OOM
            frames: List[Frame]
            size: int
            stream: int
            device_free: int # only present for OOM, the amount of
                            # memory gcu still reports to be free

    Returns:
        The Snapshot dictionary object
    """
    return _C._gcu_memorySnapshot()


def _dump_snapshot(filename="dump_snapshot.pickle"):
    """
    Saves a pickled version of the `torch.memory._snapshot()` dictionary to a file.
    This file can be opened by the interactive snapshot viewer at pytorch.org/memory_viz

    Args:
        filename (str, optional): Name of the file to create. Defaults to "dump_snapshot.pickle".
    """
    s = _snapshot()
    with open(filename, "wb") as f:
        pickle.dump(s, f)


def _set_allocator_settings(env: str):
    return _C._gcu_gcuCachingAllocator_set_allocator_settings(env)


def get_allocator_backend() -> str:
    r"""Returns a string describing the active allocator backend as set by
    ``PYTORCH_GCU_ALLOC_CONF``. Currently available backends are
    ``native`` (PyTorch's native caching allocator) and `topsMallocAsync``
    (GCU's built-in asynchronous allocator).

    .. note::
        See :ref:`gcu-memory-management` for details on choosing the allocator backend.
    """
    return _C._gcu_getAllocatorBackend()


# TODO: Add GCUPluggableAllocator Implementation
def not_support_warning():
    warnings.warn("torch_gcu does NOT support GCUPluggableAllocator for now!")
    return None


class _GCUAllocator:
    r"""Wrapper over internal GCU memory allocators."""

    def __init__(self, allocator):
        # self._allocator = allocator
        not_support_warning()

    def allocator(self):
        # return self._allocator
        not_support_warning()
        return None


class GCUPluggableAllocator(_GCUAllocator):
    r"""GCU memory allocator loaded from a so file."""

    def __init__(self, path_to_so_file: str, alloc_fn_name: str,
                 free_fn_name: str):
        r"""Memory allocators are compiled in .so files and loaded dynamically using ctypes.

        To change the active allocator use the :func:`torch.memory.gcu.change_current_allocator` function.

        Args:
            path_to_so_file(str): Path in the filesystem to the `.so` file containing
                the allocator functions
            alloc_fn_name(str): Name of the function to perform the memory allocation
                in the so file. The signature must be:
                void* alloc_fn_name(ssize_t size, int device, topsStream_t stream);
            free_fn_name(str): Name of the function to perform the memory release
                in the so file. The signature must be:
                void free_fn_name(void* ptr, size_t size, topsStream_t stream);

        .. warning::
            This is currently supported only in unix OSs

        .. note::
            See :ref:`gcu-memory-management` for details on creating and using a custom allocator
        """
        not_support_warning()



def tops_malloc_host_accessible(shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    r"""Returns a tensor for cross-machine RDMA operations.

    Args:
        shape (list[int]): The shape of the tensor to allocate.
        dtype (torch.dtype): The data type of the tensor to allocate.
    """
    return _C._gcu_tops_malloc_host_accessible(shape, dtype)


class MemPool:
    r"""MemPool represents a pool of memory in a caching allocator.

    It holds a unique pool ID (a tuple of two ints) that the GCUCachingAllocator
    uses to route memory allocations into an isolated private pool.  This is the
    GCU counterpart of :class:`torch.cuda.MemPool`.

    Args:
        allocator: reserved for future use (custom pluggable allocator).
            Currently ignored.
        use_on_oom (bool): reserved for future use. Currently ignored.
    """

    _uid_counter = 0
    _uid_lock = threading.Lock()

    def __init__(
        self,
        allocator=None,
        use_on_oom: bool = False,
    ):
        with MemPool._uid_lock:
            MemPool._uid_counter += 1
            uid = MemPool._uid_counter
        self._id: Tuple[int, int] = (0, uid)
        self._allocator = allocator
        self._use_on_oom = use_on_oom
        self._use_count = 0

    @property
    def id(self) -> Tuple[int, int]:
        r"""Returns the ID of this pool as a tuple of two ints."""
        return self._id

    @property
    def allocator(self):
        r"""Returns the allocator this MemPool routes allocations to."""
        return self._allocator

    def use_count(self) -> int:
        r"""Returns the reference count of this pool."""
        return self._use_count

    def snapshot(self):
        r"""Return a snapshot of the GCU memory allocator pool state across all
        devices.

        Interpreting the output of this function requires familiarity with the
        memory allocator internals.

        .. note::
            See :ref:`gcu-memory-management` for more details about GPU memory
            management.
        """
        return memory_snapshot()


@contextlib.contextmanager
def use_mem_pool(pool: MemPool, device: "Device" = None):
    r"""A context manager that routes allocations to a given pool.

    Args:
        pool (torch.gcu.MemPool): a MemPool object to be made active so that
            allocations route to this pool.
        device (torch.device or int, optional): selected device. Uses MemPool on
            the current device, given by :func:`~torch.gcu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This context manager makes only the current thread's allocations route to
        the given pool. If a new thread is spawned inside the context manager
        (e.g. by calling backward) the allocations in that thread will not
        route to the given pool.
    """
    device_index = (
        torch.gcu.current_device() if device is None else _get_device_index(device)
    )
    pool._use_count += 1
    _C._gcu_beginAllocateCurrentThreadToPool(device_index, pool.id)
    try:
        yield
    finally:
        _C._gcu_endAllocateToPool(device_index, pool.id)
        _C._gcu_releasePool(device_index, pool.id)
        pool._use_count -= 1
