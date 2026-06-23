#
# Copyright 2019-2023 Enflame. All Rights Reserved.
#
import importlib
import os
import threading
import traceback
import warnings
from functools import lru_cache
from typing import Any, Callable, cast, Optional, Union
import torch
import torch_gcu
import torch_gcu._C
from torch import device as _device
from torch import Tensor
from torch._utils import _dummy_type, _LazySeedTracker, classproperty
from torch.utils.backend_registration import _check_register_once
from torch.types import Device

# from . import gds
from ._utils import _get_device_index
from .graphs import (
    GCUGraph,
    graph,
    graph_pool_handle,
    is_current_stream_capturing,
    make_graphed_callables,
)
from .streams import Event, ExternalStream, Stream

from .graphs import (
    GCUGraph,
    graph,
    graph_pool_handle,
    is_current_stream_capturing,
    make_graphed_callables,
)
from .streams import Event, ExternalStream, Stream

# patches
from .reductions import apply_reductions_patch
from .storage import apply_storage_patch
from .storage_narrow import apply_narrow

try:
    from torch_gcu._C import _gcurt  # type: ignore[attr-defined]
except ImportError:
    _gcurt = None

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls: list[tuple[Callable[[], None], list[str]]] = (
    []
)  # don't invoke these until initialization occurs
_is_in_bad_fork = getattr(torch_gcu._C, "_gcu_isInBadFork", lambda: False)
_device_t = Union[_device, str, int, None]

_HAS_PYEFML = False
_PYEFML_ERR = None
try:
    try:
        import pyefml

        _HAS_PYEFML = True
    except ModuleNotFoundError:
        pass
except ImportError as err:
    _PYEFML_ERR = err

_lazy_seed_tracker = _LazySeedTracker()

# Define dummy _GcuDeviceProperties type if PyTorch was compiled without GCU
if hasattr(torch_gcu._C, "_GcuDeviceProperties"):
    _GcuDeviceProperties = torch_gcu._C._GcuDeviceProperties
else:
    _GcuDeviceProperties = _dummy_type(
        "_GcuDeviceProperties"
    )  # type: ignore[assignment, misc]

_exchange_device = torch_gcu._C._gcu_exchangeDevice
_maybe_exchange_device = torch_gcu._C._gcu_maybeExchangeDevice

has_half: bool = True

default_generators: tuple[torch._C.Generator] = ()  # type: ignore[assignment]


def _is_compiled() -> bool:
    r"""Returns true if compile with GCU support."""
    return hasattr(torch_gcu._C, "_gcu_getDeviceCount")


def _efml_based_avail() -> bool:
    return os.getenv("PYTORCH_EFML_BASED_GCU_CHECK") == "1"


def is_available() -> bool:
    r"""Return a bool indicating if GCU is currently available."""
    if not _is_compiled():
        return False
    if _efml_based_avail():
        # The user has set an env variable to request this availability check that attempts to avoid fork poisoning by
        # using EFML at the cost of a weaker GCU availability assessment. Note that if EFML discovery/initialization
        # fails, this assessment falls back to the default GCU Runtime API assessment (`topsGetDeviceCount`)
        return device_count() > 0
    else:
        # The default availability inspection never throws and returns 0 if the driver is missing or can't
        # be initialized. This uses the GCU Runtime API `topsGetDeviceCount` which in turn initializes the GCU Driver
        # API via `topsInit`
        return torch_gcu._C._gcu_getDeviceCount() > 0


def is_bf16_supported():
    if not is_available():
        return False
    return torch_gcu._C._gcu_is_bf16_supported()


def is_initialized():
    r"""Returns whether PyTorch's GCU state has been initialized."""
    return _initialized and not _is_in_bad_fork()


def _lazy_call(callable, **kwargs):
    if is_initialized():
        callable()
    else:
        # TODO(torch_deploy): this accesses linecache, which attempts to read the
        # file system to get traceback info. Patch linecache or do something
        # else here if this ends up being important.
        global _lazy_seed_tracker
        if kwargs.get("seed_all", False):
            _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())
        elif kwargs.get("seed", False):
            _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())
        else:
            # Don't store the actual traceback to avoid memory cycle
            _queued_calls.append((callable, traceback.format_stack()))


class DeferredGcuCallError(Exception):
    pass


def init():
    r"""Initialize PyTorch's GCU state.

    You may need to call this explicitly if you are interacting with
    PyTorch via its C API, as Python bindings for GCU functionality
    will not be available until this initialization takes place.
    Ordinary users should not need this, as all of PyTorch's GCU methods
    automatically initialize GCU state on-demand.

    Does nothing if the GCU state is already initialized.
    """
    _lazy_init()


def _lazy_init():
    global _initialized, _queued_calls
    if is_initialized() or hasattr(_tls, "is_initializing"):
        return
    with _initialization_lock:
        # We be double-checked locking, boys!  This is OK because
        # the above test was GIL protected anyway.  The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if is_initialized():
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if _is_in_bad_fork():
            raise RuntimeError(
                "Cannot re-initialize GCU in forked subprocess. To use GCU with "
                "multiprocessing, you must use the 'spawn' start method"
            )
        if not hasattr(torch_gcu._C, "_gcu_getDeviceCount"):
            raise AssertionError("Torch not compiled with GCU enabled")
        if _gcurt is None:
            raise AssertionError(
                "libgcurt functions unavailable. It looks like you have a broken build?"
            )
        # This function throws if there's a driver initialization error, no GCUs
        # are found or any other error occurs
        if "GCU_MODULE_LOADING" not in os.environ:
            os.environ["GCU_MODULE_LOADING"] = "LAZY"
        torch_gcu._C._gcu_init()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True

        for calls in _lazy_seed_tracker.get_calls():
            if calls:
                _queued_calls.append(calls)

        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = (
                        f"GCU call failed lazily at initialization with error: {str(e)}\n\n"
                        f"GCU call was originally invoked at:\n\n{''.join(orig_traceback)}"
                    )
                    raise DeferredGCUCallError(msg) from e
        finally:
            delattr(_tls, "is_initializing")
        _initialized = True

        if torch_gcu.is_transfer_to_gcu:
            setattr(torch.cuda, "default_generators", torch_gcu.gcu.default_generators)


def gcurt():
    _lazy_init()
    return _gcurt


class gcuStatus:
    SUCCESS: int = 0
    ERROR_NOT_READY: int = 34


class GcuError(RuntimeError):

    def __init__(self, code: int) -> None:
        msg = _gcurt.topsGetErrorString(_gcurt.topsError(code))
        super().__init__(f"{msg} ({code})")


def check_error(res: int) -> None:
    if res != _gcurt.topsError.success:
        raise GcuError(res)


class _DeviceGuard:

    def __init__(self, index: int):
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch.gcu._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = torch.gcu._maybe_exchange_device(self.prev_idx)
        return False


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = _exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = _maybe_exchange_device(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GCU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        dev = obj.device
        if dev.type == "gcu":
            idx = dev
        else:
            idx = -1
        super(device_of, self).__init__(idx)


def set_device(device: _device_t) -> None:
    r"""Sets the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``TOPS_VISIBLE_DEVICES`` environmental variable.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_device_index(device)
    if device >= 0:
        torch_gcu._C._gcu_setDevice(device)


def get_device_name(device: Optional[_device_t] = None) -> str:
    r"""Gets the name of a device.

    Args:
        device (torch.device or int, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch_gcu.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
    return get_device_properties(device).name


def get_device_capability(device: Optional[_device_t] = None) -> tuple[int, int]:
    r"""Gets the gcu capability of a device.

    Args:
        device (torch.device or int, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~torch.gcu.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor gcu capability of the device
    """
    prop = get_device_properties(device)
    return prop.major, prop.minor


def get_device_properties(device: _device_t) -> _GcuDeviceProperties:
    r"""Gets the properties of a device.

    Args:
        device (torch.device or int or str): device for which to return the
            properties of the device.

    Returns:
        _GcuDeviceProperties: the properties of the device
    """
    _lazy_init()  # will define _get_device_properties
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return _get_device_properties(device)  # type: ignore[name-defined]


def get_sip_count(device: _device_t = None) -> int:
    r"""Gets the SIP (Streaming Instruction Processor) count for a device.

    Args:
        device (torch.device or int or str, optional): device for which to return
            the SIP count. Defaults to current device if ``None``.

    Returns:
        int: The number of SIPs on the device.
    """
    _lazy_init()  # will define _get_sip_count in torch.gcu
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    # Access the C++ bound function through torch.gcu module
    import torch
    return torch.gcu._get_sip_count(device)  # type: ignore[attr-defined]


def can_device_access_peer(device: _device_t, peer_device: _device_t) -> bool:
    r"""Checks if peer access between two devices is possible."""
    _lazy_init()
    device = _get_device_index(device, optional=True)
    peer_device = _get_device_index(peer_device)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    if peer_device < 0 or peer_device >= device_count():
        raise AssertionError("Invalid peer device id")
    return torch_gcu._C._gcu_canDeviceAccessPeer(device, peer_device)


class StreamContext:
    r"""Context-manager that selects a given stream.

    All GCU kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """

    cur_stream: Optional["torch_gcu.gcu.Stream"]

    def __init__(self, stream: Optional["torch_gcu.gcu.Stream"]):
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                self.idx = -1

        self.src_prev_stream = (
            None if not torch.jit.is_scripting() else torch_gcu.default_stream(None)
        )
        self.dst_prev_stream = (
            None if not torch.jit.is_scripting() else torch_gcu.default_stream(None)
        )

    def __enter__(self):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or GCU device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = current_stream(None)

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = current_stream(cur_stream.device)
        set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no GCU device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        set_stream(self.src_prev_stream)  # type: ignore[arg-type]


def stream(stream: Optional["torch_gcu.gcu.Stream"]) -> StreamContext:
    r"""Wrapper around the Context-manager StreamContext that
    selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    ..Note:: In eager mode stream is of type Stream class while in JIT it is
    an object of the custom class ``torch.classes.gcu.Stream``.
    """
    return StreamContext(stream)


def _set_stream_by_id(stream_id, device_index, device_type):
    r"""set stream specified by the stream id, device index and
        device type

    Args: stream_id (int): stream id in stream pool
          device_index (int): device index in topo
          device_type (int): enum device type
    """
    torch_gcu._C._gcu_setStream(
        stream_id=stream_id,
        device_index=device_index,
        device_type=device_type,
    )


def set_stream(stream: Stream):
    r"""Sets the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    torch_gcu._C._gcu_setStream(
        stream_id=stream.stream_id,
        device_index=stream.device_index,
        device_type=stream.device_type,
    )


def _parse_visible_devices() -> Union[list[int], list[str]]:
    r"""Parse TOPS_VISIBLE_DEVICES environment variable."""
    var = os.getenv("TOPS_VISIBLE_DEVICES")

    if var is None:
        return list(range(64))

    def _strtoul(s: str) -> int:
        """Return -1 or positive integer sequence string starts with."""
        if not s:
            return -1
        for idx, c in enumerate(s):
            if not (c.isdigit() or (idx == 0 and c in "+-")):
                break
            if idx + 1 == len(s):
                idx += 1
        return int(s[:idx]) if idx > 0 else -1

    def parse_list_with_prefix(lst: str, prefix: str) -> list[str]:
        rcs: list[str] = []
        for elem in lst.split(","):
            # Repeated id results in empty set
            if elem in rcs:
                return cast(list[str], [])
            # Anything other but prefix is ignored
            if not elem.startswith(prefix):
                break
            rcs.append(elem)
        return rcs

    if var.startswith("GCU-"):
        return parse_list_with_prefix(var, "GCU-")
    if var.startswith("MIG-"):
        return parse_list_with_prefix(var, "MIG-")
    # TOPS_VISIBLE_DEVICES uses something like strtoul
    # which makes `1gpu2,2ampere` is equivalent to `1,2`
    rc: list[int] = []
    for elem in var.split(","):
        x = _strtoul(elem.strip())
        # Repeated ordinal results in empty set
        if x in rc:
            return cast(list[int], [])
        # Negative value aborts the sequence
        if x < 0:
            break
        rc.append(x)
    return rc


def _raw_device_count_efml() -> int:
    r"""Return number of devices as reported by EFML or negative value if EFML discovery/initialization failed."""
    from ctypes import byref, c_int, CDLL

    efml_h = CDLL("libefml.so.1")
    rc = efml_h.efmlInit()
    if rc != 0:
        warnings.warn("Can't initialize EFML")
        return -1
    dev_count = c_int(-1)
    rc = efml_h.efmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Can't get efml device count")
        return -1
    del efml_h
    return dev_count.value


def _raw_device_uuid_efml() -> Optional[list[str]]:
    r"""Return list of device UUID as reported by EFML or None if EVM discovery/initialization failed."""
    from ctypes import byref, c_int, c_void_p, CDLL, create_string_buffer

    efml_h = CDLL("libefml.so.1")
    rc = efml_h.efmlInit()
    if rc != 0:
        warnings.warn("Can't initialize EFML")
        return None
    dev_count = c_int(-1)
    rc = efml_h.efmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Can't get efml device count")
        return None
    uuids: list[str] = []
    for idx in range(dev_count.value):
        dev_id = c_void_p()
        rc = efml_h.efmlDeviceGetHandleByIndex_v2(idx, byref(dev_id))
        if rc != 0:
            warnings.warn("Can't get device handle")
            return None
        buf_len = 96
        buf = create_string_buffer(buf_len)
        rc = efml_h.efmlDeviceGetUUID(dev_id, buf, buf_len)
        if rc != 0:
            warnings.warn("Can't get device UUID")
            return None
        uuids.append(buf.raw.decode("ascii").strip("\0"))
    del efml_h
    return uuids


def _transform_uuid_to_ordinals(candidates: list[str], uuids: list[str]) -> list[int]:
    r"""Given the set of partial uuids and list of known uuids builds a set of ordinals excluding ambiguous partials IDs."""

    def uuid_to_ordinal(candidate: str, uuids: list[str]) -> int:
        best_match = -1
        for idx, uuid in enumerate(uuids):
            if not uuid.startswith(candidate):
                continue
            # Ambiguous candidate
            if best_match != -1:
                return -1
            best_match = idx
        return best_match

    rc: list[int] = []
    for candidate in candidates:
        idx = uuid_to_ordinal(candidate, uuids)
        # First invalid ordinal stops parsing
        if idx < 0:
            break
        # Duplicates result in empty set
        if idx in rc:
            return cast(list[int], [])
        rc.append(idx)
    return rc


def _device_count_efml() -> int:
    r"""Return number of devices as reported by EFML taking TOPS_VISIBLE_DEVICES into account.

    Negative value is returned if EFML discovery or initialization has failed.
    """
    visible_devices = _parse_visible_devices()
    if not visible_devices:
        return 0
    try:
        if type(visible_devices[0]) is str:
            # Skip MIG parsing
            if visible_devices[0].startswith("MIG-"):
                return -1
            uuids = _raw_device_uuid_efml()
            if uuids is None:
                return -1
            visible_devices = _transform_uuid_to_ordinals(
                cast(list[str], visible_devices), uuids
            )
        else:
            raw_cnt = _raw_device_count_efml()
            if raw_cnt <= 0:
                return raw_cnt
            # Trim the list up to a maximum available device
            for idx, val in enumerate(visible_devices):
                if cast(int, val) >= raw_cnt:
                    return idx
    except OSError:
        return -1
    except AttributeError:
        return -1
    return len(visible_devices)


def _get_efml_device_index(device: Optional[Union[int, Device]]) -> int:
    r"""Return the EFML index of the device, taking TOPS_VISIBLE_DEVICES into account."""
    idx = _get_device_index(device, optional=True)
    visible_devices = _parse_visible_devices()
    if type(visible_devices[0]) is str:
        uuids = _raw_device_uuid_efml()
        if uuids is None:
            raise RuntimeError("Can't get device UUIDs")
        visible_devices = _transform_uuid_to_ordinals(
            cast(list[str], visible_devices), uuids
        )
    visible_devices = cast(list[int], visible_devices)
    if idx < 0 or idx >= len(visible_devices):
        raise RuntimeError(
            f"device {idx} is not visible (TOPS_VISIBLE_DEVICES={visible_devices})"
        )
    return visible_devices[idx]


_cached_device_count: Optional[int] = None


def device_count() -> int:
    r"""Return the number of GPUs available."""
    global _cached_device_count
    if not _is_compiled():
        return 0
    if _cached_device_count is not None:
        return _cached_device_count
    efml_count = _device_count_efml()
    r = torch_gcu._C._gcu_getDeviceCount() if efml_count < 0 else efml_count
    # NB: Do not cache the device count prior to GCU initialization, because
    # the number of devices can change due to changes to TOPS_VISIBLE_DEVICES
    # setting prior to GCU initialization.
    if _initialized:
        _cached_device_count = r
    return r


def get_arch_list() -> list[str]:
    r"""Returns list GCU architectures this library was compiled for."""
    if not torch_gcu.gcu.is_available():
        return []
    arch_flags = torch_gcu._C._gcu_getArchFlags()
    if arch_flags is None:
        return []
    return arch_flags.split()


def get_gencode_flags() -> str:
    r"""Returns gencode flags this library was compiled with."""
    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return ""
    arch_list_ = [arch.split("_") for arch in arch_list]
    return " ".join(
        [
            f"-gencode compute=compute_{arch},code={kind}_{arch}"
            for (kind, arch) in arch_list_
        ]
    )


def current_device() -> int:
    r"""Returns the index of a currently selected device."""
    torch_gcu.gcu._lazy_init()
    return torch_gcu._C._gcu_getDevice()


def synchronize(device: _device_t = None) -> None:
    r"""Waits for all kernels in all streams on a GCU device to complete.

    Args:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`torch_gcu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    torch_gcu.gcu._lazy_init()
    with torch.gcu.device(device):
        return torch_gcu._C._gcu_synchronize()


def current_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Returns the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch_gcu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    streamdata = torch_gcu._C._gcu_getCurrentStream(
        _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def default_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Returns the default :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch_gcu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    streamdata = torch_gcu._C._gcu_getDefaultStream(
        _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def set_sync_debug_mode(debug_mode: Union[int, str]) -> None:
    r"""Sets the debug mode for gcu synchronizing operations.

    Args:
        debug_mode(str or int): if "default" or 0, don't error or warn on synchronizing operations,
            if "warn" or 1, warn on synchronizing operations, if "error" or 2, error out synchronizing operations.

    Warning:
        This is an experimental feature, and not all synchronizing operations will trigger warning or error. In
        particular, operations in torch.distributed and torch.sparse namespaces are not covered yet.
    """

    torch_gcu.gcu._lazy_init()
    if isinstance(debug_mode, str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, `warn`, `error`"
            )

    torch_gcu._C._gcu_set_sync_debug_mode(debug_mode)


def get_sync_debug_mode() -> int:
    r"""Returns current value of debug mode for gcu synchronizing operations."""

    torch_gcu.gcu._lazy_init()
    return torch_gcu._C._gcu_get_sync_debug_mode()


def _get_pyefml_handler(device: Optional[Union[Device, int]] = None):
    if not _HAS_PYEFML:
        raise ModuleNotFoundError(
            "pyefml does not seem to be installed or it can't be imported."
        ) from _PYEFML_ERR

    try:
        pyefml.efmlInit()
    except OSError as e:
        raise RuntimeError("gcu driver can't be loaded, is gcu enabled?") from e
    except Exception as e:
        raise RuntimeError("efml initialization failed") from e

    device = _get_efml_device_index(device)
    handle = pyefml.efmlDeviceGetHandleByIndex(device)
    return handle


def device_memory_used(device: Optional[Union[Device, int]] = None) -> int:
    r"""Return used global (device) memory in bytes as given by `efml`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_gcu.gcu.current_device`,
            if :attr:`device` is ``None`` (default).

    """
    handle = _get_pyefml_handler(device)
    return pyefml.efmlDeviceGetMemoryInfo(handle).used


def _get_device(device: Union[int, str, torch.device]) -> torch.device:
    r"""Return the torch.device type object from the passed in device.

    Args:
        device (torch.device or int): selected device.
    """
    if isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, int):
        return torch.device("gcu", device)
    return device


def _get_generator(device: torch.device) -> torch._C.Generator:
    r"""Return the GCU Generator object for the given device.

    Args:
        device (torch.device): selected device.
    """

    idx = device.index
    if idx is None:
        idx = current_device()
    return torch.gcu.default_generators[idx]


def _set_rng_state_offset(
    offset: int, device: Union[int, str, torch.device] = "gcu"
) -> None:
    r"""Sets the random number generator state offset of the specified GCU.

    Args:
        offset (int): The desired offset
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'gcu'`` (i.e., ``torch.device('gcu')``, the current GCU device).
    """
    final_device = _get_device(device)

    def cb():
        default_generator = _get_generator(final_device)
        default_generator.set_offset(offset)

    _lazy_call(cb)


def _get_rng_state_offset(device: Union[int, str, torch.device] = "gcu") -> int:
    r"""Returns the random number generator state offset of the specified GCU.

    Args:
        device (torch.device or int, optional): The device to return the RNG state offset of.
            Default: ``'gcu'`` (i.e., ``torch.device('gcu')``, the current GCU device).

    .. warning::
        This function eagerly initializes GCU.
    """
    _lazy_init()
    final_device = _get_device(device)
    default_generator = _get_generator(final_device)
    return default_generator.get_offset()


from .memory import *  # noqa: F403
from .random import *  # noqa: F403

from torch.storage import _LegacyStorage, _warn_typed_storage_removal


class _GcuLegacyStorage(_LegacyStorage):

    @classmethod
    def from_buffer(cls, *args, **kwargs):
        _warn_typed_storage_removal()
        raise RuntimeError("from_buffer: Not available for GCU storage")

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs):
        raise RuntimeError("_new_with_weak_ptr: Not available for GCU storage")

    @classmethod
    def _new_shared_filename(cls, manager, obj, size, *, device=None, dtype=None):
        raise RuntimeError("_new_shared_filename: Not available for GCU storage")


class ByteStorage(_GcuLegacyStorage):

    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.uint8


class DoubleStorage(_GcuLegacyStorage):

    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.double


class FloatStorage(_GcuLegacyStorage):

    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.float


class HalfStorage(_GcuLegacyStorage):

    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.half


class LongStorage(_GcuLegacyStorage):

    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.long


class IntStorage(_GcuLegacyStorage):

    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int


class ShortStorage(_GcuLegacyStorage):

    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.short


class CharStorage(_GcuLegacyStorage):

    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int8


class BoolStorage(_GcuLegacyStorage):

    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bool


class BFloat16Storage(_GcuLegacyStorage):

    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bfloat16


del _LegacyStorage
del _GcuLegacyStorage

torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(BFloat16Storage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(BoolStorage)


def is_fp16_supported():
    # always return True, because all gcu versions support
    return True


def is_support_inf_nan():
    return torch_gcu._C._gcu_is_support_inf_nan()


from torch.storage import _LegacyStorage, _warn_typed_storage_removal
from torch._utils import classproperty


class ParallelCompute:
    r"""Context-manager that set current stream cluster_num and sip_num,
        eccl stream will use the remaining SIP.

    Args:
        cluster_num:
        sip_num:

    Example:
        with torch.gcu.ParallelCompute(2,10):
            # async_op must set True
            work = torch.distributed.all_reduce(out_tensor, async_op=True)
            output = torch.matmul(a, b)
            work.wait()
    """

    def __init__(self, cluster_num=2, sip_num=10):
        assert cluster_num == 2, "cluster_num must be 2"
        assert sip_num >= 1 or sip_num < 12, "sip_num must be 1 to 11"
        self.cluster_num = cluster_num
        self.sip_num = sip_num

    def __enter__(self):
        if get_device_capability()[0] != 3:
            return
        cur_stream = current_stream()
        if cur_stream.get_limit():
            self.prev_cluster_num, self.prev_sip_num = cur_stream.get_limit()
        else:
            self.prev_cluster_num, self.prev_sip_num = 2, 12
        cur_stream.set_limit(self.cluster_num, self.sip_num)
        os.environ["TORCH_GCU_ECCL_SIP_NUM"] = str(12 - self.sip_num)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        if get_device_capability()[0] != 3:
            return
        cur_stream = current_stream()
        cur_stream.set_limit(self.prev_cluster_num, self.prev_sip_num)
        del os.environ["TORCH_GCU_ECCL_SIP_NUM"]


def gcu_device(device: _device_t = None) -> torch.device:
    r"""Returns torch.device reference to a GCU device

    Args:
        device (torch.device or int): device must has type gcu if it is a torch.device.
    """
    device_id = _get_device_index(device, optional=True)
    return torch.device("gcu:{}".format(device_id))


def get_gcu_data_ptr(arg: torch.Tensor):
    return torch_gcu._C._get_gcu_data_ptr(arg)


def get_gcu_device_ptr(arg: torch.Tensor):
    return torch_gcu._C._get_gcu_device_ptr(arg)

def copy(src: Tensor, dst: Tensor, non_blocking: bool = False) -> Tensor:
    return torch_gcu._C._copy(src, dst, non_blocking)

def generate_methods_for_tensor(attr_name, attr):
    _check_register_once(torch.Tensor, attr_name)
    setattr(torch.Tensor, attr_name, attr)


from . import amp, topstx, tunable, efficient
from .debug import *
from .autocast_utils import *

__all__ = [
    # Typed storage and tensors
    "BFloat16Storage",
    "BFloat16Tensor",
    "BoolStorage",
    "BoolTensor",
    "ByteStorage",
    "ByteTensor",
    "CharStorage",
    "CharTensor",
    # "ComplexDoubleStorage",
    # "ComplexFloatStorage",
    "DoubleStorage",
    "DoubleTensor",
    "FloatStorage",
    "FloatTensor",
    "HalfStorage",
    "HalfTensor",
    "IntStorage",
    "IntTensor",
    "LongStorage",
    "LongTensor",
    "ShortStorage",
    "ShortTensor",
    "GCUGraph",
    "GcuError",
    "DeferredGcuCallError",
    "Event",
    "ExternalStream",
    "Stream",
    "StreamContext",
    "amp",
    # "caching_allocator_alloc",
    # "caching_allocator_delete",
    # "caching_allocator_enable",
    "can_device_access_peer",
    "check_error",
    "gcuStatus",
    "gcurt",
    # "current_blas_handle",
    "current_device",
    "current_stream",
    "default_generators",
    "default_stream",
    "device",
    "device_count",
    "device_memory_used",
    "device_of",
    "empty_cache",
    "get_allocator_backend",
    # "GCUPluggableAllocator",
    # "change_current_allocator",
    "get_arch_list",
    "get_device_capability",
    "get_device_name",
    "get_device_properties",
    "get_gencode_flags",
    # "get_per_process_memory_fraction",
    "get_rng_state",
    "get_rng_state_all",
    # "get_stream_from_external",
    "get_sync_debug_mode",
    "graph",
    "graph_pool_handle",
    "graphs",
    "has_half",
    # "has_magma",
    # "host_memory_stats",
    # "host_memory_stats_as_nested_dict",
    "init",
    "initial_seed",
    # "ipc_collect",
    "is_available",
    "is_bf16_supported",
    "is_current_stream_capturing",
    "is_initialized",
    # "is_tf32_supported",
    # "jiterator",
    # "list_gcu_processes",
    "make_graphed_callables",
    "manual_seed",
    "manual_seed_all",
    "max_memory_allocated",
    "max_memory_cached",
    "max_memory_reserved",
    "mem_get_info",
    "memory",
    "memory_allocated",
    "memory_cached",
    "memory_reserved",
    "memory_snapshot",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "memory_summary",
    # "memory_usage",
    # "MemPool",
    # "MemPoolContext",
    # "use_mem_pool",
    # "temperature",
    # "power_draw",
    # "clock_rate",
    # "eccl",
    "topstx",
    # "profiler",
    "random",
    # "reset_accumulated_host_memory_stats",
    "reset_accumulated_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    # "reset_peak_host_memory_stats",
    "reset_peak_memory_stats",
    "seed",
    "seed_all",
    "set_device",
    "set_per_process_memory_fraction",
    "set_rng_state",
    "set_rng_state_all",
    "set_stream",
    "set_sync_debug_mode",
    # "sparse",
    "stream",
    "streams",
    "synchronize",
    "tunable",
    # "utilization",
    # GCU specific list
    "set_autocast_enabled",
    "get_autocast_dtype",
    "set_autocast_dtype",
    "is_autocast_enabled",
    "efficient",
    "is_support_inf_nan",
    "is_fp16_supported",
    "gcu_device",
    "get_gcu_data_ptr",
    "get_gcu_device_ptr",
    "copy",
    # internal methods, for transfer_to_gcu
    "_get_device_index",
    "_is_compiled",
    "_efml_based_avail",
    "_lazy_init",
    "_lazy_call",
    "_tls",
    "_initialization_lock",
    "_queued_calls",
    "_is_in_bad_fork",
    "_device_t",
    "_HAS_PYEFML",
    "_PYEFML_ERR",
    "_lazy_seed_tracker",
    "_GcuDeviceProperties",
    "_exchange_device",
    "_maybe_exchange_device",
    "_parse_visible_devices",
    "_raw_device_count_efml",
    "_raw_device_uuid_efml",
    "_transform_uuid_to_ordinals",
    "_device_count_efml",
    "_get_efml_device_index",
    "_cached_device_count",
    "_get_pyefml_handler",
    "_get_device",
    "_get_generator",
    "_set_rng_state_offset",
    "_get_rng_state_offset",
    "_set_stream_by_id",
    "generate_methods_for_tensor",
]
