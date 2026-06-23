import ctypes

import torch_gcu
import torch_gcu._C
from torch._utils import _dummy_type

if not hasattr(torch_gcu._C, "_GcuStreamBase"):
    # Define dummy base classes
    torch_gcu._C.__dict__["_GcuStreamBase"] = _dummy_type("_GcuStreamBase")
    torch_gcu._C.__dict__["_GcuEventBase"] = _dummy_type("_GcuEventBase")


class Stream(torch_gcu._C._GcuStreamBase):
    r"""Wrapper around a GCU stream.

    A GCU stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream, which can be positive, 0, or negative.
            A lower number indicates a higher priority. By default, the priority is set to 0.
            If the value falls outside of the allowed priority range, it will automatically be
            mapped to the nearest valid priority (lowest for large positive numbers or
            highest for large negative numbers).
    """

    def __new__(cls, device=None, priority=0, **kwargs):
        # setting device manager is expensive, so we avoid it unless necessary
        if device is None or ("stream_id" in kwargs
                              and "device_index" in kwargs):
            return super().__new__(cls, priority=priority, **kwargs)
        else:
            with torch_gcu.gcu.device(device):
                return super().__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event):
        r"""Makes all future work submitted to the stream wait for an event.

        Args:
            event (torch_gcu.Event): an event to wait for.

        .. note:: This is a wrapper around ``gcuStreamWaitEvent()``

           This function returns without waiting for :attr:`event`: only future
           operations are affected.

        """
        event.wait(self)

    def wait_stream(self, stream):
        r"""Synchronizes with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels in :attr:`stream`: only future operations are affected.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        r"""Records an event.

        Args:
            event (torch_gcu.Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self):
        r"""Checks if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed."""
        return super().query()

    def synchronize(self):
        r"""Wait for all the kernels in this stream to complete.

        .. note:: This is a wrapper around ``gcuStreamSynchronize()``: see
           `GCU Stream documentation`_ for more info.
        """
        super().synchronize()

    def set_limit(self, cluster_num, sip_num):
        r"""Set cluster_num and sip_num for this stream, success returns True, failure returns False"""
        return super().set_limit(cluster_num=cluster_num, sip_num=sip_num)

    def get_limit(self):
        r"""Return cluster_num and sip_num for this stream, if it fails, return an False"""
        return super().get_limit()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.gcu_stream)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False

    def __hash__(self):
        return hash((self.gcu_stream, self.device))

    def __repr__(self):
        return f"<torch_gcu.Stream device={self.device} gcu_stream={self.gcu_stream:#x}>"


class ExternalStream(Stream):
    r"""Wrapper around an externally allocated GCU stream.

    This class is used to wrap streams allocated in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This class doesn't manage the stream life-cycle, it is the user
       responsibility to keep the referenced stream alive while this class is
       being used.

    Args:
        stream_ptr(int): Integer representation of the `gcuStream_t` value.
            allocated externally.
        device(torch.device or int, optional): the device where the stream
            was originally allocated. if device is specified incorrectly,
            subsequent launches using this stream may fail.
    """

    def __new__(cls, stream_ptr, device=None, **kwargs):
        with torch_gcu.gcu.device(device):
            return super().__new__(cls, stream_ptr=stream_ptr, **kwargs)


class Event(torch_gcu._C._GcuEventBase):
    r"""Wrapper around a GCU event.

    GCU events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize GCU
    streams.

    The underlying GCU events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)

    """

    def __new__(cls, enable_timing=False, blocking=False, interprocess=False):
        return super().__new__(
            cls,
            enable_timing=enable_timing,
            blocking=blocking,
            interprocess=interprocess,
        )

    @classmethod
    def from_ipc_handle(cls, device, handle):
        r"""Reconstruct an event from an IPC handle on the given device."""
        return super().from_ipc_handle(device, handle)

    def record(self, stream=None):
        r"""Records the event in a given stream.

        Uses ``torch_gcu.current_stream()`` if no stream is specified. The
        stream's device must match the event's device."""
        if stream is None:
            stream = torch_gcu.gcu.current_stream()
        super().record(stream)

    def wait(self, stream=None):
        r"""Makes all future work submitted to the given stream wait for this
        event.

        Use ``torch_gcu.current_stream()`` if no stream is specified.

        .. note:: This is a wrapper around ``gcuStreamWaitEvent()``
        """
        if stream is None:
            stream = torch_gcu.gcu.current_stream()
        super().wait(stream)

    def query(self):
        r"""Checks if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super().query()

    def elapsed_time(self, end_event):
        r"""Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        """
        return super().elapsed_time(end_event)

    def synchronize(self):
        r"""Waits for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

         .. note:: This is a wrapper around ``gcuEventSynchronize()``
        """
        super().synchronize()

    def ipc_handle(self):
        r"""Returns an IPC handle of this event. If not recorded yet, the event
        will use the current device."""
        return super().ipc_handle()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.gcu_event)

    def __repr__(self):
        if self.gcu_event:
            return f"<torch_gcu.Event {self._as_parameter_.value:#x}>"
        else:
            return "<torch_gcu.Event uninitialized>"
