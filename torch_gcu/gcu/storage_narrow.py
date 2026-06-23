from typing import (
    TypeVar,
    Optional,
    Union,
    Any,
    cast,
    Dict,
    Tuple,
    )
from typing_extensions import Self
import os
import sys
import io
import warnings
import struct
import collections
import difflib
import tarfile
from contextlib import closing

import torch
from torch.types import Storage
from torch._sources import get_source_lines_and_file
from torch._prims_common import DeviceLikeType
from torch.storage import (
    _StorageBase,
    UntypedStorage,
    TypedStorage,
    _warn_typed_storage_removal,
    _get_storage_from_sequence,
    _isint,
    )
from torch.serialization import (
    _serialization_tls,
    UNSAFE_MESSAGE,
    MAGIC_NUMBER,
    PROTOCOL_VERSION,
    LONG_SIZE,
    INT_SIZE,
    SHORT_SIZE,
    StorageType,
    SourceChangeWarning,
    LoadEndianness,
    mkdtemp,
    _maybe_decode_ascii,
    _check_seekable,
    _should_read_directly,
    _is_zipfile,
    get_default_load_endianness,
    _get_restore_location,
    normalize_storage_type,
    location_tag,
    )
from torch.utils.backend_registration import _normalization_device

from torch_gcu import _C


T = TypeVar('T', bound='Union[_StorageBase, TypedStorage]')

def _gcu_TypedStorage___init__(self, *args, device=None, dtype=None, wrap_storage=None, _internal=False):
    if not _internal:
        _warn_typed_storage_removal()
    arg_error_msg = (
        'TypedStorage.__init__ received an invalid combination '
        'of arguments. Expected one of:\n'
        ' * (*, torch.device device, torch.dtype dtype)\n'
        ' * (int size, *, torch.device device, torch.dtype dtype)\n'
        ' * (Sequence data, *, torch.device device, torch.dtype dtype)\n'
        ' * (*, UntypedStorage wrap_storage, torch.dtype dtype)')

    if wrap_storage is not None:
        if len(args) != 0:
            raise RuntimeError(
                arg_error_msg +
                "\nNo positional arguments should be given when using "
                "'wrap_storage'")

        if dtype is None:
            raise RuntimeError(
                arg_error_msg +
                "\nArgument 'dtype' must be specified")

        if not isinstance(dtype, torch.dtype):
            raise TypeError(
                arg_error_msg +
                f"\nArgument 'dtype' must be torch.dtype, not {type(dtype)}")

        if device is not None:
            raise RuntimeError(
                arg_error_msg +
                "\nArgument 'device' should not be specified when 'wrap_storage' is given")

        self.dtype = dtype

        if not isinstance(wrap_storage, torch.UntypedStorage):
            raise TypeError(
                arg_error_msg +
                f"\nArgument 'wrap_storage' must be UntypedStorage, but got {type(wrap_storage)}")

        self._untyped_storage = wrap_storage

    else:
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        device = torch.device('cpu' if device is None else device)

        if self.dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
            if device.type == 'cuda':
                raise RuntimeError("Cannot create CUDA storage with quantized dtype")

        if len(args) == 0:
            self._untyped_storage = torch.UntypedStorage(device=device)

        elif len(args) == 1:
            if _isint(args[0]):
                tmp_tensor = torch.empty(int(args[0]), dtype=self.dtype, device=device)
                self._untyped_storage = tmp_tensor.untyped_storage()
            elif isinstance(args[0], collections.abc.Sequence):
                self._untyped_storage = _get_storage_from_sequence(args[0], self.dtype, device)
            else:
                raise TypeError(
                    arg_error_msg +
                    f"\nArgument type not recognized: {type(args[0])}")

        else:
            raise RuntimeError(
                arg_error_msg +
                "\nToo many positional arguments")


# deliver dtype to determine whether narrowed or not
def _gcu__utils__to(self, device, non_blocking=False, dtype=None):
    """Returns a copy of this object in device memory.

    If this object is already on the correct device, then no copy is performed
    and the original object is returned.

    Args:
        device (int): The destination device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
    """
    if self.device == device:
        return self

    kwargs = {}
    if dtype:
        kwargs["src_dtype"] = kwargs["dst_dtype"] = dtype

    if device.type == "cpu":
        pin_memory = non_blocking and self.device.type in (
            "cuda",
            torch._C._get_privateuse1_backend_name(),
        )
        untyped_storage = torch.empty(
            self.nbytes(), dtype=torch.uint8, device=device, pin_memory=pin_memory
        ).untyped_storage()
        untyped_storage.copy_(self, non_blocking, **kwargs)
        return untyped_storage

    device_module = getattr(torch, device.type, None)
    assert (
        device_module is not None
    ), f"{device.type.upper()} device module is not loaded"
    with device_module.device(device):
        if self.is_sparse and hasattr(device_module, "sparse"):
            new_type = getattr(device_module.sparse, self.__class__.__name__)
            indices = getattr(torch.Tensor._indices(self), device.type)(
                device, non_blocking
            )
            values = getattr(torch.Tensor._values(self), device.type)(
                device, non_blocking
            )
            return new_type(indices, values, self.size())
        else:
            assert (
                not self.is_sparse
            ), f"sparse storage is not supported for {device.type.upper()} tensors"
            tensor_size = self.nbytes()
            tensor_dtype = torch.uint8
            if dtype:
                element_size = torch._utils._element_size(dtype)
                tensor_size = self.nbytes() // element_size
                tensor_dtype = dtype
            tmp_tensor = torch.empty(tensor_size, dtype=tensor_dtype, device=device)
            untyped_storage = tmp_tensor.untyped_storage()
            untyped_storage.copy_(self, non_blocking, **kwargs)
            return untyped_storage


# add dtype to deliver TypedStorage.dtype if the obj fetched by TypedStorage._untyped_storage
# to determined whether narrowed or not
def _gcu__StorageBase_to(self, *, device: DeviceLikeType, non_blocking: bool = False, dtype=None):
    if not isinstance(device, torch.device):
        device = torch.device(device)
    return _gcu__utils__to(self, device, non_blocking, dtype)


# deliver dtype to determine whether narrowed or not
def _gcu_TypedStorage_to(self, *, device: DeviceLikeType, non_blocking: bool = False) -> Self:
    _warn_typed_storage_removal()
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if self.dtype in [
        torch.quint8,
        torch.quint4x2,
        torch.quint2x4,
        torch.qint32,
        torch.qint8,
    ]:
        raise RuntimeError(
            f"Cannot create {device.type.upper()} storage with quantized dtype"
        )
    to_storage = self._untyped_storage.to(device=device, non_blocking=non_blocking, dtype=self.dtype)
    return self._new_wrapped_storage(to_storage)


# add dtype to deliver TypedStorage.dtype if the obj fetched by TypedStorage._untyped_storage
# to determine whether narrowed or not
def _gcu__StorageBase_tolist(self, dtype=None):
    """Return a list containing the elements of this storage."""
    return list(self.cpu(dtype))


# deliver dtype to determine whether narrowed or not
def _gcu_TypedStorage_copy_(self, source: T, non_blocking: Optional[bool] = None, dtype=None):
    _warn_typed_storage_removal()
    kwargs = {}
    kwargs["dst_dtype"] = self.dtype
    if isinstance(source, TypedStorage):
        kwargs["src_dtype"] = source.dtype
        self._untyped_storage.copy_(source._untyped_storage, non_blocking, **kwargs)
    else:
        if dtype:
            kwargs["src_dtype"] = dtype
        self._untyped_storage.copy_(source, non_blocking, **kwargs)
    return self


# deliver dtype to determine whether narrowed or not
def _gcu__StorageBase_cpu(self, dtype=None):
    """Return a CPU copy of this storage if it's not already on the CPU."""
    if self.device.type != "cpu":
        kwargs = {}
        if dtype:
            kwargs["src_dtype"] = kwargs["dst_dtype"] = dtype
        return torch.UntypedStorage(self.size()).copy_(self, False, **kwargs)
    return self


# deliver dtype to determine whether narrowed or not
def _gcu_TypedStorage_cpu(self):
    """Return a CPU copy of this storage if it's not already on the CPU."""
    _warn_typed_storage_removal()
    return self._new_wrapped_storage(self._untyped_storage.cpu(self.dtype))


# deliver dtype to determine whether narrowed or not
def _gcu_TypedStorage__write_file(self, *args, **kwargs):
    return self._untyped_storage._write_file(*args, self.dtype, **kwargs)


# deliver dtype to determine whether narrowed or not
def _gcu_TypedStorage__set_from_file(self, *args, **kwargs):
    return self._untyped_storage._set_from_file(*args, self.dtype, **kwargs)


# add dtype to deliver TypedStorage.dtype if the obj fetched by TypedStorage._untyped_storage
# to determine whether narrowed or not
def _gcu__StorageBase_wrap_storage_to(self, device=None, non_blocking=False, dtype=None):
    r"""Return a copy of this object in custom device memory.

    If this object is already in device memory and on the correct device, then
    no copy is performed and the original object is returned.

    Args:
        device (int): The destination device id. Defaults to the current device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
        the copy will be asynchronous with respect to the host. Otherwise,
        the argument has no effect.
    """
    # There should be a judgment related to storage device and a judgment related to storage type,
    # but it depends on the extended function, so this part is temporarily omitted in the automatic generation.
    device_idx = _normalization_device("gcu", device)

    if self.is_gcu:
        # storage has already on expected device.
        if self.get_device() == device_idx:
            return self
    # For sparse storage, custom need to extend the implementation by themselves.
    if self.is_sparse:
        raise RuntimeError(f"Can not support a sparse storage move to gcu backend")
    # create untyped_storage and copy data
    tensor_size = self.nbytes()
    tensor_dtype = torch.uint8
    kwargs = {}
    if dtype:
        element_size = torch._utils._element_size(dtype)
        tensor_size = self.nbytes() // element_size
        tensor_dtype = dtype
        kwargs["src_dtype"] = kwargs["dst_dtype"] = dtype
    tmp_tensor = torch.empty(tensor_size, dtype=tensor_dtype, device=torch.device(f'gcu:{device_idx}'))
    untyped_storage = tmp_tensor.untyped_storage()
    untyped_storage.copy_(self, non_blocking, **kwargs)
    return untyped_storage


# deliver dtype to determine whether narrowed or not
def _gcu_TypedStorage_wrap_typed_storage_to(self: torch.storage.TypedStorage,
                                            device=None, non_blocking=False,
                                            **kwargs) -> torch.storage.TypedStorage:
    torch.storage._warn_typed_storage_removal()
    gcu_storage = self._untyped_storage.gcu(device, non_blocking, self.dtype, **kwargs)
    return self._new_wrapped_storage(gcu_storage)


# deliver dtype to determine whether narrowed or not
def _gcu__legacy_save(obj, f, pickle_module, pickle_protocol) -> None:
    import torch.nn as nn

    serialized_container_types = {}
    serialized_storages: Dict[str, Tuple[torch.UntypedStorage, torch.dtype]] = {}

    # Since loading storages that view the same data with different dtypes is
    # not supported, we need to keep track of the dtype associated with each
    # storage data_ptr and throw an error if the dtype is ever different.
    # TODO: This feature could be added in the future
    storage_dtypes: Dict[int, torch.dtype] = {}

    def persistent_id(obj: Any) -> Optional[Tuple]:
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            if obj in serialized_container_types:
                return None
            serialized_container_types[obj] = True
            source_file = source = None
            try:
                source_lines, _, source_file = get_source_lines_and_file(obj)
                source = "".join(source_lines)
            except (
                Exception
            ):  # saving the source is optional, so we can ignore any errors
                warnings.warn(
                    "Couldn't retrieve source code for container of "
                    "type " + obj.__name__ + ". It won't be checked "
                    "for correctness upon loading."
                )
            return ("module", obj, source_file, source)

        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            storage: torch.UntypedStorage

            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                dtype = obj.dtype
                storage_numel = obj._size()

            elif isinstance(obj, torch.UntypedStorage):
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                dtype = torch.uint8
                storage_numel = storage.nbytes()
            else:
                raise TypeError(f"type not recognized: {type(obj)}")

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that "
                            "view the same data as different types"
                        )
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype

            view_metadata: Optional[Tuple[str, int, int]]

            # Offset is always 0, but we keep it for backwards compatibility
            # with the old serialization format (which supported storage views)
            offset = 0
            storage_key = str(storage._cdata)
            location = location_tag(storage)

            # TODO: There's an issue here with FC. It might be impossible to
            # solve, but it's worth noting. Imagine we save a list `[storage,
            # tensor]`, where `tensor.storage()` is the same as `storage`, and
            # `tensor.element_size() > 1`. Let's say that `tensor.dtype ==
            # torch.float`.  The storage will be serialized with element size
            # of 1, since we're choosing to serialize the first occurrence of
            # a duplicate storage. Since this legacy serialization format saves
            # the numel of the storage, rather than nbytes directly, we'll be
            # effectively saving nbytes in this case.  We'll be able to load it
            # and the tensor back up with no problems in _this_ and future
            # versions of pytorch, but in older versions, here's the problem:
            # the storage will be loaded up as a UntypedStorage, and then the
            # FloatTensor will loaded and the UntypedStorage will be assigned to
            # it. Since the storage dtype does not match the tensor dtype, this
            # will cause an error.  If we reverse the list, like `[tensor,
            # storage]`, then we will save the `tensor.storage()` as a faked
            # `FloatStorage`, and the saved size will be the correct
            # dtype-specific numel count that old versions expect. `tensor`
            # will be able to load up properly in old versions, pointing to
            # a FloatStorage. However, `storage` is still being translated to
            # a UntypedStorage, and it will try to resolve to the same
            # FloatStorage that `tensor` contains. This will also cause an
            # error. It doesn't seem like there's any way around this.
            # Probably, we just cannot maintain FC for the legacy format if the
            # saved list contains both a tensor and a storage that point to the
            # same data.  We should still be able to maintain FC for lists of
            # just tensors, as long as all views share the same dtype as the
            # tensor they are viewing.

            if storage_key not in serialized_storages:
                serialized_storages[storage_key] = (storage, dtype)
            is_view = storage._cdata != storage._cdata
            if is_view:
                view_metadata = (str(storage._cdata), offset, storage.nbytes())
            else:
                view_metadata = None

            res = (
                "storage",
                storage_type,
                storage_key,
                location,
                storage_numel,
                view_metadata,
            )
            return res
        return None

    sys_info = dict(
        protocol_version=PROTOCOL_VERSION,
        little_endian=sys.byteorder == "little",
        type_sizes=dict(
            short=SHORT_SIZE,
            int=INT_SIZE,
            long=LONG_SIZE,
        ),
    )

    pickle_module.dump(MAGIC_NUMBER, f, protocol=pickle_protocol)
    pickle_module.dump(PROTOCOL_VERSION, f, protocol=pickle_protocol)
    pickle_module.dump(sys_info, f, protocol=pickle_protocol)

    class PyTorchLegacyPickler(pickle_module.Pickler):
        def persistent_id(self, obj):
            return persistent_id(obj)

    pickler = PyTorchLegacyPickler(f, protocol=pickle_protocol)
    pickler.dump(obj)

    serialized_storage_keys = sorted(serialized_storages.keys())
    pickle_module.dump(serialized_storage_keys, f, protocol=pickle_protocol)
    f.flush()
    for key in serialized_storage_keys:
        storage, dtype = serialized_storages[key]
        storage._write_file(
            f, _should_read_directly(f), True, torch._utils._element_size(dtype), dtype
        )


# deliver dtype to determine whether narrowed or not
def _gcu__save(
    obj,
    zip_file,
    pickle_module,
    pickle_protocol,
    _disable_byteorder_record,
):
    serialized_storages = {}
    id_map: Dict[int, str] = {}

    # Since loading storages that view the same data with different dtypes is
    # not supported, we need to keep track of the dtype associated with each
    # storage data_ptr and throw an error if the dtype is ever different.
    # TODO: This feature could be added in the future
    storage_dtypes: Dict[int, torch.dtype] = {}

    def persistent_id(obj):
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()
                dtype = obj.dtype

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()
                dtype = torch.uint8

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if str(storage.device) != "meta" and storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that "
                            "view the same data as different types"
                        )
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = id_map.setdefault(storage._cdata, str(len(id_map)))
            if hasattr(obj, "_fake_device") and obj._fake_device is not None:
                location = str(obj._fake_device)
            else:
                location = location_tag(storage)
            serialized_storages[storage_key] = (storage, dtype)

            return ("storage", storage_type, storage_key, location, storage_numel)

        return None

    # Write the pickle data for `obj`
    data_buf = io.BytesIO()

    class PyTorchPickler(pickle_module.Pickler):  # type: ignore[name-defined]
        def persistent_id(self, obj):
            return persistent_id(obj)

    pickler = PyTorchPickler(data_buf, protocol=pickle_protocol)
    pickler.dump(obj)
    data_value = data_buf.getvalue()
    zip_file.write_record("data.pkl", data_value, len(data_value))

    # Write byte order marker
    if not _disable_byteorder_record:
        if sys.byteorder not in ["little", "big"]:
            raise ValueError("Unknown endianness type: " + sys.byteorder)

        zip_file.write_record("byteorder", sys.byteorder, len(sys.byteorder))

    # Write each tensor to a file named tensor/the_tensor_key in the zip archive
    for key in sorted(serialized_storages.keys()):
        name = f"data/{key}"
        storage, dtype = serialized_storages[key]
        num_bytes = storage.nbytes()
        global _serialization_tls
        if _serialization_tls.skip_data:
            zip_file.write_record_metadata(name, num_bytes)
        else:
            # given that we copy things around anyway, we might use storage.cpu()
            # this means to that to get tensors serialized, you need to implement
            # .cpu() on the underlying Storage
            if storage.device.type != "cpu":
                storage = storage.cpu(dtype)
            # Now that it is on the CPU we can directly copy it into the zip file
            zip_file.write_record(name, storage, num_bytes)


def _get_map_location_str(map_location):
    if isinstance(map_location, (str, bytes)):
        return map_location
    elif isinstance(map_location, torch.device):
        return map_location.type


#  modify order, map_location after construct TypedStorage
#  and deliver dtype to determine whether narrowed or not
def _gcu__legacy_load(f, map_location, pickle_module, **pickle_load_args):
    deserialized_objects: Dict[int, Any] = {}

    restore_location = _get_restore_location(map_location)
    map_location_str = _get_map_location_str(map_location)

    class UnpicklerWrapper(pickle_module.Unpickler):  # type: ignore[name-defined]
        def find_class(self, mod_name, name):
            if type(name) is str and "Storage" in name:
                try:
                    return StorageType(name)
                except KeyError:
                    pass
            return super().find_class(mod_name, name)

    def _check_container_source(container_type, source_file, original_source):
        try:
            current_source = "".join(get_source_lines_and_file(container_type)[0])
        except Exception:  # saving the source is optional, so we can ignore any errors
            warnings.warn(
                "Couldn't retrieve source code for container of "
                "type " + container_type.__name__ + ". It won't be checked "
                "for correctness upon loading."
            )
            return
        if original_source != current_source:
            if container_type.dump_patches:
                file_name = container_type.__name__ + ".patch"
                diff = difflib.unified_diff(
                    current_source.split("\n"),
                    original_source.split("\n"),
                    source_file,
                    source_file,
                    lineterm="",
                )
                lines = "\n".join(diff)
                try:
                    with open(file_name, "a+") as f:
                        file_size = f.seek(0, 2)
                        f.seek(0)
                        if file_size == 0:
                            f.write(lines)
                        elif file_size != len(lines) or f.read() != lines:
                            raise OSError
                    msg = (
                        "Saved a reverse patch to " + file_name + ". "
                        "Run `patch -p0 < " + file_name + "` to revert your "
                        "changes."
                    )
                except OSError:
                    msg = (
                        "Tried to save a patch, but couldn't create a "
                        "writable file " + file_name + ". Make sure it "
                        "doesn't exist and your working directory is "
                        "writable."
                    )
            else:
                msg = (
                    "you can retrieve the original source code by "
                    "accessing the object's source attribute or set "
                    "`torch.nn.Module.dump_patches = True` and use the "
                    "patch tool to revert the changes."
                )
            msg = f"source code of class '{torch.typename(container_type)}' has changed. {msg}"
            warnings.warn(msg, SourceChangeWarning)

    def legacy_load(f):
        deserialized_objects: Dict[int, Any] = {}

        def persistent_load(saved_id):
            if isinstance(saved_id, tuple):
                # Ignore containers that don't have any sources saved
                if all(saved_id[1:]):
                    _check_container_source(*saved_id)
                return saved_id[0]
            return deserialized_objects[int(saved_id)]

        with closing(
            tarfile.open(fileobj=f, mode="r:", format=tarfile.PAX_FORMAT)
        ) as tar, mkdtemp() as tmpdir:
            if pickle_module is torch._weights_only_unpickler:
                raise RuntimeError(
                    "Cannot use ``weights_only=True`` with files saved in the "
                    "legacy .tar format. " + UNSAFE_MESSAGE
                )
            tar.extract("storages", path=tmpdir)
            with open(os.path.join(tmpdir, "storages"), "rb", 0) as f:
                num_storages = pickle_module.load(f, **pickle_load_args)
                for _ in range(num_storages):
                    args = pickle_module.load(f, **pickle_load_args)
                    key, location, storage_type = args
                    dtype = storage_type._dtype
                    obj = cast(Storage, torch.UntypedStorage)._new_with_file(
                        f, torch._utils._element_size(dtype)
                    )

                    # TODO: Once we decide to break serialization FC, we can
                    # stop wrapping with TypedStorage

                    if (map_location_str and 'gcu' in map_location_str) or (map_location_str is None and 'gcu' in location):
                        deserialized_objects[key] = restore_location(
                            torch.storage.TypedStorage(wrap_storage=obj, dtype=dtype, _internal=True), location)
                    else:
                        obj = restore_location(obj, location)
                        deserialized_objects[key] = torch.storage.TypedStorage(
                            wrap_storage=obj, dtype=dtype, _internal=True
                        )

                storage_views = pickle_module.load(f, **pickle_load_args)
                for target_cdata, root_cdata, offset, numel in storage_views:
                    root = deserialized_objects[root_cdata]
                    element_size = torch._utils._element_size(root.dtype)
                    offset_bytes = offset * element_size
                    # TODO: Once we decide to break serialization FC, we can
                    # stop wrapping with TypedStorage
                    deserialized_objects[target_cdata] = torch.storage.TypedStorage(
                        wrap_storage=root._untyped_storage[
                            offset_bytes : offset_bytes + numel * element_size
                        ],
                        dtype=root.dtype,
                        _internal=True,
                    )

            tar.extract("tensors", path=tmpdir)
            with open(os.path.join(tmpdir, "tensors"), "rb", 0) as f:
                num_tensors = pickle_module.load(f, **pickle_load_args)
                for _ in range(num_tensors):
                    args = pickle_module.load(f, **pickle_load_args)
                    key, storage_id, _original_tensor_type = args
                    storage = deserialized_objects[storage_id]
                    (ndim,) = struct.unpack("<i", f.read(4))
                    # skip next 4 bytes; legacy encoding treated ndim as 8 bytes
                    f.read(4)
                    numel = struct.unpack(f"<{ndim}q", f.read(8 * ndim))
                    stride = struct.unpack(f"<{ndim}q", f.read(8 * ndim))
                    (storage_offset,) = struct.unpack("<q", f.read(8))
                    tensor = torch.empty((0,), dtype=storage.dtype).set_(
                        storage._untyped_storage, storage_offset, numel, stride
                    )
                    deserialized_objects[key] = tensor

            pickle_file = tar.extractfile("pickle")
            unpickler = UnpicklerWrapper(pickle_file, **pickle_load_args)
            unpickler.persistent_load = persistent_load
            result = unpickler.load()
            return result

    deserialized_objects = {}

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        if typename == "module":
            # Ignore containers that don't have any sources saved
            if all(data[1:]):
                _check_container_source(*data)
            return data[0]
        elif typename == "storage":
            storage_type, root_key, location, numel, view_metadata = data
            location = _maybe_decode_ascii(location)
            dtype = storage_type.dtype

            nbytes = numel * torch._utils._element_size(dtype)

            if root_key not in deserialized_objects:
                if torch._guards.active_fake_mode() is not None:
                    obj = cast(Storage, torch.UntypedStorage(nbytes, device="meta"))
                else:
                    obj = cast(Storage, torch.UntypedStorage(nbytes))
                    obj._torch_load_uninitialized = True

                # TODO: Once we decide to break serialization FC, we can
                # stop wrapping with TypedStorage
                if (map_location_str and 'gcu' in map_location_str) or (map_location_str is None and 'gcu' in location):
                    typed_storage = restore_location(
                        torch.storage.TypedStorage(wrap_storage=obj, dtype=dtype, _internal=True), location)
                else:
                    obj = restore_location(obj, location)
                    typed_storage = torch.storage.TypedStorage(
                        wrap_storage=obj, dtype=dtype, _internal=True
                    )

                deserialized_objects[root_key] = typed_storage
            else:
                typed_storage = deserialized_objects[root_key]
                if typed_storage._data_ptr() == 0:
                    typed_storage = torch.storage.TypedStorage(
                        device=typed_storage._untyped_storage.device,
                        dtype=dtype,
                        _internal=True,
                    )

            if view_metadata is not None:
                view_key, offset, view_size = view_metadata
                offset_bytes = offset * torch._utils._element_size(dtype)
                view_size_bytes = view_size * torch._utils._element_size(dtype)
                if view_key not in deserialized_objects:
                    # TODO: Once we decide to break serialization FC, we can
                    # stop wrapping with TypedStorage
                    deserialized_objects[view_key] = torch.storage.TypedStorage(
                        wrap_storage=typed_storage._untyped_storage[
                            offset_bytes : offset_bytes + view_size_bytes
                        ],
                        dtype=dtype,
                        _internal=True,
                    )
                res = deserialized_objects[view_key]

            else:
                res = typed_storage
            return res
        else:
            raise RuntimeError(f"Unknown saved id type: {saved_id[0]}")

    _check_seekable(f)
    f_should_read_directly = _should_read_directly(f)

    if f_should_read_directly and f.tell() == 0:
        # legacy_load requires that f has fileno()
        # only if offset is zero we can attempt the legacy tar file loader
        try:
            return legacy_load(f)
        except tarfile.TarError:
            if _is_zipfile(f):
                # .zip is used for torch.jit.save and will throw an un-pickling error here
                raise RuntimeError(
                    f"{f.name} is a zip archive (did you mean to use torch.jit.load()?)"
                ) from None
            # if not a tarfile, reset file offset and proceed
            f.seek(0)

    if not hasattr(f, "readinto") and (3, 8, 0) <= sys.version_info < (3, 8, 2):
        raise RuntimeError(
            "torch.load does not work with file-like objects that do not implement readinto on Python 3.8.0 and 3.8.1. "
            f'Received object of type "{type(f)}". Please update to Python 3.8.2 or newer to restore this '
            "functionality."
        )

    magic_number = pickle_module.load(f, **pickle_load_args)
    if magic_number != MAGIC_NUMBER:
        raise RuntimeError("Invalid magic number; corrupt file?")
    protocol_version = pickle_module.load(f, **pickle_load_args)
    if protocol_version != PROTOCOL_VERSION:
        raise RuntimeError(f"Invalid protocol version: {protocol_version}")

    _sys_info = pickle_module.load(f, **pickle_load_args)
    unpickler = UnpicklerWrapper(f, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()

    deserialized_storage_keys = pickle_module.load(f, **pickle_load_args)

    if torch._guards.active_fake_mode() is None:
        offset = f.tell() if f_should_read_directly else None
        for key in deserialized_storage_keys:
            assert key in deserialized_objects
            typed_storage = deserialized_objects[key]
            typed_storage._untyped_storage._set_from_file(
                f,
                offset,
                f_should_read_directly,
                torch._utils._element_size(typed_storage.dtype),
                typed_storage.dtype,
            )
            if offset is not None:
                offset = f.tell()

    torch._utils._validate_loaded_sparse_tensors()

    return result


#  modify order, map_location after construct TypedStorage
def _gcu__load(
    zip_file,
    map_location,
    pickle_module,
    pickle_file="data.pkl",
    overall_storage=None,
    **pickle_load_args,
):
    restore_location = _get_restore_location(map_location)
    map_location_str = _get_map_location_str(map_location)

    loaded_storages = {}

    # check if byteswapping is needed
    byteordername = "byteorder"
    byteorderdata = None
    if zip_file.has_record(byteordername):
        byteorderdata = zip_file.get_record(byteordername)
        if byteorderdata not in [b"little", b"big"]:
            raise ValueError("Unknown endianness type: " + byteorderdata.decode())
    elif (
        get_default_load_endianness() == LoadEndianness.LITTLE
        or get_default_load_endianness() is None
    ):
        byteorderdata = b"little"
    elif get_default_load_endianness() == LoadEndianness.BIG:
        byteorderdata = b"big"
    elif get_default_load_endianness() == LoadEndianness.NATIVE:
        pass
    else:
        raise ValueError("Invalid load endianness type")

    if (
        not zip_file.has_record(byteordername)
        and get_default_load_endianness() is None
        and sys.byteorder == "big"
    ):
        # Default behaviour was changed
        # See https://github.com/pytorch/pytorch/issues/101688
        warnings.warn(
            "The default load endianness for checkpoints without a byteorder mark "
            "on big endian machines was changed from 'native' to 'little' endian, "
            "to avoid this behavior please use "
            "torch.serialization.set_default_load_endianness to set "
            "the desired default load endianness",
            UserWarning,
        )

    def load_tensor(dtype, numel, key, location):
        name = f"data/{key}"
        if torch._guards.detect_fake_mode(None) is not None:
            nbytes = numel * torch._utils._element_size(dtype)
            storage = torch.UntypedStorage(nbytes, device="meta")
        elif overall_storage is not None:
            storage_offset = zip_file.get_record_offset(name)
            storage = overall_storage[storage_offset : storage_offset + numel]
        else:
            storage = (
                zip_file.get_storage_from_record(name, numel, torch.UntypedStorage)
                ._typed_storage()
                ._untyped_storage
            )
        # swap here if byteswapping is needed
        if byteorderdata is not None:
            if byteorderdata.decode() != sys.byteorder:
                storage.byteswap(dtype)

        # TODO: Once we decide to break serialization FC, we can
        # stop wrapping with TypedStorage
        if (map_location_str and 'gcu' in map_location_str) or (map_location_str is None and 'gcu' in location):
            typed_storage = restore_location(torch.storage.TypedStorage(
                wrap_storage=storage, dtype=dtype, _internal=True), location)
        else:
            typed_storage = torch.storage.TypedStorage(
                wrap_storage=restore_location(storage, location),
                dtype=dtype,
                _internal=True,
            )

        if typed_storage._data_ptr() != 0:
            loaded_storages[key] = typed_storage

        return typed_storage

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        assert (
            typename == "storage"
        ), f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        storage_type, key, location, numel = data
        if storage_type is torch.UntypedStorage:
            dtype = torch.uint8
        else:
            dtype = storage_type.dtype

        if key in loaded_storages:
            typed_storage = loaded_storages[key]
        else:
            nbytes = numel * torch._utils._element_size(dtype)
            typed_storage = load_tensor(
                dtype, nbytes, key, _maybe_decode_ascii(location)
            )

        return typed_storage

    load_module_mapping: Dict[str, str] = {
        # See https://github.com/pytorch/pytorch/pull/51633
        "torch.tensor": "torch._tensor"
    }

    # Need to subclass Unpickler instead of directly monkey-patching the find_class method
    # because it's marked readonly in pickle.
    # The type: ignore is because mypy can't statically determine the type of this class.
    class UnpicklerWrapper(pickle_module.Unpickler):  # type: ignore[name-defined]
        # from https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path/13405732
        # Lets us override the imports that pickle uses when unpickling an object.
        # This is useful for maintaining BC if we change a module path that tensor instantiation relies on.
        def find_class(self, mod_name, name):
            if type(name) is str and "Storage" in name:
                try:
                    return StorageType(name)
                except KeyError:
                    pass
            mod_name = load_module_mapping.get(mod_name, mod_name)
            return super().find_class(mod_name, name)

    # Load the data (which may in turn use `persistent_load` to load tensors)
    data_file = io.BytesIO(zip_file.get_record(pickle_file))

    unpickler = UnpicklerWrapper(data_file, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    # Needed for tensors where storage device and rebuild tensor device are
    # not connected (wrapper subclasses and tensors rebuilt using numpy)
    global _serialization_tls
    _serialization_tls.map_location = map_location
    result = unpickler.load()
    _serialization_tls.map_location = None

    torch._utils._validate_loaded_sparse_tensors()
    torch._C._log_api_usage_metadata(
        "torch.load.metadata", {"serialization_id": zip_file.serialization_id()}
    )
    return result


def apply_narrow():

    _C._replace_StorageBase_methods()

    torch._utils._to = _gcu__utils__to

    _StorageBase.cpu = _gcu__StorageBase_cpu
    _StorageBase.to = _gcu__StorageBase_to
    _StorageBase.tolist = _gcu__StorageBase_tolist
    _StorageBase.gcu = _gcu__StorageBase_wrap_storage_to

    TypedStorage.__init__ = _gcu_TypedStorage___init__
    TypedStorage.copy_ = _gcu_TypedStorage_copy_
    TypedStorage.cpu = _gcu_TypedStorage_cpu
    TypedStorage.to = _gcu_TypedStorage_to
    TypedStorage._write_file = _gcu_TypedStorage__write_file
    TypedStorage._set_from_file = _gcu_TypedStorage__set_from_file
    TypedStorage.gcu = _gcu_TypedStorage_wrap_typed_storage_to

    torch.serialization._legacy_save = _gcu__legacy_save
    torch.serialization._save = _gcu__save
    torch.serialization._legacy_load = _gcu__legacy_load
    torch.serialization._load = _gcu__load
