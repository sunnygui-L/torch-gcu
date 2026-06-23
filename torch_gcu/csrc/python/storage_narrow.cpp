/*
 * Copyright 2024 Enflame. All Rights Reserved.
 */

#include "python/storage_narrow.h"

#include <c10/core/CPUAllocator.h>
#include <c10/util/error.h>
#include <torch/csrc/Storage.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/serialization.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#ifdef _MSC_VER
#define LSEEK _lseeki64
#else
#define LSEEK lseek
#endif

namespace torch_gcu {

static bool isUnsupportedOperation() {
  THPObjectPtr io(PyImport_ImportModule("io"));
  if (!io) throw python_error();
  THPObjectPtr exception(PyObject_GetAttrString(io, "UnsupportedOperation"));
  if (!exception) throw python_error();
  return PyErr_ExceptionMatches(exception.get());
}

// Call Python fildes.read(nbytes) and copy it to buf.
static Py_ssize_t doPartialPythonReadBuffered(PyObject* fildes, void* buf,
                                              size_t raw_nbytes) {
  // If we request a large amount of data, f.read() will internally try to
  // allocate a buffer of that size.  This is counterproductive, because
  // it's not the buffer we ultimately want to write the data into.  Read
  // less than that and avoid allocating too much extra memory.
  // TODO: Maybe 260 KB is a bit small...
  const size_t nbytes =
      std::min<size_t>(raw_nbytes, 262144u);  // 2^18 (~260 KB)

  THPObjectPtr r(PyObject_CallMethod(fildes, "read", "i", nbytes));
  if (!r) throw python_error();

  auto size = PyBytes_GET_SIZE(r.get());
  const void* py_buf = PyBytes_AsString(r.get());

  // we read EOF
  if (size == 0) {
    return 0;
  }

  // Slurp it into the buffer we actually want
  memcpy(buf, py_buf, size);

  return size;
}

// Either does fildes.readinto(buf) or fildes.write(buf)
static Py_ssize_t doPartialPythonIO(PyObject* fildes, void* buf, size_t nbytes,
                                    bool is_read) {
  auto rw_flag = is_read ? PyBUF_WRITE : PyBUF_READ;
  THPObjectPtr memview(PyMemoryView_FromMemory(
      reinterpret_cast<char*>(buf), static_cast<Py_ssize_t>(nbytes), rw_flag));
  if (!memview) throw python_error();

  std::string method = "write";
  if (is_read) {
    method = "readinto";
  }
  THPObjectPtr r(
      PyObject_CallMethod(fildes, method.c_str(), "O", memview.get()));
  if (r) {
    return PyLong_AsSsize_t(r.get());
  }

  // fildes.readinto can return UnsupportedOperation so fall back to
  // fildes.read.
  if (is_read && isUnsupportedOperation()) {
    PyErr_Clear();
    return doPartialPythonReadBuffered(fildes, buf, nbytes);
  }
  throw python_error();
}

// Call Python fildes.readinto(buf)
static Py_ssize_t doPartialPythonReadInto(PyObject* fildes, void* buf,
                                          size_t nbytes) {
  return doPartialPythonIO(fildes, buf, nbytes, /* is_read */ true);
}

template <class io>
Py_ssize_t doPartialRead(io fildes, void* buf, size_t nbytes);

template <>
Py_ssize_t doPartialRead<int>(int fildes, void* buf, size_t nbytes) {
  return read(fildes, buf, nbytes);
}

template <>
Py_ssize_t doPartialRead<PyObject*>(PyObject* fildes, void* buf,
                                    size_t nbytes) {
  // Try to use fildes.readinto() instead of fildes.read()
  // because it is more memory efficient.
  // TODO: Stop calling PyObject_HasAttrString() in a loop on our read loop
  auto has_readinto = PyObject_HasAttrString(fildes, "readinto") == 1;
  if (has_readinto) {
    return doPartialPythonReadInto(fildes, buf, nbytes);
  }
  return doPartialPythonReadBuffered(fildes, buf, nbytes);
}

// Requires that we read EXACTLY nbytes; fails if we don't.
template <typename io>
void doRead(io fildes, void* raw_buf, size_t nbytes) {
  char* buf = static_cast<char*>(raw_buf);
  while (nbytes > 0) {
    errno = 0;  // doPartialRead may not set errno
    // we read in 1GB blocks to avoid bugs on Mac OS X Lion
    // see https://github.com/pytorch/pytorch/issues/1031 for more details
    Py_ssize_t r =
        doPartialRead(fildes, buf, std::min<size_t>(nbytes, 1073741824));
    if (r < 0) {
      int err = errno;
      TORCH_INTERNAL_ASSERT(err != 0,
                            "read(): impossible! r < 0, but no errno was set");
      TORCH_INTERNAL_ASSERT(err != EAGAIN, "read(): non-blocking fd ", fildes,
                            " read EAGAIN; cowardly refusing to spin-wait");
      if (err == EINTR) {
        continue;
      } else {
        TORCH_CHECK(false, "read(): fd ", fildes, " failed with ",
                    c10::utils::str_error(err));
      }
    } else if (r == 0) {
      break;
    }
    buf += r;
    // This is guaranteed by POSIX, but I just want to be double-sure
    // to not underflow a signed integer.
    AT_ASSERT(static_cast<size_t>(r) <= nbytes);
    nbytes -= r;
  }
  if (nbytes != 0) {
    TORCH_CHECK(false, "unexpected EOF, expected ", nbytes,
                " more bytes. The file might be corrupted.");
  }
}

// Call Python fildes.write(buf)
static Py_ssize_t doPartialPythonWrite(PyObject* fildes, void* buf,
                                       size_t nbytes) {
  return doPartialPythonIO(fildes, buf, nbytes, /* is_read */ false);
}

template <class io>
Py_ssize_t doPartialWrite(io fildes, void* buf, size_t nbytes);

template <>
Py_ssize_t doPartialWrite<int>(int fildes, void* buf, size_t nbytes) {
  return write(fildes, buf, nbytes);
}

template <>
Py_ssize_t doPartialWrite<PyObject*>(PyObject* fildes, void* buf,
                                     size_t nbytes) {
  return doPartialPythonWrite(fildes, buf, nbytes);
}

template <typename io>
void doWrite(io fildes, void* raw_buf, size_t nbytes) {
  char* buf = static_cast<char*>(raw_buf);
  while (nbytes > 0) {
    errno = 0;  // doPartialWrite may not set errno
    // we write in 1GB blocks to avoid bugs on Mac OS X Lion
    // see https://github.com/pytorch/pytorch/issues/1031 for more details
    Py_ssize_t r =
        doPartialWrite(fildes, buf, std::min<size_t>(nbytes, 1073741824));
    if (r < 0) {
      int err = errno;
      TORCH_INTERNAL_ASSERT(err != 0,
                            "write(): impossible! r < 0, but no errno was set");
      TORCH_INTERNAL_ASSERT(err != EAGAIN, "write(): non-blocking fd ", fildes,
                            " read EAGAIN; cowardly refusing to spin-wait");
      if (err == EINTR) {
        continue;
      } else {
        TORCH_CHECK(false, "write(): fd ", fildes, " failed with ",
                    c10::utils::str_error(err));
      }
    }
    buf += r;
    AT_ASSERT(static_cast<size_t>(r) <= nbytes);
    nbytes -= r;
  }
}

static PyObject* THGPStorage_copy_(PyObject* self, PyObject* args,
                                   PyObject* kwargs) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);

  at::Storage self_ = torch::createStorage(self);

  // deliver dtype to determine whether narrow or not
  at::ScalarType src_dtype = at::ScalarType::Byte;
  at::ScalarType dst_dtype = at::ScalarType::Byte;
  if (kwargs) {
    PyObject* src_dtype_obj = PyDict_GetItemString(kwargs, "src_dtype");
    PyObject* dst_dtype_obj = PyDict_GetItemString(kwargs, "dst_dtype");

    if (src_dtype_obj && dst_dtype_obj) {
      src_dtype = reinterpret_cast<THPDtype*>(src_dtype_obj)->scalar_type;
      dst_dtype = reinterpret_cast<THPDtype*>(dst_dtype_obj)->scalar_type;
    }

    if (src_dtype_obj) {
      PyDict_DelItemString(kwargs, "src_dtype");
    }
    if (dst_dtype_obj) {
      PyDict_DelItemString(kwargs, "dst_dtype");
    }
  }

  static torch::PythonArgParser parser({
      "copy_(Storage src, bool? non_blocking=None)",
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  at::Storage src = r.storage(0);
  bool non_blocking = r.toBoolOptional(1).value_or(false);

  // See Note [Invalid Python Storages]
  auto invalid = src.data() == nullptr &&
                 src.device_type() != c10::DeviceType::Meta &&
                 src.sym_nbytes() != 0;
  TORCH_CHECK(!invalid,
              "Attempted to call copy_() on an invalid python storage.")

  TORCH_CHECK(self_.nbytes() == src.nbytes(), "size does not match, self was ",
              self_.nbytes(), " bytes but src was ", src.nbytes(), " bytes");

  auto dst_options =
      c10::TensorOptions().device(self_.device()).dtype(dst_dtype);
  auto dst_t = at::empty({0}, dst_options).set_(self_);

  auto src_options = c10::TensorOptions().device(src.device()).dtype(src_dtype);
  auto src_t = at::empty({0}, src_options).set_(src);
  dst_t.copy_(src_t, non_blocking);

  Py_INCREF(self);
  return self;

  END_HANDLE_TH_ERRORS
}

// deliver dtype to determine whether narrow or not
template <class io>
void THGPStorage_writeFileRaw(c10::StorageImpl* self, io fd, bool save_size,
                              uint64_t element_size, at::ScalarType dtype) {
  c10::DeviceGuard guard(self->device());
  uint8_t* data{};
  at::Tensor cpu_tensor;
  size_t size_bytes = self->nbytes();
  size_t numel = size_bytes / element_size;
  if (self->device_type() == at::kCPU) {
    // We are using a mutable pointer here because we're ultimately
    // calling into a Python API that requires that, even though it
    // won't mutate the data.
    data = static_cast<uint8_t*>(self->mutable_data());
  } else {
    // Here we use a tensor.to() to impl D2H for all non-CPU device.
    size_t device_tensor_size = size_bytes;
    at::ScalarType device_tensor_dtype = at::ScalarType::Byte;
    if (dtype != at::ScalarType::Undefined) {
      device_tensor_size = numel;
      device_tensor_dtype = dtype;
    }
    auto device_tensor = at::from_blob(
        self->mutable_data(), {static_cast<int64_t>(device_tensor_size)}, {1},
        nullptr, at::device(self->device()).dtype(device_tensor_dtype),
        {self->device()});

    cpu_tensor = device_tensor.to(at::kCPU);
    data = (uint8_t*)cpu_tensor.data_ptr();
  }
  if (save_size) {
    if (torch::utils::THP_nativeByteOrder() ==
        torch::utils::THPByteOrder::THP_LITTLE_ENDIAN)
      torch_gcu::doWrite(fd, &numel, sizeof(int64_t));
    else {
      int64_t nsize{};  // convert big endian cpu to little endian storage
      torch::utils::THP_encodeBuffer(
          (uint8_t*)&nsize, (const int64_t*)&numel,
          torch::utils::THPByteOrder::THP_LITTLE_ENDIAN, 1);
      torch_gcu::doWrite(fd, &nsize, sizeof(int64_t));
    }
  }
  // fast track for bytes and little endian
  if (element_size == 1 || torch::utils::THP_nativeByteOrder() ==
                               torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
    torch_gcu::doWrite(fd, data, size_bytes);
  } else {
    size_t buffer_size = std::min(numel, (size_t)5000);
    std::vector<uint8_t> le_buffer;
    le_buffer.resize(buffer_size * element_size);
    for (size_t i = 0; i < numel; i += buffer_size) {
      size_t to_convert = std::min(numel - i, buffer_size);
      if (element_size == 2) {
        torch::utils::THP_encodeBuffer(
            le_buffer.data(), (const int16_t*)data + i,
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN, to_convert);
      } else if (element_size == 4) {
        torch::utils::THP_encodeBuffer(
            le_buffer.data(), (const int32_t*)data + i,
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN, to_convert);
      } else if (element_size == 8) {
        torch::utils::THP_encodeBuffer(
            le_buffer.data(), (const int64_t*)data + i,
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN, to_convert);
      }
      torch_gcu::doWrite(fd, le_buffer.data(), to_convert * element_size);
    }
  }
}

static PyObject* THGPStorage_writeFile(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  const auto& storage = THPStorage_Unpack(self);
  // See Note [Invalid Python Storages]

  auto invalid = storage.data() == nullptr &&
                 storage.device_type() != c10::DeviceType::Meta &&
                 storage.sym_nbytes() != 0;
  TORCH_CHECK(!invalid,
              "Attempted to call _write_file() on an invalid python storage.")

  PyObject* file = PyTuple_GetItem(args, 0);
  bool is_real_file = PyTuple_GetItem(args, 1) == Py_True;
  bool save_size = PyTuple_GetItem(args, 2) == Py_True;
  PyObject* element_size_obj = PyTuple_GET_ITEM(args, 3);

  TORCH_CHECK(element_size_obj != Py_None,
              "_write_file: need to specify element size");
  uint64_t element_size = THPUtils_unpackUInt64(element_size_obj);

  // deliver dtype to determine whether narrow or not
  at::ScalarType scalar_type = at::ScalarType::Undefined;
  if (PyTuple_Size(args) == 5) {
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 4);
    scalar_type = reinterpret_cast<THPDtype*>(dtype_obj)->scalar_type;
  }

  if (!is_real_file) {
    THGPStorage_writeFileRaw<PyObject*>(storage.unsafeGetStorageImpl(), file,
                                        save_size, element_size, scalar_type);
    Py_RETURN_NONE;
  }

  int fd = PyObject_AsFileDescriptor(file);
  TORCH_CHECK(fd != -1,
              "_write_file couldn't retrieve a file descriptor "
              "from given object");
  THGPStorage_writeFileRaw(storage.unsafeGetStorageImpl(), fd, save_size,
                           element_size, scalar_type);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// deliver dtype to determine whether narrow or not
template <class io>
c10::intrusive_ptr<c10::StorageImpl> THGPStorage_readFileRaw(
    io file, c10::intrusive_ptr<c10::StorageImpl> storage,
    uint64_t element_size, at::ScalarType dtype) {
  c10::OptionalDeviceGuard guard;
  if (storage.defined()) {
    guard.reset_device(storage->device());
  }
  int64_t size{};
  torch_gcu::doRead(file, &size, sizeof(int64_t));
  if (torch::utils::THP_nativeByteOrder() ==
      torch::utils::THPByteOrder::THP_BIG_ENDIAN) {
    int64_t tsize = size;  // convert little endian storage to big endian cpu
    torch::utils::THP_decodeBuffer(&size, (const uint8_t*)&tsize, true, 1);
  }
  size_t nbytes = element_size * size;
  if (!storage.defined()) {
    storage = c10::make_intrusive<at::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(), nbytes,
        c10::GetDefaultCPUAllocator(),
        /*resizable=*/true);
  } else {
    size_t _storage_nbytes = storage->nbytes();
    TORCH_CHECK(_storage_nbytes == nbytes,
                "storage has wrong byte size: expected %ld got %ld", nbytes,
                _storage_nbytes);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  std::unique_ptr<char[]> cpu_data;

  uint8_t* data{};
  if (storage->device_type() == at::kCPU) {
    data = static_cast<uint8_t*>(storage->mutable_data());
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    cpu_data = std::unique_ptr<char[]>(new char[nbytes]);
    data = (uint8_t*)cpu_data.get();
  }

  // fast track for bytes and little endian
  if (element_size == 1 || torch::utils::THP_nativeByteOrder() ==
                               torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
    torch_gcu::doRead(file, data, storage->nbytes());
  } else {
    int64_t buffer_size = std::min(size, (int64_t)5000);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    std::unique_ptr<uint8_t[]> le_buffer(
        new uint8_t[buffer_size * element_size]);

    for (int64_t i = 0; i < size; i += buffer_size) {
      size_t to_convert = std::min(size - i, buffer_size);
      torch_gcu::doRead(file, le_buffer.get(), element_size * to_convert);

      // NOLINTNEXTLINE(bugprone-branch-clone)
      if (element_size == 2) {
        torch::utils::THP_decodeBuffer((int16_t*)data + i, le_buffer.get(),
                                       true, to_convert);
      } else if (element_size == 4) {
        torch::utils::THP_decodeBuffer((int32_t*)data + i, le_buffer.get(),
                                       true, to_convert);
      } else if (element_size == 8) {
        torch::utils::THP_decodeBuffer((int64_t*)data + i, le_buffer.get(),
                                       true, to_convert);
      }
    }
  }

  if (storage->device_type() != at::kCPU) {
    // Here we use a tensor.copy_() to impl H2D for all non-CPU device.
    size_t tensor_size = nbytes;
    at::ScalarType tensor_dtype = at::ScalarType::Byte;
    if (dtype != at::ScalarType::Undefined) {
      tensor_size = size;
      tensor_dtype = dtype;
    }
    auto cpu_tensor =
        at::from_blob((void*)data, {static_cast<int64_t>(tensor_size)},
                      at::device(at::kCPU).dtype(tensor_dtype));
    auto device_tensor = at::from_blob(
        storage->mutable_data(), {static_cast<int64_t>(tensor_size)}, {1},
        nullptr, at::device(storage->device()).dtype(tensor_dtype),
        {storage->device()});
    device_tensor.copy_(cpu_tensor);
  }
  return storage;
}

static PyObject* THGPStorage_setFromFile(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  const auto& storage = THPStorage_Unpack(self);
  PyObject* file = PyTuple_GET_ITEM(args, 0);
  PyObject* offset = PyTuple_GET_ITEM(args, 1);
  bool is_real_file = PyTuple_GET_ITEM(args, 2) == Py_True;

  PyObject* element_size_obj = PyTuple_GET_ITEM(args, 3);

  TORCH_CHECK(element_size_obj != Py_None,
              "_set_from_file: need to specify element size");
  uint64_t element_size = THPUtils_unpackUInt64(element_size_obj);

  // deliver dtype to determine whether narrow or not
  at::ScalarType scalar_type = at::ScalarType::Undefined;
  if (PyTuple_Size(args) == 5) {
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 4);
    scalar_type = reinterpret_cast<THPDtype*>(dtype_obj)->scalar_type;
  }

  if (!is_real_file) {
    // offset can be implemented with a call to the Python object's seek()
    // but it is currently unnecessary to support this.
    TORCH_CHECK(offset == Py_None,
                "_set_from_file: offset is NYI for filelike objects");

    auto self_storage_impl = c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(
        storage.unsafeGetStorageImpl());
    auto storage_impl = THGPStorage_readFileRaw<PyObject*>(
        file, std::move(self_storage_impl), element_size, scalar_type);
    if (!storage_impl.defined()) {
      return nullptr;
    }
    Py_INCREF(self);
    return (PyObject*)self;
  }

  // file is backed by a fd
  const int fd = PyObject_AsFileDescriptor(file);
  const auto fd_original_pos = LSEEK(fd, 0, SEEK_CUR);
  if (offset != Py_None) {
    LSEEK(fd, THPUtils_unpackLong(offset), SEEK_SET);
  }
  TORCH_CHECK(fd != -1,
              "_set_from_file couldn't retrieve a file "
              "descriptor from given object");
  auto self_storage_impl = c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(
      storage.unsafeGetStorageImpl());
  auto storage_impl = THGPStorage_readFileRaw<int>(fd, self_storage_impl,
                                                   element_size, scalar_type);
  if (!storage_impl.defined()) return nullptr;
  Py_INCREF(self);

  // the file descriptor is returned to original position and
  // the file handle at python call-site needs updating to the
  // advanced position
  const auto fd_current_pos = LSEEK(fd, 0, SEEK_CUR);
  LSEEK(fd, fd_original_pos, SEEK_SET);
  const auto seek_return =
      PyObject_CallMethod(file, "seek", "Li", (long long)fd_current_pos, 0);
  if (seek_return == nullptr) {
    return nullptr;
  }
  Py_DECREF(seek_return);

  return self;
  END_HANDLE_TH_ERRORS
}

void ReplaceStorageBaseMethods() {
  PyMethodDef* THPStorage_methods = THPStorageType.tp_methods;

  while (true) {
    if (!THPStorage_methods->ml_name) {
      return;
    }

    if (std::string(THPStorage_methods->ml_name) == "copy_") {
      PyMethodDef copy__method = {
          "copy_", castPyCFunctionWithKeywords(THGPStorage_copy_),
          METH_VARARGS | METH_KEYWORDS, nullptr};
      *THPStorage_methods = copy__method;
    } else if (std::string(THPStorage_methods->ml_name) == "_write_file") {
      PyMethodDef _write_file_method = {"_write_file", THGPStorage_writeFile,
                                        METH_VARARGS, nullptr};
      *THPStorage_methods = _write_file_method;
    } else if (std::string(THPStorage_methods->ml_name) == "_set_from_file") {
      PyMethodDef _set_from_file_method = {
          "_set_from_file", THGPStorage_setFromFile, METH_VARARGS, nullptr};
      *THPStorage_methods = _set_from_file_method;
    }

    THPStorage_methods++;
  }
}

}  // namespace torch_gcu
