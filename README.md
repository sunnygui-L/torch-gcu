# Enflame GCU Extension for PyTorch (torch\_gcu)

> **Language**: English | [中文](README_zh.md)

<!-- toc -->

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Option 1: Docker (Recommended)](#option-1-docker-recommended)
  - [Option 2: Build from Source](#option-2-build-from-source)
- [Quick Start](#quick-start)
  - [Verify Installation](#verify-installation)
  - [Tensor Operations](#tensor-operations)
  - [Model Training](#model-training)
  - [Automatic Mixed Precision (AMP)](#automatic-mixed-precision-amp)
  - [Distributed Training](#distributed-training)
  - [CUDA Code Migration](#cuda-code-migration)
  - [LibTorch C++ Integration](#libtorch-c-integration)
- [Key Features](#key-features)
- [Version Compatibility](#version-compatibility)
  - [PyTorch and Python Version Matrix](#pytorch-and-python-version-matrix)
  - [TopsRider SDK Compatibility](#topsrider-sdk-compatibility)
- [Hardware Support](#hardware-support)
- [User Manual](#user-manual)
- [Trouble Shooting](#trouble-shooting)
- [License](#license)

<!-- tocstop -->

## Overview

**torch\_gcu** is the [Enflame Technology](https://www.enflame-tech.com/) PyTorch backend extension that enables PyTorch workloads to run on **Enflame GCU (General Compute Unit)** devices. It integrates with PyTorch through the official [PrivateUse1 backend mechanism](https://pytorch.org/tutorials/advanced/privateuseone.html), allowing users to leverage the powerful computing capabilities of Enflame GCU hardware while maintaining a familiar PyTorch programming experience.

**torch\_gcu** provides:

- **Broad operator coverage** — extensive support for PyTorch ATen operators on GCU
- **Distributed training** — via ECCL (Enflame Collective Communication Library), supporting collective and P2P operations
- **torch.compile / Inductor** — backend integration for graph-level optimization
- **AMP** — Automatic Mixed Precision training with `torch.gcu.amp`
- **Profiler** — integration with PyTorch's native profiler for GCU activity tracing
- **GCU Graph** — `torch.gcu.GCUGraph` for reducing kernel launch overhead (similar to CUDA Graphs)
- **CUDA one-click migration** — `transfer_to_gcu` utility for running existing CUDA code on GCU with minimal changes
- **LibTorch C++ integration** — libtorch\_gcu for C++ inference workloads

## Project Structure

```
torch_gcu/
├── CMakeLists.txt          # Top-level CMake build configuration
├── setup.py                # Python package build script (pip install / bdist_wheel)
├── codegen/                # Code generation scripts and templates for operator registration
│   └── templates/          # Jinja2/code templates used by the generator
├── torch_gcu/              # Python package & C++ source (installed as `import torch_gcu`)
│   ├── csrc/               # C++ source code
│   │   ├── aten/           # ATen operator implementations and dispatch registration
│   │   ├── gcu/            # GCU device management, memory, stream, and hardware utilities
│   │   ├── distributed/    # ECCL-based distributed communication backend
│   │   ├── profiler/       # GCU profiler integration with PyTorch profiler
│   │   ├── aotfusion/      # AOT (Ahead-Of-Time) fusion optimization passes
│   │   ├── efficient_ops/  # High-performance fused operator implementations
│   │   ├── libkineto_gcu/  # Kineto profiler plugin for GCU activity tracing
│   │   └── python/         # Python C++ extension bindings (pybind11)
│   ├── gcu/                # Python-side GCU runtime modules
│   │   ├── amp/            # Automatic Mixed Precision (autocast, GradScaler)
│   │   ├── inductor/       # torch.compile / Inductor backend for GCU
│   │   ├── profiler/       # Python profiler wrappers
│   │   └── autograd/       # Autograd function extensions
│   └── distributed/        # Python distributed training utilities
├── manuals/                # User documentation (Sphinx/RST)
├── cmake/                  # CMake helper modules
├── scripts/                # Build and utility scripts
└── tools/                  # Development and analysis tools
```

## Installation

### Prerequisites

torch\_gcu depends on the **Enflame TOPS software stack**. Before installing torch\_gcu, ensure the following components are installed:


| Component       | Description                                                                                      |
| --------------- | ------------------------------------------------------------------------------------------------ |
| **TopsRider**   | Enflame AI development toolkit (recommended: install all components via the TopsRider installer) |
| **TopsRuntime** | GCU runtime library                                                                              |
| **TopsAten**    | Optimized operator library for GCU                                                               |
| **ECCL**        | Enflame Collective Communication Library (for distributed training)                              |

> **Recommended**: Use the **TopsRider installer** for one-click setup of all dependencies. Refer to the TopsRider installation documentation for details.

### Option 1: Docker (Recommended)

The easiest way to get started is using the pre-built Docker image with all dependencies included.

1. **Pull and start the container**:

   ```bash
   IMAGE=registry-egc.enflame-tech.com/artifacts/torch_gcu:v2.10.0-TR3.7.107-ubuntu2204

   docker run --name torch_gcu -d \
     -v /home:/home \
     --shm-size 8G \
     --ipc=host --network host \
     --cap-add SYS_PTRACE \
     --security-opt seccomp=unconfined \
     --privileged \
     "$IMAGE" \
     tail -f /dev/null
   ```
2. **Update the host GCU driver** (to match the image's software version):

   ```bash
   # Extract the matching driver from the container
   docker cp torch_gcu:/enflame/driver ./

   # Install the driver on the host
   sudo driver/enflame-x86_64-gcc-1.7.2.2402-20260429134535.run -y

   # Restart the container to pick up the new driver
   docker restart torch_gcu
   ```
3. **Enter the container and verify**:

   ```bash
   docker exec -it torch_gcu bash
   python -c "import torch; import torch_gcu; print(torch.gcu.is_available())"
   ```

### Option 2: Build from Source

1. **Clone the repository and start the container**:

   ```bash
   cd /home
   git clone git@github.com:EnflameTechnology/torch-gcu.git
   docker exec -it torch_gcu bash
   ```
2. **Build and install inside the container**:

   ```bash
   cd torch_gcu
   python setup.py bdist_wheel
   pip install ./dist/torch_gcu-2.10.0-*.whl
   ```

## Quick Start

### Verify Installation

```python
import torch
import torch_gcu

print(torch.gcu.is_available())  # True
print(torch.gcu.device_count())  # Number of GCU devices
```

### Tensor Operations

```python
import torch
import torch_gcu

x = torch.randn(4, 4).to("gcu")
y = torch.randn(4, 4).to("gcu")
z = x @ y
print(z)
```

Tensors can be moved to GCU in multiple ways:

```python
a = torch.tensor([1, 2, 3]).gcu()
b = torch.tensor([1, 2, 3]).to("gcu")
c = torch.tensor([1, 2, 3], device="gcu")
```

### Model Training

```python
import torch
import torch.nn as nn
import torch_gcu

model = nn.Linear(128, 10).gcu()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

inputs = torch.randn(32, 128).gcu()
targets = torch.randint(0, 10, (32,)).gcu()
criterion = nn.CrossEntropyLoss()

for step in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

print(f"Final loss: {loss.item():.4f}")
```

### Automatic Mixed Precision (AMP)

```python
import torch
import torch_gcu

model = MyModel().gcu()
optimizer = torch.optim.Adam(model.parameters())
scaler = torch.gcu.amp.GradScaler()

for inputs, targets in dataloader:
    inputs, targets = inputs.gcu(), targets.gcu()
    with torch.gcu.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

Alternatively, use `torch.autocast("gcu")` as the context manager.

### Distributed Training

torch\_gcu supports distributed training via the **ECCL** backend:

```python
import torch
import torch.distributed as dist
import torch_gcu

dist.init_process_group(backend="eccl", world_size=world_size, rank=rank)

model = MyModel().gcu()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

Supported collective operations include: `broadcast`, `all_reduce`, `reduce`, `all_gather`, `gather`, `scatter`, `reduce_scatter`, `all_to_all`, `barrier`, and all P2P operations (`send`, `recv`, `isend`, `irecv`).

Launch with `torchrun`:

```bash
torchrun --nproc_per_node=8 train.py
```

### CUDA Code Migration

torch\_gcu provides a **one-click migration** utility for running existing CUDA code on GCU without modifying user code:

```python
import torch
import torch_gcu
from torch_gcu import transfer_to_gcu  # Add this line before your code

# Your existing CUDA code works as-is
x = torch.randn(4, 4).cuda()  # Automatically redirected to GCU
model = MyModel().cuda()       # Automatically redirected to GCU
```

> **Note**: This feature requires the CUDA build of PyTorch. `torch.jit.script` is globally disabled when `transfer_to_gcu` is active.

For manual migration, replace `cuda` with `gcu` and `nccl` with `eccl`:


| CUDA Code                     | GCU Code                     |
| ----------------------------- | ---------------------------- |
| `tensor.cuda()`               | `tensor.gcu()`               |
| `model.to("cuda")`            | `model.to("gcu")`            |
| `torch.cuda.synchronize()`    | `torch.gcu.synchronize()`    |
| `backend="nccl"`              | `backend="eccl"`             |
| `torch.cuda.amp.autocast()`   | `torch.gcu.amp.autocast()`   |
| `torch.cuda.amp.GradScaler()` | `torch.gcu.amp.GradScaler()` |

### LibTorch C++ Integration

torch\_gcu provides **libtorch\_gcu** for C++ inference and development. This section covers compilation and testing.

#### Building libtorch_gcu

1. **Build from source with libtorch_gcu enabled**:

   ```bash
   cd torch_gcu
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_LIBTORCH_GCU=ON \
         -DPYTORCH_INSTALL_DIR=/path/to/libtorch \
         ..
   make -j$(nproc)
   make install
   ```

   After building, the libtorch\_gcu package will be installed to the build directory under `libtorch_gcu/`.
2. **Package structure**:

   ```
   libtorch_gcu/
   ├── include/           # Header files (torch_gcu.h)
   ├── lib/              # Shared libraries (libtorch_gcu.so)
   └── share/cmake/TorchGCU/  # CMake config files
   ```

#### Using libtorch_gcu in C++ Projects

1. **Download libtorch**:

   Get the official PyTorch libtorch package (version 2.10.0):

   - ABI=1: `https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.10.0%2Bcpu.zip`
   - ABI=0: `https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip`

   > **Important**: Ensure libtorch and libtorch\_gcu use the **same ABI version** (both ABI=1 or both ABI=0).
   >
2. **Example C++ code** (`example-app.cpp`):

   ```cpp
   #include <iostream>
   #include <torch/torch.h>
   #include "torch_gcu.h"  // Include libtorch_gcu header

   int main() {
       // Create tensors on GCU using torch::kPrivateUse1
       torch::Tensor tensor1 =
           torch::rand({2, 3}, torch::TensorOptions().device(torch::kPrivateUse1));
       torch::Tensor tensor2 =
           torch::rand({2, 3}).to(torch::kPrivateUse1);

       std::cout << tensor1 << std::endl;
       return 0;
   }
   ```
3. **CMakeLists.txt**:

   ```cmake
   cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
   project(example-app)
   set(CMAKE_CXX_STANDARD 17)

   # Find libtorch
   find_package(Torch REQUIRED)
   # Find libtorch_gcu
   find_package(TorchGCU REQUIRED)

   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${TORCH_GCU_CXX_FLAGS}")

   add_executable(example-app example-app.cpp)
   target_link_libraries(example-app ${TORCH_LIBRARIES})
   target_link_libraries(example-app ${TORCH_GCU_LIBRARIES})
   ```
4. **Build and run**:

   ```bash
   mkdir build && cd build
   cmake -DCMAKE_PREFIX_PATH="/path/to/libtorch;/path/to/libtorch_gcu" ..
   cmake --build . --config Release
   ./example-app
   ```

   Expected output:

   ```
    0.1394  0.3388  0.5241
    0.8096  0.9733  0.0577
   [ privateuseoneFloatType{2,3} ]
   ```

#### LibTorch Test Examples

The torch\_gcu repository includes test examples demonstrating various libtorch\_gcu features:

1. **Basic tensor operations** (`test_from_blob.cc`):

   - Tests `torch::from_blob` with GCU device memory
   - Demonstrates direct memory allocation with `topsMalloc`
2. **Model inference** (`test_resnet50.cc`):

   - Single-threaded inference with TorchScript model
   - Multi-threaded inference with single stream
   - Multi-threaded inference with multiple GCU streams

**Building and running tests**:

```bash
# Prerequisites: Install torchvision for model generation
pip install torchvision

# Build torch_gcu with tests
cd torch_gcu
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_LIBTORCH_GCU=ON \
      -DPYTORCH_INSTALL_DIR=/path/to/libtorch \
      ..
make -j$(nproc)

# Navigate to test directory
cd test/ci_test/libtorch

# Generate test model
python gen_resnet50.py

# Set library paths
export LD_LIBRARY_PATH=/path/to/libtorch_gcu/lib:/path/to/libtorch/lib:$LD_LIBRARY_PATH

# Run tests
./test_from_blob
./test_resnet50
```

**Test capabilities**:

- `test_from_blob`: Validates GCU memory allocation and tensor creation
- `test_resnet50`:
  - `SingleThreadSingleStream`: Basic model inference accuracy test
  - `MultiThreadSingleStream`: Thread safety with default stream
  - `MultiThreadMultiStream`: Stream management with `torch_gcu::GCUStreamGuard`

> **Note**: libtorch\_gcu currently supports **single-device inference only**. Multi-device and training scenarios are not yet supported in the C++ API.

## Key Features


| Feature                | Description                                                                     |
| ---------------------- | ------------------------------------------------------------------------------- |
| **Operator Coverage**  | Extensive ATen operator support; unsupported ops automatically fall back to CPU |
| **Distributed (ECCL)** | Full collective and P2P communication support                                   |
| **torch.compile**      | Inductor backend integration for graph-level optimizations                      |
| **AMP**                | `torch.gcu.amp.autocast()` and `GradScaler`                                     |
| **Profiler**           | `ProfilerActivity.GCU` for GCU kernel tracing; export to Chrome trace           |
| **GCU Graph**          | `torch.gcu.GCUGraph` for reducing dispatch overhead                             |
| **CUDA Migration**     | `transfer_to_gcu` for zero-change CUDA-to-GCU migration                         |
| **LibTorch**           | C++ inference via libtorch\_gcu                                                 |
| **Op Debug**           | Rich debugging tools: sync mode, CPU fallback, op statistics, I/O dump          |
| **Memory Management**  | PyTorch-compatible caching allocator; memory snapshot visualization             |
| **Stream Control**     | Hardware stream/SIP resource management (S60)                                   |

## Version Compatibility

### PyTorch and Python Version Matrix


| PyTorch Version | Python Version  |
| --------------- | --------------- |
| 2.10.0          | 3.9, 3.10, 3.12 |

### TopsRider SDK Compatibility


| TopsRider Version | torch\_gcu Version | PyTorch Version | Branch |
| ----------------- | ------------------ | --------------- | ------ |
| v3.7.1            | 2.10.0             | 2.10.0          | main   |

> **Note**: torch\_gcu versions follow the naming convention `{PyTorch version}`, where the version number directly maps to the supported PyTorch release. The TopsRider SDK version determines the underlying driver and library compatibility.

## Hardware Support


| Product Series                | Product Model |
| ----------------------------- | ------------- |
| Enflame CloudBlazer Inference | S60           |

## User Manual

For detailed documentation including API references, operator support lists, and advanced topics, refer to the bundled user manual under `manuals/`:


| Document                                                                          | Description                                                         |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [User Guide](manuals/user_guide/content/source/index.rst)                         | Complete usage guide including migration, AMP, profiling, debugging |
| [Operator Support List](manuals/oplist_torch_gcu/content/source/torch_gcu_op.rst) | Comprehensive list of supported operators with notes                |
| [CUDA Migration Guide](manuals/user_guide/content/source/transfer_to_gcu.rst)     | Detailed`transfer_to_gcu` API conversion tables                     |
| [Op Debug Guide](manuals/user_guide/content/source/torch_gcu_op_debug.rst)        | Op-level debugging tools and configuration                          |
| [LibTorch Guide](manuals/user_guide/content/source/libtorch_gcu.rst)              | C++ inference with libtorch\_gcu                                    |

## Trouble Shooting

- **`torch.gcu.is_available()` returns `False`**

  - Ensure TopsRider / TOPS SDK is installed and the GCU driver is loaded
  - Verify `LD_LIBRARY_PATH` includes TOPS library paths (e.g., `/opt/tops/lib`)
- **`ImportError: libtopsrt.so: cannot open shared object file`**

  - TopsRuntime is not installed or not in the library search path
  - Fix: `export LD_LIBRARY_PATH=/opt/tops/lib:$LD_LIBRARY_PATH`
- **Distributed init fails with `backend="eccl"`**

  - Ensure ECCL is installed (included in TopsRider)
  - Check that multi-card communication is properly configured
- **Unexpected precision loss or overflow**

  - GCU hardware does not natively support 64-bit types (F64, I64). torch\_gcu performs implicit down-casting to 32-bit, which may cause precision loss. Use `TORCH_GCU_ENABLE_INT64_AND_UINT64=true` to attempt 64-bit support where available.
- **Performance is slower than expected**

  - Ensure profiler is disabled for benchmarking (profiler introduces synchronization gaps)
  - Check for CPU fallback ops: `export ENFLAME_LOG_DEBUG_LEVEL="DEBUG"` and `export ENFLAME_LOG_DEBUG_MOD="TORCH_GCU/FALLBACK"`
  - Consider enabling GCU Graph for repetitive workloads

## License

torch\_gcu has a BSD-style license. Copyright (c) Enflame Technology. All Rights Reserved.

See the [LICENSE](LICENSE) file for details.
