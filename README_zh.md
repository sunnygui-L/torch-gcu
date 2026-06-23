# 燧原 GCU PyTorch 扩展 (torch\_gcu)

> **语言**: [English](README.md) | 中文

<!-- toc -->

- [概述](#概述)
- [项目结构](#项目结构)
- [安装](#安装)
  - [前置依赖](#前置依赖)
  - [方式一：Docker安装（推荐）](#方式一docker安装推荐)
  - [方式二：源码编译安装](#方式二源码编译安装)
- [快速开始](#快速开始)
  - [验证安装](#验证安装)
  - [Tensor运算](#tensor运算)
  - [模型训练](#模型训练)
  - [自动混合精度（AMP）](#自动混合精度amp)
  - [分布式训练](#分布式训练)
  - [CUDA代码一键迁移](#cuda代码一键迁移)
  - [LibTorch C++ 集成](#libtorch-c-集成)
- [核心特性](#核心特性)
- [版本配套](#版本配套)
  - [PyTorch与Python版本对照](#pytorch与python版本对照)
  - [TopsRider SDK配套关系](#topsrider-sdk配套关系)
- [硬件支持](#硬件支持)
- [用户手册](#用户手册)
- [常见问题](#常见问题)
- [许可证](#许可证)

<!-- tocstop -->

## 概述

**torch\_gcu** 是[燧原科技](https://www.enflame-tech.com/)开发的 PyTorch 后端扩展，使 PyTorch 工作负载能够运行在**燧原 GCU（通用计算单元）** 设备上。它通过 PyTorch 官方的 [PrivateUse1 后端机制](https://pytorch.org/tutorials/advanced/privateuseone.html)进行集成，让用户在保持 PyTorch 原生编程体验的同时，充分发挥燧原 GCU 硬件的强大算力。

**torch\_gcu** 提供以下核心能力：

- **广泛的算子覆盖** — 对 PyTorch ATen 算子的全面 GCU 支持
- **分布式训练** — 基于 ECCL（Enflame Collective Communication Library）, 支持集合通信和点对点通信
- **torch.compile / Inductor** — 图级别优化的后端集成
- **AMP** — 通过 `torch.gcu.amp` 支持自动混合精度训练
- **Profiler** — 集成 PyTorch 原生 Profiler，支持 GCU 活动追踪
- **GCU Graph** — `torch.gcu.GCUGraph` 用于降低 kernel 调度开销（类似 CUDA Graphs）
- **CUDA 一键迁移** — `transfer_to_gcu` 工具，现有 CUDA 代码零改动运行在 GCU 上
- **LibTorch C++ 集成** — libtorch\_gcu 支持 C++ 推理场景

## 项目结构

```
torch_gcu/
├── CMakeLists.txt          # 顶层 CMake 构建配置
├── setup.py                # Python 包构建脚本（pip install / bdist_wheel）
├── codegen/                # 算子注册代码生成脚本与模板
│   └── templates/          # 生成器使用的 Jinja2/代码模板
├── torch_gcu/              # Python 包 & C++ 源码（通过 `import torch_gcu` 使用）
│   ├── csrc/               # C++ 源代码
│   │   ├── aten/           # ATen 算子实现与 dispatch 注册
│   │   ├── gcu/            # GCU 设备管理、显存、Stream 及硬件工具
│   │   ├── distributed/    # 基于 ECCL 的分布式通信后端
│   │   ├── profiler/       # GCU Profiler 与 PyTorch Profiler 集成
│   │   ├── aotfusion/      # AOT（提前编译）融合优化 Pass
│   │   ├── efficient_ops/  # 高性能融合算子实现
│   │   ├── libkineto_gcu/  # Kineto Profiler 插件，用于 GCU 活动追踪
│   │   └── python/         # Python C++ 扩展绑定（pybind11）
│   ├── gcu/                # Python 侧 GCU 运行时模块
│   │   ├── amp/            # 自动混合精度（autocast、GradScaler）
│   │   ├── inductor/       # torch.compile / Inductor 后端（GCU）
│   │   ├── profiler/       # Python Profiler 封装
│   │   └── autograd/       # Autograd 函数扩展
│   └── distributed/        # Python 分布式训练工具
├── manuals/                # 用户文档（Sphinx/RST）
├── cmake/                  # CMake 辅助模块
├── scripts/                # 构建与工具脚本
└── tools/                  # 开发与分析工具
```

## 安装

### 前置依赖

torch\_gcu 依赖 **燧原 TOPS 软件栈**。安装 torch\_gcu 前，请确保以下组件已正确安装：


| 组件            | 说明                                                                  |
| --------------- | --------------------------------------------------------------------- |
| **TopsRider**   | 燧原 AI 开发工具包（推荐：通过 TopsRider installer 一键安装所有组件） |
| **TopsRuntime** | GCU 运行时库                                                          |
| **TopsAten**    | GCU 优化算子库                                                        |
| **ECCL**        | 燧原集合通信库（分布式训练所需）                                      |

> **推荐**：使用 **TopsRider installer** 一键安装所有依赖。具体安装方法请参考 TopsRider 安装文档。

### 方式一：Docker安装（推荐）

使用预构建 Docker 镜像是最快的上手方式，镜像中已包含所有依赖。

1. **拉取镜像并启动容器**：

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
2. **更新主机 GCU 驱动**（使其与镜像中的软件版本匹配）：

   ```bash
   # 从容器中获取匹配的驱动版本
   docker cp torch_gcu:/enflame/driver ./

   # 在主机上安装驱动
   sudo driver/enflame-x86_64-gcc-1.7.2.2402-20260429134535.run -y

   # 重启容器以使用更新后的驱动
   docker restart torch_gcu
   ```
3. **进入容器并验证**：

   ```bash
   docker exec -it torch_gcu bash
   python -c "import torch; import torch_gcu; print(torch.gcu.is_available())"
   ```

### 方式二：源码编译安装

1. **克隆代码仓库并启动容器**：

   ```bash
   cd /home
   git clone git@github.com:EnflameTechnology/torch-gcu.git
   docker exec -it torch_gcu bash
   ```
2. **在容器中编译安装**：

   ```bash
   cd torch_gcu
   python setup.py bdist_wheel
   pip install ./dist/torch_gcu-2.10.0-*.whl
   ```

## 快速开始

### 验证安装

```python
import torch
import torch_gcu

print(torch.gcu.is_available())  # True
print(torch.gcu.device_count())  # GCU 设备数量
```

### Tensor运算

```python
import torch
import torch_gcu

x = torch.randn(4, 4).to("gcu")
y = torch.randn(4, 4).to("gcu")
z = x @ y
print(z)
```

将 Tensor 移动到 GCU 的多种方式：

```python
a = torch.tensor([1, 2, 3]).gcu()
b = torch.tensor([1, 2, 3]).to("gcu")
c = torch.tensor([1, 2, 3], device="gcu")
```

### 模型训练

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

### 自动混合精度（AMP）

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

也可以使用 `torch.autocast("gcu")` 作为上下文管理器。

### 分布式训练

torch\_gcu 通过 **ECCL** 后端支持分布式训练：

```python
import torch
import torch.distributed as dist
import torch_gcu

dist.init_process_group(backend="eccl", world_size=world_size, rank=rank)

model = MyModel().gcu()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

支持的集合通信操作包括：`broadcast`、`all_reduce`、`reduce`、`all_gather`、`gather`、`scatter`、`reduce_scatter`、`all_to_all`、`barrier`，以及全部点对点操作（`send`、`recv`、`isend`、`irecv`）。

使用 `torchrun` 启动：

```bash
torchrun --nproc_per_node=8 train.py
```

### CUDA代码一键迁移

torch\_gcu 提供**一键迁移**工具，无需修改用户代码即可将现有 CUDA 代码运行在 GCU 上：

```python
import torch
import torch_gcu
from torch_gcu import transfer_to_gcu  # 在用户代码之前添加此行

# 现有 CUDA 代码无需修改
x = torch.randn(4, 4).cuda()  # 自动重定向到 GCU
model = MyModel().cuda()       # 自动重定向到 GCU
```

> **注意**：此功能依赖 CUDA 版本的 PyTorch。启用 `transfer_to_gcu` 后 `torch.jit.script` 将被全局禁用。

手动迁移时，将 `cuda` 替换为 `gcu`，`nccl` 替换为 `eccl`：


| CUDA 代码                     | GCU 代码                     |
| ----------------------------- | ---------------------------- |
| `tensor.cuda()`               | `tensor.gcu()`               |
| `model.to("cuda")`            | `model.to("gcu")`            |
| `torch.cuda.synchronize()`    | `torch.gcu.synchronize()`    |
| `backend="nccl"`              | `backend="eccl"`             |
| `torch.cuda.amp.autocast()`   | `torch.gcu.amp.autocast()`   |
| `torch.cuda.amp.GradScaler()` | `torch.gcu.amp.GradScaler()` |

### LibTorch C++ 集成

torch\_gcu 提供 **libtorch\_gcu** 用于 C++ 推理和开发。本节介绍编译和测试方法。

#### 编译 libtorch_gcu

1. **从源码编译并启用 libtorch_gcu**：

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

   编译完成后，libtorch\_gcu 包将被安装到构建目录下的 `libtorch_gcu/` 中。
2. **包结构**：

   ```
   libtorch_gcu/
   ├── include/           # 头文件（torch_gcu.h）
   ├── lib/              # 动态库（libtorch_gcu.so）
   └── share/cmake/TorchGCU/  # CMake 配置文件
   ```

#### 在 C++ 项目中使用 libtorch_gcu

1. **下载 libtorch**：

   获取官方 PyTorch libtorch 包（版本 2.10.0）：

   - ABI=1：`https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.10.0%2Bcpu.zip`
   - ABI=0：`https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip`

   > **重要**：确保 libtorch 和 libtorch\_gcu 使用**相同的 ABI 版本**（都是 ABI=1 或都是 ABI=0）。
   >
2. **C++ 示例代码**（`example-app.cpp`）：

   ```cpp
   #include <iostream>
   #include <torch/torch.h>
   #include "torch_gcu.h"  // 引入 libtorch_gcu 头文件

   int main() {
       // 使用 torch::kPrivateUse1 在 GCU 上创建张量
       torch::Tensor tensor1 =
           torch::rand({2, 3}, torch::TensorOptions().device(torch::kPrivateUse1));
       torch::Tensor tensor2 =
           torch::rand({2, 3}).to(torch::kPrivateUse1);

       std::cout << tensor1 << std::endl;
       return 0;
   }
   ```
3. **CMakeLists.txt**：

   ```cmake
   cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
   project(example-app)
   set(CMAKE_CXX_STANDARD 17)

   # 查找 libtorch
   find_package(Torch REQUIRED)
   # 查找 libtorch_gcu
   find_package(TorchGCU REQUIRED)

   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${TORCH_GCU_CXX_FLAGS}")

   add_executable(example-app example-app.cpp)
   target_link_libraries(example-app ${TORCH_LIBRARIES})
   target_link_libraries(example-app ${TORCH_GCU_LIBRARIES})
   ```
4. **编译和运行**：

   ```bash
   mkdir build && cd build
   cmake -DCMAKE_PREFIX_PATH="/path/to/libtorch;/path/to/libtorch_gcu" ..
   cmake --build . --config Release
   ./example-app
   ```

   预期输出：

   ```
    0.1394  0.3388  0.5241
    0.8096  0.9733  0.0577
   [ privateuseoneFloatType{2,3} ]
   ```

#### LibTorch 测试示例

torch\_gcu 仓库包含测试示例，展示 libtorch\_gcu 的各项功能：

1. **基础张量操作**（`test_from_blob.cc`）：

   - 测试 `torch::from_blob` 在 GCU 设备内存上的使用
   - 演示使用 `topsMalloc` 直接分配内存
2. **模型推理**（`test_resnet50.cc`）：

   - 单线程单流推理（TorchScript 模型）
   - 多线程单流推理
   - 多线程多流推理（使用 GCU 流）

**编译和运行测试**：

```bash
# 前置条件：安装 torchvision 用于生成模型
pip install torchvision

# 编译 torch_gcu 及测试
cd torch_gcu
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_LIBTORCH_GCU=ON \
      -DPYTORCH_INSTALL_DIR=/path/to/libtorch \
      ..
make -j$(nproc)

# 进入测试目录
cd test/ci_test/libtorch

# 生成测试模型
python gen_resnet50.py

# 设置库路径
export LD_LIBRARY_PATH=/path/to/libtorch_gcu/lib:/path/to/libtorch/lib:$LD_LIBRARY_PATH

# 运行测试
./test_from_blob
./test_resnet50
```

**测试功能**：

- `test_from_blob`：验证 GCU 内存分配和张量创建
- `test_resnet50`：
  - `SingleThreadSingleStream`：基础模型推理精度测试
  - `MultiThreadSingleStream`：使用默认流的线程安全测试
  - `MultiThreadMultiStream`：使用 `torch_gcu::GCUStreamGuard` 的流管理测试

> **注意**：libtorch\_gcu 目前仅支持**单卡推理**场景。多卡和训练场景在 C++ API 中暂不支持。

## 核心特性


| 特性               | 说明                                                           |
| ------------------ | -------------------------------------------------------------- |
| **算子覆盖**       | 广泛的 ATen 算子支持；未支持算子自动回退到 CPU 执行            |
| **分布式（ECCL）** | 完整的集合通信和点对点通信支持                                 |
| **torch.compile**  | Inductor 后端集成，支持图级别优化                              |
| **AMP**            | `torch.gcu.amp.autocast()` 和 `GradScaler`                     |
| **Profiler**       | `ProfilerActivity.GCU` 支持 GCU kernel 追踪；导出 Chrome Trace |
| **GCU Graph**      | `torch.gcu.GCUGraph` 降低调度开销                              |
| **CUDA 迁移**      | `transfer_to_gcu` 零改动 CUDA→GCU 迁移                        |
| **LibTorch**       | libtorch\_gcu C++ 推理                                         |
| **算子调试**       | 丰富的调试工具：同步模式、CPU 回退、算子统计、I/O 转储         |
| **显存管理**       | 兼容 PyTorch 的缓存分配器；支持 memory snapshot 可视化         |
| **Stream 控制**    | 硬件 Stream/SIP 资源管理（S60）                                |

## 版本配套

### PyTorch与Python版本对照


| PyTorch 版本 | Python 版本     |
| ------------ | --------------- |
| 2.10.0       | 3.9, 3.10, 3.12 |

### TopsRider SDK配套关系


| TopsRider 版本 | torch\_gcu 版本 | PyTorch 版本 | 分支 |
| -------------- | --------------- | ------------ | ---- |
| v3.7.1         | 2.10.0          | 2.10.0       | main |

> **说明**：torch\_gcu 版本号遵循 `{PyTorch版本}` 命名规则，版本号直接映射到支持的 PyTorch 发布版本。TopsRider SDK 版本决定底层驱动和库的兼容性。

## 硬件支持


| 产品系列                  | 产品型号 |
| ------------------------- | -------- |
| 燧原 CloudBlazer 推理系列 | S60      |

## 用户手册

详细文档包括 API 参考、算子支持列表和高级主题，请参考 `manuals/` 目录下的用户手册：


| 文档                                                                     | 说明                                          |
| ------------------------------------------------------------------------ | --------------------------------------------- |
| [用户指南](manuals/user_guide/content/source/index.rst)                  | 完整使用指南，涵盖迁移、AMP、性能分析、调试等 |
| [算子支持列表](manuals/oplist_torch_gcu/content/source/torch_gcu_op.rst) | 完整的算子支持情况列表及注意事项              |
| [CUDA迁移指南](manuals/user_guide/content/source/transfer_to_gcu.rst)    | `transfer_to_gcu` 详细接口转换对照表          |
| [算子调试指南](manuals/user_guide/content/source/torch_gcu_op_debug.rst) | 算子级别调试工具与配置                        |
| [LibTorch指南](manuals/user_guide/content/source/libtorch_gcu.rst)       | libtorch\_gcu C++ 推理                        |

## 常见问题

- **`torch.gcu.is_available()` 返回 `False`**

  - 确认 TopsRider / TOPS SDK 已正确安装且 GCU 驱动已加载
  - 检查 `LD_LIBRARY_PATH` 是否包含 TOPS 库路径（如 `/opt/tops/lib`）
- **`ImportError: libtopsrt.so: cannot open shared object file`**

  - TopsRuntime 未安装或不在库搜索路径中
  - 解决：`export LD_LIBRARY_PATH=/opt/tops/lib:$LD_LIBRARY_PATH`
- **使用 `backend="eccl"` 初始化分布式失败**

  - 确认 ECCL 已安装（TopsRider 中包含）
  - 检查多卡通信环境是否正确配置
- **出现精度损失或数值溢出**

  - GCU 硬件不原生支持 64 位类型（F64、I64），torch\_gcu 会进行隐式降精到 32 位，可能导致精度损失。设置 `TORCH_GCU_ENABLE_INT64_AND_UINT64=true` 可在支持的场景下启用 64 位支持。
- **性能低于预期**

  - 确保基准测试时关闭 Profiler（Profiler 会引入同步间隙）
  - 检查是否存在 CPU 回退算子：`export ENFLAME_LOG_DEBUG_LEVEL="DEBUG"` 和 `export ENFLAME_LOG_DEBUG_MOD="TORCH_GCU/FALLBACK"`
  - 对重复性工作负载考虑使用 GCU Graph

## 许可证

torch\_gcu 采用 BSD 风格许可证。版权所有 (c) 燧原科技。

详见 [LICENSE](LICENSE) 文件。
