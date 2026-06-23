.. _libtorch_gcu:

####################################
libtorch_gcu 使用说明
####################################

libtorch 是 PyTorch 的 C++接口库，它为 C++ 程序提供了与 PyTorch 相同的深度学习功能，但允许开发者在无需 Python 依赖的情况下构建和运行模型。

libtorch 的文档见 https://pytorch.org/cppdocs/installing.html

libtorch_gcu 是为了支持 libtorch 在 GCU 上运行而开发的插件, 需要和 libtorch 一起配合使用


=========
版本说明
=========

适用软件版本：``v3.7.1``


==============
依赖与支持情况
==============

支持libtorch版本: ``v2.10.0``

Tops相关软件依赖: ``topsruntime``, ``eccl``, ``topsaten``, ``sdk``


==============
使用限制
==============

- libtorch_gcu 目前仅支持单卡推理场景

===========
使用示例
===========

- 一、准备 libtorch 和 libtorch_gcu 包

    libtorch 2.10.0 版本下载地址

        abi = 1: ``https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.10.0%2Bcpu.zip``

        abi = 0: ``https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip``

    libtorch_gcu可以在TopsRider中获取，同样提供两个版本，分别对应abi=1和abi=0的libtorch版本

        abi = 1: ``libtorch_gcu-cxx11-abi-2.10.0+xxx.zip``

        abi = 0: ``libtorch_gcu-2.10.0+xxx.zip``

.. warning::
   使用时务必保证libtorch和libtorch_gcu的abi版本一致，否则会出现编译报错。

- 二、编写源文件

    .. code-block:: cpp
        :linenos:
        :emphasize-lines: 6,11,13


        // example-app.cpp
        #include <iostream>
        #include <torch/torch.h>

        // 需要Include头文件以使能libtorch_gcu
        #include "torch_gcu.h"

        int main() {
            // 将tensor的device设置为torch::kPrivateUse1
            // 后续相关计算会在GCU上执行
            torch::Tensor tensor1 =
                torch::rand({2, 3}, torch::TensorOptions().device(torch::kPrivateUse1));
            torch::Tensor tensor2 =
                torch::rand({2, 3}).to(torch::kPrivateUse1);
            std::cout << tensor1 << std::endl;
        }


- 三、编写 CMakeLists.txt

    .. code-block:: cmake
        :linenos:
        :emphasize-lines: 9,16


        # CMakeLists.txt
        cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
        project(example-app)
        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

        find_package(Torch REQUIRED)
        # 引用libtorch_gcu库
        find_package(TorchGCU REQUIRED)

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${TORCH_GCU_CXX_FLAGS}")

        add_executable(example-app example-app.cpp)
        target_link_libraries(example-app ${TORCH_LIBRARIES})
        # 链接libtorch_gcu相关so
        target_link_libraries(example-app ${TORCH_GCU_LIBRARIES})


- 四、编译工程

    .. code-block:: console

        mkdir build
        cd build
        cmake -DCMAKE_PREFIX_PATH="./libtorch;./libtorch_gcu" ..
        cmake --build . --config Release

    编译后在build目录中会生成二进制文件 ``example-app``

- 五、运行编译得到的二进制文件可以看到输出

    .. code-block:: console

        0.1394  0.3388  0.5241
        0.8096  0.9733  0.0577
        [ privateuseoneFloatType{2,3} ]


==============
其他说明
==============
libtorch_gcu支持Op debug工具，详细说明见 :ref:`op_debug` 。
