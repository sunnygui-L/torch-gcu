# Workaround Records (For Internal Members Only)

## Overview

这个文档记录了 torch_gcu 中的一些 workaround 方法, 以供参考

每条记录按照以下形式组织

-  **Issue**: Brief description of the problem.
   - **Description**: : Detailed explanation of the issue, including any error messages or symptoms.
   - **Workaround**: Steps taken to resolve the issue.
   - **Date**: Date when the workaround was implemented.
   - **Notes**: Any additional notes or considerations.

## Known Workaround Records

- **Issue**: `ProcessGroupECCL` 不能复用 event 来进行计算 stream 和通信 stream 的同步
  - **Description**: 与 cudaruntime 不同, topstruntime 在对同一个 `topsEvent` 反复调用 `topsEventRecord()` 会严重影响性能, 因此在 `ProcessGroupECCL` 不能像 `ProcessGroupECCL` 一样对 event 进行复用. 详细见 http://wiki.enflame.cn/display/bwyuchi/ProcessGroupECCL
  - **Workaround**: 对每个通信 work 都重新创建 event 用来进行计算 stream 和通信 stream 的同步, 以新建 event 为代价避免对同一个 `topsEvent` 反复调用 `topsEventRecord()` 带来的性能损失. 详细见 http://gerrit.enflame.cn/c/torch_gcu/+/168223
  - **Date**: 2024-08-26
  - **Notes**: None

- **Issue**: `ProcessGroupECCL::gather()` 接口下发时间长问题
  - **Description**: gather 语义是通过调用 Send + Receive 实现的, 为了让 Sender 进程和 Receiver 进程保持行为上的一致 (复用相同的 `collective()` 接口), 对于 Sender 进程会创建一个空 Tensor (调用 `at::Tensor` 的默认构造) 作为输出. 但是这个空 Tensor 会造成 `collective()` 接口中的 `extractStorages()` 接口的耗时很长. 详细见 http://wiki.enflame.cn/pages/viewpage.action?pageId=207140029
  - **Workaround**: 将 Sender 进程的 input tensor 作为 output (Sender 进程并不关心 output), 来避免对空 Tensor 进行 `extractStorages()`. 详细见 http://gerrit.enflame.cn/c/torch_gcu/+/168223/5/torch_gcu/csrc/distributed/ProcessGroupECCL.cpp#b3752
  - **Date**: 2024-08-26
  - **Notes**: None
