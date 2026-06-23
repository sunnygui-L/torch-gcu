.. _gcu_graph:

####################################
GCUGraph 使用说明
####################################

GCUGraph 是 torch_gcu 提供的对标 PyTorch CUDAGraph 的功能

CUDAGraph 的相关使用方法见:

- https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- https://pytorch.org/docs/2.9/generated/torch.cuda.CUDAGraph.html

torch_gcu 的 GCUGraph 功能和 PyTorch 中的 CUDAGraph 使用方法一致, 只需要做两种替换:

-  ``torch.cuda.CUDAGraph`` 替换为 ``torch.gcu.GCUGraph``
-  其余 ``torch.cuda`` 下的接口替换为 ``torch.gcu`` 下的同名接口, 例如将 ``torch.cuda.is_current_stream_capturing()`` 替换为 ``torch.gcu.is_current_stream_capturing()``

也可以尝试一键迁移的方式自动替换 :ref:`transfer_to_gcu`

==============
使用限制
==============

- 不支持 ``torch.gcu.make_graphed_callables()``

- 不支持捕获随机数算子

===========
使用示例
===========

.. code-block:: python
    :linenos:
    :emphasize-lines: 13,14,15,19,31

    import torch
    import torch_gcu

    def model(output_tensor, input_tensor):
        # ...

    @torch.no_grad()
    def graph_test():
        static_input = torch.randn(1024, 1024, device="gcu", dtype=torch.float32)
        static_output = torch.randn(1024, 1024, device="gcu", dtype=torch.float32)

        # ============== Capture ==============
        g = torch.gcu.GCUGraph()
        g.enable_debug_mode()
        with torch.gcu.graph(g):
            model(static_output, static_input)

        # dump
        g.debug_dump("dist_graph.dot")

        # ============== Relay ==============
        # real run
        real_inputs = [
            torch.randn(1024, 1024, device="gcu", dtype=torch.float32)
            for _ in range(15)
        ]

        for real_in in real_inputs:
            # run graph
            static_input.copy_(real_in)
            g.replay()
