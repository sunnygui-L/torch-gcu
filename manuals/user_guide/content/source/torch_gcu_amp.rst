.. _amp:

###################################
amp使用说明
###################################

官方文档参考：https://pytorch.org/docs/2.9/amp

==================
Autocasting
==================

有两种方式调用autocast功能函数：

.. code-block:: python
    :linenos:
    :emphasize-lines: 9,13

    import torch
    import torch_gcu

    M, N = 256, 1024
    c_lhs = torch.randn((M, N)).gcu()
    c_rhs = torch.randn((N, M)).gcu()

    # 方式一：
    with torch.gcu.amp.autocast():
        d_out = func(c_lhs, c_rhs)

    # 方式二：
    with torch.autocast("gcu"):
        d_out = func(c_lhs, c_rhs)


==================
Gradient Scaling
==================

简单的接口调用示例如下，import之后GradScaler对应变量的使用方式同cuda：

.. code-block:: python
    :linenos:
    :emphasize-lines: 3

    import torch
    import torch_gcu
    from torch.gcu.amp import GradScaler


详细的参数设置和api含义可参考官方文档使用说明。

