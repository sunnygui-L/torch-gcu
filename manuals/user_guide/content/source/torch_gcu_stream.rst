.. _gcu_stream:

####################################
Stream 限制硬件资源使用
####################################

为了让用户控制计算资源的分配， ``torch.gcu.Stream`` 提供了 ``set_limit`` 和 ``get_limit`` 接口。

- set_limit：设置stream可以使用的cluster个数和sip个数
- get_limit：获取stream已经设置的cluster个数和sip个数

.. code-block:: python

    def set_limit(self, cluster_num, sip_num):
        # Set cluster_num and sip_num for this stream, success returns True, failure returns False

    def get_limit(self):
        # Return cluster_num and sip_num for this stream, if it fails, return an False

cluster_num的取值范围是1或2，sip_num的取值范围是1到12。

==============
使用限制
==============

- 目前仅支持 S60 系列加速卡，所有stream的cluster_num * sip_num加起来小于等于24才能真正并行起来


===========
使用示例
===========

.. code-block:: python
    :linenos:
    :emphasize-lines: 10,11

    import torch
    import torch_gcu

    def limit_test():
        cluster_num = 2
        sip_num = 10
        s = torch.gcu.Stream()
        s.set_limit(cluster_num, sip_num)
        x = torch.tensor([1.0, 2.0, 3.0]).gcu()
        # use this stream
        with torch.gcu.stream(s):
            x = x + 5
            y = x * 2
