import torch
from torch.distributed.distributed_c10d import  _rank_not_in_group, _warn_not_in_group,\
      _check_single_tensor, _ensure_all_tensors_same_dtype, _get_default_group

from torch.distributed.c10d_logger import _exception_logger
from torch._C._distributed_c10d import  AllreduceOptions,AllToAllOptions,AllgatherOptions,ReduceScatterOptions,ReduceOp,ProcessGroup

@_exception_logger
def all_to_all_vd(
    output,
    input,
    output_split_sizes=None,
    input_split_sizes=None,
    flag = 0,
    group=None,
    async_op=False

):
    """
    Split input tensor and then scatter the split list to all processes in a group.

    Later the received tensors are concatenated from all the processes in the group
    and returned as a single output tensor.


    Args:
        output (Tensor): 按maxsize分配；表示当前rank从其他rank收的数据.
        input (Tensor): 按maxsize分配；表示当前rank发送给其他rank的数据. max size 通过环境变量指定， 类似这样： export ECCL_ALLTOALLV_MAXSIZE=1024.
        output_split_sizes: (Tensor): 表示从其他rank收的数据；类型是int， 长度是nranks.
        input_split_sizes: (Tensor): 表示发送给所有rank的数据大小；类型是int， 长度是nranks.
        flag:表示是否需要内部做一次alltoall收集recounts信息， 如果flag=0，表示不需要；如果flag=1，表示需要.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.
     """

    if _rank_not_in_group(group):
        _warn_not_in_group("all_to_all_vd")
        return

    opts = AllToAllOptions()
    _check_single_tensor(output, "output")
    _check_single_tensor(input, "input")
    _check_single_tensor(output_split_sizes, "output_split_sizes")
    _check_single_tensor(input_split_sizes, "input_split_sizes")
    _ensure_all_tensors_same_dtype(output, input)

    if input.is_complex():
        input = torch.view_as_real(input)
    if output.is_complex():
        output = torch.view_as_real(output)

    output_split_sizes = [] if output_split_sizes is None else output_split_sizes
    input_split_sizes = [] if input_split_sizes is None else input_split_sizes

    group = group or _get_default_group()

    work = group._get_backend(input.device).alltoallv_d(
        output, input, output_split_sizes, input_split_sizes, flag, opts
    )

    if async_op:
        return work
    else:
        work.wait()


@_exception_logger
def all_reduce_outplace(output_tensor, input_tensor, op=ReduceOp.SUM, group=None, async_op=False):
    """
    Reduces the tensor data across all machines in a way that all get the final result.

    After the call ``tensor`` is going to be bitwise identical in all processes.

    Complex tensors are supported.

    Args:
        output_tensor (Tensor): output of the collective. The function
            operates out-place.
        input_tensor (Tensor): Input of the collective. The function
            operates out-place.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    _check_single_tensor(output_tensor, "tensor")
    _check_single_tensor(input_tensor, "tensor")
    if _rank_not_in_group(group):
        _warn_not_in_group("all_reduce_outplace")
        return

    opts = AllreduceOptions()
    opts.reduceOp = op
    if group is None:
        group = _get_default_group()

    work = group._get_backend(input_tensor.device).allreduce_outplace(output_tensor, input_tensor, opts)

    if async_op:
        return work
    else:
        work.wait()

def get_unique_id(group: ProcessGroup=None) -> list[int]:
    if group is None:
        group = _get_default_group()
    # _get_backend only focuses on device type
    return group._get_backend(torch.device('gcu:0')).get_unique_id()

@_exception_logger
def all_gather_into_tensor_v(output_tensor, input_tensor, recv_counts, group=None, async_op=False):
    """
    Gather tensors from all ranks and put them in a single output tensor.
    This function not requires all input tensors to be the same size on each process.

    Args:
        output_tensor (Tensor): Output tensor to accommodate tensor elements
            from all ranks. It must be correctly sized to have one of the
            following forms:
            (i) a concatenation of all the input tensors along the primary
            dimension; for definition of "concatenation", see ``torch.cat()``;
            (ii) a stack of all the input tensors along the primary dimension;
            for definition of "stack", see ``torch.stack()``.
            Examples below may better explain the supported output forms.
        input_tensor (Tensor): Tensor to be gathered from current rank.
        recv_counts (list[int]): List of sizes of tensors to be received from
            each rank in dim 0. The list must be of length equal to the world size.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """

    _check_single_tensor(input_tensor, "input_tensor")
    _check_single_tensor(output_tensor, "output_tensor")
    if _rank_not_in_group(group):
        _warn_not_in_group("all_gather_into_tensor_v")
        return

    output_tensor = (
        output_tensor
        if not output_tensor.is_complex()
        else torch.view_as_real(output_tensor)
    )
    input_tensor = (
        input_tensor
        if not input_tensor.is_complex()
        else torch.view_as_real(input_tensor)
    )

    opts = AllgatherOptions()
    opts.asyncOp = async_op

    group = group or _get_default_group()
    work = group._get_backend(input_tensor.device).allgatherv(output_tensor, input_tensor, recv_counts, opts)

    if async_op:
        return work
    else:
        work.wait()

@_exception_logger
def reduce_scatter_tensor_v(output_tensor, input_tensor, scatter_counts, op=ReduceOp.SUM, group=None, async_op=False):
    """
    Reduce tensors from all ranks and scatter the results to different ranks.
    This function allows different output sizes for each rank.

    Args:
        output_tensor (Tensor): Output tensor to store the reduced result for current rank.
            The size should match the scatter count for current rank.
        input_tensor (Tensor): Input tensor containing data to be reduced and scattered.
            It should be sized to accommodate data for all ranks concatenated.
        scatter_counts (list[int]): List of sizes of tensors to be scattered to
            each rank in dim 0. The list must be of length equal to the world size.
        op (optional): One of the values from ``torch.distributed.ReduceOp``
            enum. Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Example:
        Assume 3 ranks with scatter_counts = [2, 3, 1]
        - input_tensor on each rank should have size [6, ...] (2+3+1=6 in dim 0)
        - output_tensor on rank 0 should have size [2, ...]
        - output_tensor on rank 1 should have size [3, ...]
        - output_tensor on rank 2 should have size [1, ...]
        
        After the operation:
        - rank 0 gets the reduced result of input_tensor[0:2, ...] from all ranks
        - rank 1 gets the reduced result of input_tensor[2:5, ...] from all ranks  
        - rank 2 gets the reduced result of input_tensor[5:6, ...] from all ranks
    """

    _check_single_tensor(input_tensor, "input_tensor")
    _check_single_tensor(output_tensor, "output_tensor")
    if _rank_not_in_group(group):
        _warn_not_in_group("reduce_scatter_tensor_v")
        return

    output_tensor = (
        output_tensor
        if not output_tensor.is_complex()
        else torch.view_as_real(output_tensor)
    )
    input_tensor = (
        input_tensor
        if not input_tensor.is_complex()
        else torch.view_as_real(input_tensor)
    )

    opts = ReduceScatterOptions()
    opts.reduceOp = op
    opts.asyncOp = async_op

    group = group or _get_default_group()
    
    work = group._get_backend(input_tensor.device).reduce_scatterv(output_tensor, input_tensor, scatter_counts, opts)

    if async_op:
        return work
    else:
        work.wait()