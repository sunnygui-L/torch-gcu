.. _python_op:

####################
Python接口支持情况
####################

Supported Interfaces Mapping
===========================

.. list-table:: Supported Interfaces
   :header-rows: 1

   * - torch.gcu Interface
     - torch.cuda Interface

   * - torch.gcu.BFloat16Storage
     - torch.cuda.BFloat16Storage
   * - torch.gcu.BFloat16Tensor
     - torch.cuda.BFloat16Tensor
   * - torch.gcu.BoolStorage
     - torch.cuda.BoolStorage
   * - torch.gcu.BoolTensor
     - torch.cuda.BoolTensor
   * - torch.gcu.ByteStorage
     - torch.cuda.ByteStorage
   * - torch.gcu.ByteTensor
     - torch.cuda.ByteTensor
   * - torch.gcu.CharStorage
     - torch.cuda.CharStorage
   * - torch.gcu.CharTensor
     - torch.cuda.CharTensor
   * - torch.gcu.DoubleStorage
     - torch.cuda.DoubleStorage
   * - torch.gcu.DoubleTensor
     - torch.cuda.DoubleTensor
   * - torch.gcu.Event
     - torch.cuda.Event
   * - torch.gcu.ExternalStream
     - torch.cuda.ExternalStream
   * - torch.gcu.FloatStorage
     - torch.cuda.FloatStorage
   * - torch.gcu.FloatTensor
     - torch.cuda.FloatTensor
   * - torch.gcu.GCUGraph
     - torch.cuda.CUDAGraph
   * - torch.gcu.HalfStorage
     - torch.cuda.HalfStorage
   * - torch.gcu.HalfTensor
     - torch.cuda.HalfTensor
   * - torch.gcu.IntStorage
     - torch.cuda.IntStorage
   * - torch.gcu.IntTensor
     - torch.cuda.IntTensor
   * - torch.gcu.LongStorage
     - torch.cuda.LongStorage
   * - torch.gcu.LongTensor
     - torch.cuda.LongTensor
   * - torch.gcu.ShortStorage
     - torch.cuda.ShortStorage
   * - torch.gcu.ShortTensor
     - torch.cuda.ShortTensor
   * - torch.gcu.Stream
     - torch.cuda.Stream
   * - torch.gcu.StreamContext
     - torch.cuda.StreamContext
   * - torch.gcu.amp.GradScaler
     - torch.cuda.amp.GradScaler
   * - torch.gcu.amp.amp_definitely_not_available
     - torch.cuda.amp.amp_definitely_not_available
   * - torch.gcu.amp.autocast
     - torch.cuda.amp.autocast
   * - torch.gcu.amp.custom_bwd
     - torch.cuda.amp.custom_bwd
   * - torch.gcu.amp.custom_fwd
     - torch.cuda.amp.custom_fwd
   * - torch.gcu.caching_allocator_alloc
     - torch.cuda.caching_allocator_alloc
   * - torch.gcu.caching_allocator_delete
     - torch.cuda.caching_allocator_delete
   * - torch.gcu.can_device_access_peer
     - torch.cuda.can_device_access_peer
   * - torch.gcu.check_error
     - torch.cuda.check_error
   * - torch.gcu.current_device
     - torch.cuda.current_device
   * - torch.gcu.current_stream
     - torch.cuda.current_stream
   * - torch.gcu.default_generators
     - torch.cuda.default_generators
   * - torch.gcu.default_stream
     - torch.cuda.default_stream
   * - torch.gcu.device
     - torch.cuda.device
   * - torch.gcu.device_count
     - torch.cuda.device_count
   * - torch.gcu.device_of
     - torch.cuda.device_of
   * - torch.gcu.empty_cache
     - torch.cuda.empty_cache
   * - torch.gcu.gcuStatus
     - torch.cuda.cudaStatus
   * - torch.gcu.gcurt
     - torch.cuda.cudart
   * - torch.gcu.get_allocator_backend
     - torch.cuda.get_allocator_backend
   * - torch.gcu.get_arch_list
     - torch.cuda.get_arch_list
   * - torch.gcu.get_device_capability
     - torch.cuda.get_device_capability
   * - torch.gcu.get_device_name
     - torch.cuda.get_device_name
   * - torch.gcu.get_device_properties
     - torch.cuda.get_device_properties
   * - torch.gcu.get_gencode_flags
     - torch.cuda.get_gencode_flags
   * - torch.gcu.get_rng_state
     - torch.cuda.get_rng_state
   * - torch.gcu.get_rng_state_all
     - torch.cuda.get_rng_state_all
   * - torch.gcu.get_sync_debug_mode
     - torch.cuda.get_sync_debug_mode
   * - torch.gcu.graph
     - torch.cuda.graph
   * - torch.gcu.graph_pool_handle
     - torch.cuda.graph_pool_handle
   * - torch.gcu.has_half
     - torch.cuda.has_half
   * - torch.gcu.init
     - torch.cuda.init
   * - torch.gcu.initial_seed
     - torch.cuda.initial_seed
   * - torch.gcu.is_available
     - torch.cuda.is_available
   * - torch.gcu.is_bf16_supported
     - torch.cuda.is_bf16_supported
   * - torch.gcu.is_current_stream_capturing
     - torch.cuda.is_current_stream_capturing
   * - torch.gcu.is_initialized
     - torch.cuda.is_initialized
   * - torch.gcu.make_graphed_callables
     - torch.cuda.make_graphed_callables
   * - torch.gcu.manual_seed
     - torch.cuda.manual_seed
   * - torch.gcu.manual_seed_all
     - torch.cuda.manual_seed_all
   * - torch.gcu.max_memory_allocated
     - torch.cuda.max_memory_allocated
   * - torch.gcu.max_memory_cached
     - torch.cuda.max_memory_cached
   * - torch.gcu.max_memory_reserved
     - torch.cuda.max_memory_reserved
   * - torch.gcu.mem_get_info
     - torch.cuda.mem_get_info
   * - torch.gcu.memory.caching_allocator_alloc
     - torch.cuda.memory.caching_allocator_alloc
   * - torch.gcu.memory.caching_allocator_delete
     - torch.cuda.memory.caching_allocator_delete
   * - torch.gcu.memory.empty_cache
     - torch.cuda.memory.empty_cache
   * - torch.gcu.memory.get_allocator_backend
     - torch.cuda.memory.get_allocator_backend
   * - torch.gcu.memory.max_memory_allocated
     - torch.cuda.memory.max_memory_allocated
   * - torch.gcu.memory.max_memory_cached
     - torch.cuda.memory.max_memory_cached
   * - torch.gcu.memory.max_memory_reserved
     - torch.cuda.memory.max_memory_reserved
   * - torch.gcu.memory.mem_get_info
     - torch.cuda.memory.mem_get_info
   * - torch.gcu.memory.memory_allocated
     - torch.cuda.memory.memory_allocated
   * - torch.gcu.memory.memory_cached
     - torch.cuda.memory.memory_cached
   * - torch.gcu.memory.memory_reserved
     - torch.cuda.memory.memory_reserved
   * - torch.gcu.memory.memory_snapshot
     - torch.cuda.memory.memory_snapshot
   * - torch.gcu.memory.memory_stats
     - torch.cuda.memory.memory_stats
   * - torch.gcu.memory.memory_stats_as_nested_dict
     - torch.cuda.memory.memory_stats_as_nested_dict
   * - torch.gcu.memory.memory_summary
     - torch.cuda.memory.memory_summary
   * - torch.gcu.memory.reset_accumulated_memory_stats
     - torch.cuda.memory.reset_accumulated_memory_stats
   * - torch.gcu.memory.reset_max_memory_allocated
     - torch.cuda.memory.reset_max_memory_allocated
   * - torch.gcu.memory.reset_max_memory_cached
     - torch.cuda.memory.reset_max_memory_cached
   * - torch.gcu.memory.reset_peak_memory_stats
     - torch.cuda.memory.reset_peak_memory_stats
   * - torch.gcu.memory.set_per_process_memory_fraction
     - torch.cuda.memory.set_per_process_memory_fraction
   * - torch.gcu.memory_allocated
     - torch.cuda.memory_allocated
   * - torch.gcu.memory_cached
     - torch.cuda.memory_cached
   * - torch.gcu.memory_reserved
     - torch.cuda.memory_reserved
   * - torch.gcu.memory_snapshot
     - torch.cuda.memory_snapshot
   * - torch.gcu.memory_stats
     - torch.cuda.memory_stats
   * - torch.gcu.memory_stats_as_nested_dict
     - torch.cuda.memory_stats_as_nested_dict
   * - torch.gcu.memory_summary
     - torch.cuda.memory_summary
   * - torch.gcu.random.get_rng_state
     - torch.cuda.random.get_rng_state
   * - torch.gcu.random.get_rng_state_all
     - torch.cuda.random.get_rng_state_all
   * - torch.gcu.random.initial_seed
     - torch.cuda.random.initial_seed
   * - torch.gcu.random.manual_seed
     - torch.cuda.random.manual_seed
   * - torch.gcu.random.manual_seed_all
     - torch.cuda.random.manual_seed_all
   * - torch.gcu.random.seed
     - torch.cuda.random.seed
   * - torch.gcu.random.seed_all
     - torch.cuda.random.seed_all
   * - torch.gcu.random.set_rng_state
     - torch.cuda.random.set_rng_state
   * - torch.gcu.random.set_rng_state_all
     - torch.cuda.random.set_rng_state_all
   * - torch.gcu.reset_accumulated_memory_stats
     - torch.cuda.reset_accumulated_memory_stats
   * - torch.gcu.reset_max_memory_allocated
     - torch.cuda.reset_max_memory_allocated
   * - torch.gcu.reset_max_memory_cached
     - torch.cuda.reset_max_memory_cached
   * - torch.gcu.reset_peak_memory_stats
     - torch.cuda.reset_peak_memory_stats
   * - torch.gcu.seed
     - torch.cuda.seed
   * - torch.gcu.seed_all
     - torch.cuda.seed_all
   * - torch.gcu.set_device
     - torch.cuda.set_device
   * - torch.gcu.set_per_process_memory_fraction
     - torch.cuda.set_per_process_memory_fraction
   * - torch.gcu.set_rng_state
     - torch.cuda.set_rng_state
   * - torch.gcu.set_rng_state_all
     - torch.cuda.set_rng_state_all
   * - torch.gcu.set_stream
     - torch.cuda.set_stream
   * - torch.gcu.set_sync_debug_mode
     - torch.cuda.set_sync_debug_mode
   * - torch.gcu.stream
     - torch.cuda.stream
   * - torch.gcu.synchronize
     - torch.cuda.synchronize
   * - torch.gcu.topstx.mark
     - torch.cuda.nvtx.mark
   * - torch.gcu.topstx.range
     - torch.cuda.nvtx.range
   * - torch.gcu.topstx.range_end
     - torch.cuda.nvtx.range_end
   * - torch.gcu.topstx.range_pop
     - torch.cuda.nvtx.range_pop
   * - torch.gcu.topstx.range_push
     - torch.cuda.nvtx.range_push
   * - torch.gcu.topstx.range_start
     - torch.cuda.nvtx.range_start

.. warning::
    请尽量避免在S60系列设备上频繁调用torch.gcu.empty_cache，因为在S60系列设备上不支持释放显存时，自动进行碎片整理。
    频繁调用torch.gcu.empty_cache容易造成显存碎片化，导致OOM。