/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#include "topspti_strings.h"

namespace libkineto_gcu {

const char* memcpyKindString(Topspti_ActivityMemcpyKind kind) {
  switch (kind) {
    case TOPSPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "HtoD";
    case TOPSPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "DtoH";
      // case TOPSPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      //   return "HtoA";
      // case TOPSPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      //   return "AtoH";
      // case TOPSPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      //   return "AtoA";
      // case TOPSPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      //   return "AtoD";
      // case TOPSPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "DtoA";
    case TOPSPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "DtoD";
    case TOPSPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "HtoH";
    case TOPSPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "PtoP";
    default:
      break;
  }
  return "<unknown>";
}

const char* memoryKindString(Topspti_ActivityMemoryKind kind) {
  switch (kind) {
    case TOPSPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
      return "Unknown";
    case TOPSPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
      return "Pageable";
    case TOPSPTI_ACTIVITY_MEMORY_KIND_PINNED:
      return "Pinned";
    case TOPSPTI_ACTIVITY_MEMORY_KIND_DEVICE:
      return "Device";
    // case TOPSPTI_ACTIVITY_MEMORY_KIND_ARRAY:
    //   return "Array";
    // case TOPSPTI_ACTIVITY_MEMORY_KIND_MANAGED:
    //   return "Managed";
    // case TOPSPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
    //   return "Device Static";
    // case TOPSPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
    //   return "Managed Static";
    case TOPSPTI_ACTIVITY_MEMORY_KIND_FORCE_INT:
      return "Force Int";
    default:
      return "Unrecognized";
  }
}

// const char* overheadKindString(Topspti_ActivityOverheadKind kind) {
//   switch (kind) {
//     case TOPSPTI_ACTIVITY_OVERHEAD_UNKNOWN:
//       return "Unknown";
//     case TOPSPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
//       return "Driver Compiler";
//     case TOPSPTI_ACTIVITY_OVERHEAD_TOPSPTI_BUFFER_FLUSH:
//       return "Buffer Flush";
//     case TOPSPTI_ACTIVITY_OVERHEAD_TOPSPTI_INSTRUMENTATION:
//       return "Instrumentation";
//     case TOPSPTI_ACTIVITY_OVERHEAD_TOPSPTI_RESOURCE:
//       return "Resource";
//     case TOPSPTI_ACTIVITY_OVERHEAD_FORCE_INT:
//       return "Force Int";
//     default:
//       return "Unrecognized";
//   }
// }

static const char* runtimeCbidNames[] = {"INVALID",
                                         "topsMemset",
                                         "topsMemsetAsync",
                                         "topsMemsetD8",
                                         "topsMemsetD8Async",
                                         "topsMemsetD16",
                                         "topsMemsetD16Async",
                                         "topsMemsetD32",
                                         "topsMemsetD32Async",
                                         "topsMemcpy",
                                         "topsMemcpyAsync",
                                         "topsMemcpyWithStream",
                                         "topsMemcpyHtoD",
                                         "topsMemcpyDtoH",
                                         "topsMemcpyDtoD",
                                         "topsMemcpyHtoDAsync",
                                         "topsMemcpyDtoHAsync",
                                         "topsMemcpyDtoDAsync",
                                         "topsMemcpyToSymbol",
                                         "topsMemcpyToSymbolAsync",
                                         "topsMemcpyFromSymbol",
                                         "topsMemcpyFromSymbolAsync",
                                         "topsModuleLaunchKernel",
                                         "topsModuleLaunchKernelEx",
                                         "topsLaunchKernel",
                                         "topsLaunchKernelExC",
                                         "topsLaunchCooperativeKernel",
                                         "topsGraphInstantiate",
                                         "topsGraphLaunch",
                                         "topsGraphExecDestroy",
                                         "topsGraphDestroy",
                                         "topsGraphDebugDotPrint",
                                         "topsGraphInstantiateWithFlags",
                                         "topsDriverGetVersion",
                                         "topsRuntimeGetVersion",
                                         "topsDeviceGet",
                                         "topsDeviceComputeCapability",
                                         "topsDeviceGetName",
                                         "topsDeviceTotalMem",
                                         "topsDeviceSynchronize",
                                         "topsDeviceReset",
                                         "topsSetDevice",
                                         "topsGetDevice",
                                         "topsGetDeviceCount",
                                         "topsDeviceGetAttribute",
                                         "topsGetDeviceProperties",
                                         "topsDeviceSetLimit",
                                         "topsDeviceGetLimit",
                                         "topsGetDeviceFlags",
                                         "topsSetDeviceFlags",
                                         "topsStreamCreate",
                                         "topsStreamCreateWithFlags",
                                         "topsStreamDestroy",
                                         "topsStreamGetId",
                                         "topsStreamSynchronize",
                                         "topsStreamWaitEvent",
                                         "topsEventCreateWithFlags",
                                         "topsEventCreate",
                                         "topsEventRecord",
                                         "topsEventDestroy",
                                         "topsEventSynchronize",
                                         "topsMalloc",
                                         "topsHostMalloc",
                                         "topsHostGetDevicePointer",
                                         "topsHostRegister",
                                         "topsHostUnregister",
                                         "topsFree",
                                         "topsHostFree",
                                         "topsMemPtrGetInfo",
                                         "topsModuleLoadData",
                                         "topsModuleLoad",
                                         "topsModuleGetFunction",
                                         "topsIpcOpenEventHandle",
                                         "topsIpcGetEventHandle",
                                         "topsMemGetInfo",
                                         "topsGetErrorString",
                                         "topsStreamGetLaunchLimit",
                                         "topsStreamSetLaunchLimit",
                                         "SIZE",
                                         "FORCE_INT = 0x7fffffff"};

const char* runtimeCbidName(Topspti_CallbackId cbid) {
  constexpr int names_size =
      sizeof(runtimeCbidNames) / sizeof(runtimeCbidNames[0]);
  if (cbid < 0 || cbid >= names_size) {
    return runtimeCbidNames[TOPSPTI_RUNTIME_TRACE_CBID_INVALID];
  }
  return runtimeCbidNames[cbid];
}

// From
// https://docs.nvidia.com/topspti/modules.html#group__TOPSPTI__ACTIVITY__API_1g80e1eb47615e31021f574df8ebbe5d9a
//   enum Topspti_ActivitySynchronizationType
// const char* syncTypeString(Topspti_ActivitySynchronizationType kind) {
//   switch (kind) {
//     case TOPSPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE:
//       return "Event Sync";
//     case TOPSPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT:
//       return "Stream Wait Event";
//     case TOPSPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE:
//       return "Stream Sync";
//     case TOPSPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE:
//       return "Context Sync";
//     case TOPSPTI_ACTIVITY_SYNCHRONIZATION_TYPE_UNKNOWN:
//     default:
//       return "Unknown Sync";
//   }
//   return "<unknown>";
// }
}  // namespace libkineto_gcu
