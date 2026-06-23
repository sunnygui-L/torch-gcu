/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#include "DeviceProperties.h"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tops_runtime.h>

#include <vector>

#include "Logger.h"

namespace libkineto_gcu {

#define topsDeviceProp_t topsDeviceProp_t
#define gcuError_t topsError_t
#define gcuSuccess topsSuccess
#define gcuGetDeviceCount topsGetDeviceCount
#define gcuGetDeviceProperties topsGetDeviceProperties

static const std::vector<topsDeviceProp_t> createDeviceProps() {
  std::vector<topsDeviceProp_t> props;
  int device_count;
  gcuError_t error_id = gcuGetDeviceCount(&device_count);
  // Return empty vector if error.
  if (error_id != gcuSuccess) {
    LOG(ERROR) << "gcuGetDeviceCount failed with code " << error_id;
    return {};
  }
  VLOG(0) << "Device count is " << device_count;
  for (size_t i = 0; i < device_count; ++i) {
    topsDeviceProp_t prop;
    error_id = gcuGetDeviceProperties(&prop, i);
    // Return empty vector if any device property fail to get.
    if (error_id != gcuSuccess) {
      LOG(ERROR) << "gcuGetDeviceProperties failed with " << error_id;
      return {};
    }
    props.push_back(prop);
    LOGGER_OBSERVER_ADD_DEVICE(i);
  }
  return props;
}

static const std::vector<topsDeviceProp_t>& deviceProps() {
  static const std::vector<topsDeviceProp_t> props = createDeviceProps();
  return props;
}

static const std::string createDevicePropertiesJson(
    size_t id, const topsDeviceProp_t& props) {
  std::string gcuSpecific = "";
  gcuSpecific = fmt::format(R"JSON(
      , "sipCount": {}, "sharedMemPerBlock": {},
      "maxSharedMemoryPerMultiProcessor": {})JSON",
                            props.sipCount, props.sharedMemPerBlock,
                            props.maxSharedMemoryPerMultiProcessor);

  return fmt::format(R"JSON(
      {{
        "id": {}, "name": "{}", "totalGlobalMem": {},
        "computeMajor": {}, "computeMinor": {},
        "maxThreadsPerBlock": {}, "maxThreadsPerMultiprocessor": {},
       
        "sharedMemPerBlock": {}, "numSms": {}{}
      }})JSON",
                     id, props.name, props.totalGlobalMem, props.major,
                     props.minor, props.maxThreadsPerBlock,
                     props.maxThreadsPerMultiProcessor, props.sharedMemPerBlock,
                     props.multiProcessorCount, gcuSpecific);
}

static const std::string createDevicePropertiesJson() {
  std::vector<std::string> jsonProps;
  const auto& props = deviceProps();
  for (size_t i = 0; i < props.size(); i++) {
    jsonProps.push_back(createDevicePropertiesJson(i, props[i]));
  }
  return fmt::format("{}", fmt::join(jsonProps, ","));
}

const std::string& devicePropertiesJson() {
  static std::string devicePropsJson = createDevicePropertiesJson();
  return devicePropsJson;
}

int smCount(uint32_t deviceId) {
  const std::vector<topsDeviceProp_t>& props = deviceProps();
  return deviceId >= props.size() ? 0 : props[deviceId].multiProcessorCount;
}

float blocksPerSm(const Topspti_ActivityKernel& kernel) {
  return (kernel.gridX * kernel.gridY * kernel.gridZ) /
         (float)smCount(kernel.deviceId);
}

float warpsPerSm(const Topspti_ActivityKernel& kernel) {
  constexpr int threads_per_warp = 32;
  return blocksPerSm(kernel) * (kernel.blockX * kernel.blockY * kernel.blockZ) /
         threads_per_warp;
}

float kernelOccupancy(const Topspti_ActivityKernel& kernel) {
  float blocks_per_sm = -1.0;
  int sm_count = smCount(kernel.deviceId);
  if (sm_count) {
    blocks_per_sm =
        (kernel.gridX * kernel.gridY * kernel.gridZ) / (float)sm_count;
  }
  //   return kernelOccupancy(kernel.deviceId, kernel.registersPerThread,
  //                          kernel.staticSharedMemory,
  //                          kernel.dynamicSharedMemory, kernel.blockX,
  //                          kernel.blockY, kernel.blockZ, blocks_per_sm);
  return 0;
}

float kernelOccupancy(uint32_t deviceId, uint16_t registersPerThread,
                      int32_t staticSharedMemory, int32_t dynamicSharedMemory,
                      int32_t blockX, int32_t blockY, int32_t blockZ,
                      float blocksPerSm) {
  // Calculate occupancy
  float occupancy = -1.0;
  const std::vector<topsDeviceProp_t>& props = deviceProps();
  //   if (deviceId < props.size()) {
  //     topsOccFuncAttributes occFuncAttr;
  //     occFuncAttr.maxThreadsPerBlock = INT_MAX;
  //     occFuncAttr.numRegs = registersPerThread;
  //     occFuncAttr.sharedSizeBytes = staticSharedMemory;
  //     occFuncAttr.partitionedGCConfig = PARTITIONED_GC_OFF;
  //     occFuncAttr.shmemLimitConfig = FUNC_SHMEM_LIMIT_DEFAULT;
  //     occFuncAttr.maxDynamicSharedSizeBytes = 0;
  //     const topsOccDeviceState occDeviceState = {};
  //     int blockSize = blockX * blockY * blockZ;
  //     size_t dynamicSmemSize = dynamicSharedMemory;
  //     topsOccResult occ_result;
  //     topsOccDeviceProp prop(props[deviceId]);
  //     topsOccError status = topsOccMaxActiveBlocksPerMultiprocessor(
  //         &occ_result, &prop, &occFuncAttr, &occDeviceState, blockSize,
  //         dynamicSmemSize);
  //     if (status == GCU_OCC_SUCCESS) {
  //       if (occ_result.activeBlocksPerMultiprocessor < blocksPerSm) {
  //         blocksPerSm = occ_result.activeBlocksPerMultiprocessor;
  //       }
  //       occupancy = blocksPerSm * blockSize /
  //                   (float)props[deviceId].maxThreadsPerMultiProcessor;
  //     } else {
  //       LOG_EVERY_N(ERROR, 1000)
  //           << "Failed to calculate occupancy, status = " << status;
  //     }
  //   }
  return occupancy;
}

}  // namespace libkineto_gcu
