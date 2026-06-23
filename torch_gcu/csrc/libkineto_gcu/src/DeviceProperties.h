/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include <stdint.h>

#include <string>

#include "topspti.h"

namespace libkineto_gcu {

// Return compute properties for each device as a json string
const std::string& devicePropertiesJson();

int smCount(uint32_t deviceId);

float blocksPerSm(const Topspti_ActivityKernel& kernel);
float warpsPerSm(const Topspti_ActivityKernel& kernel);

// Return estimated achieved occupancy for a kernel
float kernelOccupancy(const Topspti_ActivityKernel& kernel);
float kernelOccupancy(uint32_t deviceId, uint16_t registersPerThread,
                      int32_t staticSharedMemory, int32_t dynamicSharedMemory,
                      int32_t blockX, int32_t blockY, int32_t blockZ,
                      float blocks_per_sm);

}  // namespace libkineto_gcu
