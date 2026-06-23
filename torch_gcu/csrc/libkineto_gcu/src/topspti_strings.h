/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

#include "topspti.h"

namespace libkineto_gcu {

const char* memoryKindString(Topspti_ActivityMemoryKind kind);
const char* memcpyKindString(Topspti_ActivityMemcpyKind kind);
const char* runtimeCbidName(Topspti_CallbackId cbid);
// const char* overheadKindString(Topspti_ActivityOverheadKind kind);
// const char* syncTypeString(Topspti_ActivitySynchronizationType kind);

}  // namespace libkineto_gcu
