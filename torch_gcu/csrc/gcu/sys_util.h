/*
 * Copyright 2021-2023 Enflame. All Rights Reserved.
 */
#pragma once

#include <json/json.h>
#include <unistd.h>

#include <string>

#include "gcu/gcu_macros.h"
#include "gcu/namespace.h"

GCU_TORCH_GCU_UTIL_NS_BEGIN

std::string GetEnvString(const char* name, const std::string& defval);

int64_t GetEnvInt(const char* name, int64_t defval);

double GetEnvDouble(const char* name, double defval);

bool GetEnvBoolNoPrint(const char* name, bool defval);

bool GetEnvBool(const char* name, bool defval);

// Retrieves the current EPOCH time in nanoseconds.
int64_t NowNs();

TORCH_GCU_API std::string GetTimeStamp();

std::string GetTimeStampMillis();

void CreatDirIfNotExist(const std::string& path);

void SaveJsonToFile(const std::string& file, Json::Value value);

GCU_TORCH_GCU_UTIL_NS_END