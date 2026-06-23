/*
 * Copyright 2021-2023 Enflame. All Rights Reserved.
 */
#pragma once

#define GCU_TORCH_GCU_NS_BEGIN namespace torch_gcu {
#define GCU_UTIL_NS_BEGIN namespace util {
#define GCU_NS_BEGIN namespace {
#define GCU_NS_END }

#define GCU_TORCH_GCU_UTIL_NS_BEGIN \
  GCU_TORCH_GCU_NS_BEGIN;           \
  GCU_UTIL_NS_BEGIN

#define GCU_TORCH_GCU_UTIL_NS_END \
  GCU_NS_END;                     \
  GCU_NS_END
