/*
 * Copyright 2023-2025 Enflame. All Rights Reserved.
 */

#pragma once

// See c10/macros/Export.h for a detailed explanation of what the function
// of these macros are.  We need one set of macros for every separate library
// we build.

#ifdef _WIN32
#if defined(C10_GCU_BUILD_SHARED_LIBS)
#define C10_GCU_EXPORT __declspec(dllexport)
#define C10_GCU_IMPORT __declspec(dllimport)
#else
#define C10_GCU_EXPORT
#define C10_GCU_IMPORT
#endif
#else  // _WIN32
#if defined(__GNUC__)
#define C10_GCU_EXPORT __attribute__((__visibility__("default")))
#else  // defined(__GNUC__)
#define C10_GCU_EXPORT
#endif  // defined(__GNUC__)
#define C10_GCU_IMPORT C10_GCU_EXPORT
#endif  // _WIN32

#ifdef C10_GCU_BUILD_MAIN_LIB
#define C10_GCU_API C10_GCU_EXPORT
#else
#define C10_GCU_API C10_GCU_IMPORT
#endif

#define TORCH_GCU_API C10_GCU_API

/**
 * The maximum number of GCUs that we recognizes.
 */
#define C10_COMPILE_TIME_MAX_GCUS 16
