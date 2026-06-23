cmake_minimum_required(VERSION "3.21.0")

# ###############################################################################
# #########                         OPTIONS                           ###########
# ###############################################################################
add_compile_options(-w)
# ABI=0/1
set(_GLIBCXX_USE_CXX11_ABI 1)

# TODO: turn default USE_C10D_ECCL to ON
if(BUILD_LIBTORCH_GCU)
  option(USE_C10D_ECCL "USE C10D NCCL" OFF)
else()
  option(USE_C10D_ECCL "USE C10D NCCL" ON)
  add_definitions(-DUSE_C10D_ECCL)
endif()

set(BUILD_TORCH_VERSION 2.10.0)
if(BUILD_LIBTORCH_GCU)
  set(LIBTORCH_VERSION ${BUILD_TORCH_VERSION})
endif()

# jsoncpp version
set(JSONCPP_TAG 1.8.4)

if(BUILD_LIBTORCH_GCU AND NOT PACKAGE_VERSION)
  find_program(PYTHON_EXECUTABLE python3 REQUIRED)
  execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "
import importlib.util
build_utils_spec = importlib.util.spec_from_file_location(\"build_utils\", \"${CMAKE_SOURCE_DIR}/build_utils.py\")
build_utils_m = importlib.util.module_from_spec(build_utils_spec)
build_utils_spec.loader.exec_module(build_utils_m)
print(build_utils_m.get_tag('${CMAKE_SOURCE_DIR}', build_utils_m.get_tops_version('${CMAKE_SOURCE_DIR}/.version')))"
      OUTPUT_VARIABLE PACKAGE_VERSION_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REPLACE "\n" "" PACKAGE_VERSION "${PACKAGE_VERSION_OUTPUT}")
endif()

option(USE_KINETO_GCU "Use KinetoGCU profiling library" ON)
if (USE_KINETO_GCU)
  add_definitions(-DUSE_KINETO_GCU)
endif()

# message
message("\n ******** TorchGCU Build Options Summary Begin ******** \n")
message(STATUS "CMAKE_BUILD_TYPE=" ${CMAKE_BUILD_TYPE})
message(STATUS "_GLIBCXX_USE_CXX11_ABI=" ${_GLIBCXX_USE_CXX11_ABI})
message(STATUS "USE_C10D_ECCL=" ${USE_C10D_ECCL})
message(STATUS "BUILD_LIBTORCH_GCU=" ${BUILD_LIBTORCH_GCU})
message(STATUS "USE_KINETO_GCU=" ${USE_KINETO_GCU})

if(LIBTORCH_VERSION)
  message(STATUS ${LIBTORCH_VERSION})
endif()

message(STATUS "JSONCPP_TAG=" ${JSONCPP_TAG})

if(BUILD_LIBTORCH_GCU)
  message(STATUS "Found PACKAGE VERSION=" ${PACKAGE_VERSION})
endif()

if(NOT BUILD_LIBTORCH_GCU)
  message(STATUS "PYTORCH_INSTALL_DIR=" ${PYTORCH_INSTALL_DIR})
endif()

message(STATUS "Enable ICTEST_FORCE_INSTALL to auto install sanitizer libs")
set(ICTEST_FORCE_INSTALL ON)

message(" ******** TorchGCU Build Options Summary End ********")

################################################################################
##########                       DEPENDENCY                          ###########
################################################################################

include(FetchContent)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../../../../lib;$ORIGIN/lib;$ORIGIN")
set(DISABLE_GIT_SUBMODULE_CHECK ON)
################################################################################
##########                         PROJECT                           ###########
################################################################################

project(torch_gcu LANGUAGES C CXX)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CTest)

file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/lib)
set(CMAKE_FPKG_LIBDIR "lib")
set(CMAKE_FPKG_PYTHON_PACKAGES "python_packages")
enable_testing()

set(USE_CXX11_ABI_FOR_PKG ${_GLIBCXX_USE_CXX11_ABI})

set(CXX_STANDARD_REQUIRED ON)

if(BUILD_LIBTORCH_GCU)
  include(gtest)
endif()

# modify the following variables to change the prebuild caps version
file(MAKE_DIRECTORY ${CMAKE_INSTALL_LIBDIR}
    ${CMAKE_BINARY_DIR}/bin
    ${CMAKE_BINARY_DIR}/lib)

if((EXISTS "/etc/tlinux-release") AND CMAKE_CXX_COMPILER)
    set(TOPSCC_COMPIER_OPTIONS --gcc-toolchain=${CMAKE_CXX_COMPILER})
    foreach(dir ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
        list(APPEND TOPSCC_COMPIER_OPTIONS -isystem ${dir})
    endforeach()
    foreach(dir ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})
        list(APPEND TOPSCC_COMPIER_OPTIONS -L${dir} -B${dir})
    endforeach()
    set(TOPSCC_COMPIER_OPTIONS "${TOPSCC_COMPIER_OPTIONS}" CACHE STRING "" FORCE)
endif()

include(cmake/jsoncpp.cmake)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(RPATH_VALUE $ORIGIN)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${RPATH_VALUE}/lib/:${RPATH_VALUE}/:/opt/tops/lib:/opt/tops/extras/TOPSPTI/lib64")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
if(BUILD_LIBTORCH_GCU)
  set(TORCHGCU_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/libtorch_gcu/lib)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${TORCHGCU_INSTALL_LIBDIR})
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
  message(
      WARNING "C++ standard version definition detected in environment variable."
      "PyTorch requires -std=c++17. Please remove -std=c++ settings in your environment.")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TORCHGCU_ROOT "${PROJECT_SOURCE_DIR}/torch_gcu/csrc")

set(TORCHGCU_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")
set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)

################################################################################
##########                       BUILD_TYPE                          ###########
################################################################################

# Define build type
IF(CMAKE_BUILD_TYPE MATCHES Debug)
  message("Debug build.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DEBUG")
ELSEIF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  message("RelWithDebInfo build")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
ELSE()
  message("Release build.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
ENDIF()

################################################################################
##########                  COMPILE OPTIONS                          ###########
################################################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")
# Eigen fails to build with some versions, so convert this to a warning
# Details at http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1459
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
# These flags are not available in GCC-4.8.5. Set only when using clang.
# Compared against https://gcc.gnu.org/onlinedocs/gcc-4.8.5/gcc/Option-Summary.html
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-invalid-partial-specialization")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-typedef-redefinition")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-aligned-allocation-unavailable")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++17-extensions")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-constexpr-not-const")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Qunused-arguments")
  if (${COLORIZE_OUTPUT})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
  endif()
endif()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
  if (${COLORIZE_OUTPUT})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
  endif()
endif()
if ((APPLE AND (NOT ("${CLANG_VERSION_STRING}" VERSION_LESS "9.0")))
  OR (CMAKE_COMPILER_IS_GNUCXX
  AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0 AND NOT APPLE)))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-new")
endif()
if (WERROR)
  check_cxx_compiler_flag("-Werror" COMPILER_SUPPORT_WERROR)
  if (NOT COMPILER_SUPPORT_WERROR)
    set(WERROR FALSE)
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
endif(WERROR)
if (NOT APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
  #  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-uninitialized")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -O0")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-math-errno")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-trapping-math")

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX512_CPU_DEFINITION")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX2_CPU_DEFINITION")
endif()

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_SVE256_CPU_DEFINITION")
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_C_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CMAKE_CXX_FLAGS}")
  set(CXXFLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CXXFLAGS}")
else()
  set(CMAKE_C_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-s,-z,noexecstack -fPIE -pie ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-s,-z,noexecstack -fPIE -pie ${CMAKE_CXX_FLAGS}")
  set(CXXFLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-s,-z,noexecstack -fPIE -pie ${CXXFLAGS}")
endif()

################################################################################
##########                         INCLUDES                          ###########
################################################################################

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/torch_gcu)
include_directories(${PROJECT_SOURCE_DIR}/torch_gcu/csrc/)
include_directories(/opt/tops/include/logging)

include_directories(/opt/tops/include/)
include_directories(/opt/tops/include/tops/)
include_directories(/opt/tops/extras/TOPSPTI/include/)
include_directories(/usr/include/gcu)

# Set installed PyTorch dir
if(NOT BUILD_LIBTORCH_GCU)
  if(DEFINED PYTORCH_INSTALL_DIR)
    include_directories(${PYTORCH_INSTALL_DIR}/include)
    include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include)
    include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/distributed)
  else()
    message(FATAL_ERROR "Cannot find installed PyTorch directory")
  endif()
endif()

# Set Python include dir
if(NOT BUILD_LIBTORCH_GCU)
  if(DEFINED PYTHON_INCLUDE_DIR)
    include_directories(${PYTHON_INCLUDE_DIR})
  else()
    message(FATAL_ERROR "Cannot find installed Python head file directory")
  endif()
endif()

if (BUILD_LIBTORCH_GCU)
  include(torch_archive)
  set(LIBTROCH_DIR ${torch_x86_SOURCE_DIR}/libtorch)
  include_directories(${LIBTROCH_DIR}/include)
  include_directories(${LIBTROCH_DIR}/include/torch/csrc/api/include)
endif()

# ################################################################################
# ##########                         KINETO_GCU                        ###########
# ################################################################################
if (USE_KINETO_GCU)
  include(GNUInstallDirs)

  set(TOPSPTI_INCLUDE_DIR "/opt/tops/extras/TOPSPTI/include")
  include_directories(${TOPSPTI_INCLUDE_DIR})
  link_directories(/opt/tops/lib/)
  link_directories(/opt/tops/extras/TOPSPTI/lib64/)

  include_directories(${PROJECT_SOURCE_DIR}/torch_gcu/csrc/libkineto_gcu/include)

  set(LIBKINETO_GCU_DIR "${PROJECT_SOURCE_DIR}/torch_gcu/csrc/libkineto_gcu/")
  set(LIBKINETO_GCU_SOURCE_DIR "${LIBKINETO_GCU_DIR}/src")
  set(LIBKINETO_GCU_INCLUDE_DIR "${LIBKINETO_GCU_DIR}/include")
  set(LIBKINETO_GCU_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

  file(GLOB _LIBKINETO_GCU_SRCS ${LIBKINETO_GCU_DIR}/src/*.cc ${LIBKINETO_GCU_DIR}/src/*.cpp)
  list(APPEND LIBKINETO_GCU_SRCS ${_LIBKINETO_GCU_SRCS})

  set(KINETO_GCU_COMPILE_OPTIONS "-DKINETO_GCU_NAMESPACE=libkineto_gcu")
  list(APPEND KINETO_GCU_COMPILE_OPTIONS "-DFMT_HEADER_ONLY")
  list(APPEND KINETO_GCU_COMPILE_OPTIONS "-DENABLE_IPC_FABRIC")
  if(NOT MSVC)
    # list(APPEND KINETO_GCU_COMPILE_OPTIONS "-std=c++17")
  else()
    # list(APPEND KINETO_GCU_COMPILE_OPTIONS "/std:c++17")
    list(APPEND KINETO_GCU_COMPILE_OPTIONS "-DWIN32_LEAN_AND_MEAN")
    list(APPEND KINETO_GCU_COMPILE_OPTIONS "-DNOGDI")
  endif()
  list(APPEND KINETO_GCU_COMPILE_OPTIONS "-DHAS_TOPSPTI")

  if(NOT TARGET fmt)
    if(NOT FMT_SOURCE_DIR)
      set(FMT_SOURCE_DIR "/opt/fmt"
        CACHE STRING "fmt source directory from FetchContent")
    endif()

    # Build FMT.
    # FMT and some other libraries use BUILD_SHARED_LIBS to control
    # the library type.
    # Save and restore the value after configuring FMT
    set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
    set(FMT_LIBRARY_TYPE static CACHE STRING "Set lib type to static")
    add_subdirectory("${FMT_SOURCE_DIR}" "${LIBKINETO_GCU_BINARY_DIR}/fmt")
    set_property(TARGET fmt PROPERTY POSITION_INDEPENDENT_CODE ON)
    set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)
  endif()

  set(FMT_INCLUDE_DIR "${FMT_SOURCE_DIR}/include")
  message(STATUS "KinetoGCU: FMT_SOURCE_DIR = ${FMT_SOURCE_DIR}")
  message(STATUS "KinetoGCU: FMT_INCLUDE_DIR = ${FMT_INCLUDE_DIR}")

  include_directories(${FMT_INCLUDE_DIR})
  include_directories(${TOPSPTI_INCLUDE_DIR})
  include_directories(${LIBKINETO_GCU_INCLUDE_DIR})
  include_directories(${LIBKINETO_GCU_SOURCE_DIR})

  message(INFO " TOPSPTI_INCLUDE_DIR = ${TOPSPTI_INCLUDE_DIR}")

  add_library(kineto_gcu SHARED ${LIBKINETO_GCU_SRCS})

  set(GCU_topspti_LIBRARY topspti)
  target_link_options(kineto_gcu PUBLIC -Wl,-rpath,/opt/tops/extras/TOPSPTI/lib64)
  target_link_libraries(kineto_gcu "${GCU_topspti_LIBRARY}")
  target_link_libraries(kineto_gcu $<BUILD_INTERFACE:fmt::fmt-header-only>)
  add_dependencies(kineto_gcu fmt::fmt-header-only)

  target_compile_options(kineto_gcu PRIVATE "${KINETO_GCU_COMPILE_OPTIONS}")
  target_compile_options(kineto_gcu PRIVATE -U_GLIBCXX_USE_CXX11_ABI -fPIC -D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
  target_compile_options(kineto_gcu PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
endif()

################################################################################
##########                         SOURCES                           ###########
################################################################################

set(GCU_SRCS)
set(ATEN_SRCS)
set(COMPILE_SRCS)
set(AOTFUSION_SRCS)
set(DIST_SRCS)
set(PROF_SRCS)
set(PYTHON_SRCS)
set(EFFICIENT_SRCS)

if(BUILD_LIBTORCH_GCU)
  set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/libtorch_gcu/include/)
else()
  set(INSTALL_INCLUDE_DIR ${CMAKE_BINARY_DIR}/packages/torch_gcu/include/)
endif()

add_subdirectory(${TORCHGCU_ROOT}/gcu)
add_subdirectory(${TORCHGCU_ROOT}/aten)
add_subdirectory(${TORCHGCU_ROOT}/aotfusion)
if(USE_KINETO_GCU)
  add_subdirectory(${TORCHGCU_ROOT}/profiler)
endif()
add_subdirectory(${TORCHGCU_ROOT}/efficient_ops)

if(USE_C10D_ECCL)
  add_subdirectory(${TORCHGCU_ROOT}/distributed)
endif()

add_subdirectory(${TORCHGCU_ROOT}/python)

install(FILES ${TORCHGCU_ROOT}/torch_gcu.h
  DESTINATION ${INSTALL_INCLUDE_DIR}
)

if(USE_C10D_ECCL)
  set(LIBTORCH_GCU_SRCS ${GCU_SRCS} ${ATEN_SRCS} ${DIST_SRCS} ${COMPILE_SRCS} ${AOTFUSION_SRCS} ${EFFICIENT_SRCS})
else()
  set(LIBTORCH_GCU_SRCS ${GCU_SRCS} ${ATEN_SRCS} ${COMPILE_SRCS} ${AOTFUSION_SRCS} ${EFFICIENT_SRCS})
endif()


if(USE_KINETO_GCU)
  set(LIBTORCH_GCU_PYTHON_SRCS ${PYTHON_SRCS} ${PROF_SRCS})
else()
  set(LIBTORCH_GCU_PYTHON_SRCS ${PYTHON_SRCS})
endif()

# ###############################################################################
# #########                         LIBRARY                           ###########
# ###############################################################################
if(NOT BUILD_LIBTORCH_GCU)
  link_directories(${PYTORCH_INSTALL_DIR}/lib)
endif()
link_directories(/opt/tops/lib)
link_directories(/opt/tops/extras/TOPSPTI/lib64/)

if (BUILD_LIBTORCH_GCU)
  link_directories(${LIBTROCH_DIR}/lib)
endif()

set(PLUGIN_NAME torch_gcu)

# libtorch_gcu.so
add_library(${PLUGIN_NAME} SHARED ${LIBTORCH_GCU_SRCS})

if(USE_C10D_ECCL)
  target_link_libraries(${PLUGIN_NAME} PUBLIC topsrt eccl topsaten topstx)
else()
  target_link_libraries(${PLUGIN_NAME} PUBLIC topsrt topsaten topstx)
endif()

if(USE_C10D_ECCL)
  target_compile_definitions(${PLUGIN_NAME} PRIVATE USE_C10D_ECCL)
endif()

target_link_libraries(${PLUGIN_NAME} PUBLIC ilog_abi1)
target_link_libraries(${PLUGIN_NAME} PUBLIC jsoncpp_static)
target_link_libraries(${PLUGIN_NAME} PUBLIC torch torch_cpu c10 fmt::fmt-header-only)

if(NOT BUILD_LIBTORCH_GCU)
  # libtorch_gcu_python.so
  set(PLUGIN_PY_NAME ${PLUGIN_NAME}_python)
  add_library(${PLUGIN_PY_NAME} SHARED ${LIBTORCH_GCU_PYTHON_SRCS})
  target_link_libraries(${PLUGIN_PY_NAME} PUBLIC ${PLUGIN_NAME})
  target_link_libraries(${PLUGIN_PY_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libtorch_python.so)
  if (USE_KINETO_GCU)
    target_link_libraries(${PLUGIN_PY_NAME} PUBLIC kineto_gcu)
  endif()
endif()


set(ATEN_THREADING "OMP" CACHE STRING "ATen parallel backend")
message(STATUS "Using ATen parallel backend: ${ATEN_THREADING}")
if ("${ATEN_THREADING}" STREQUAL "OMP")
  target_compile_definitions(${PLUGIN_NAME} PUBLIC "-DAT_PARALLEL_OPENMP=1")
elseif ("${ATEN_THREADING}" STREQUAL "NATIVE")
  target_compile_definitions(${PLUGIN_NAME} PUBLIC "-DAT_PARALLEL_NATIVE=1")
elseif ("${ATEN_THREADING}" STREQUAL "TBB")
  target_compile_definitions(${PLUGIN_NAME} PUBLIC "-DAT_PARALLEL_NATIVE_TBB=1")
else()
  message(FATAL_ERROR "Unknown ATen parallel backend: ${ATEN_THREADING}")
endif()

include(GNUInstallDirs)
target_compile_options(${PLUGIN_NAME} PRIVATE "-DC10_BUILD_MAIN_LIB")
# install(TARGETS ${PLUGIN_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

target_compile_options(${PLUGIN_NAME} PRIVATE -U_GLIBCXX_USE_CXX11_ABI -fPIC -D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
target_compile_options(${PLUGIN_NAME} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

if(NOT BUILD_LIBTORCH_GCU)
  target_compile_options(${PLUGIN_PY_NAME} PRIVATE -U_GLIBCXX_USE_CXX11_ABI -fPIC -D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
endif()

if(BUILD_LIBTORCH_GCU)
  set(PACKAGE_ABI_FLAG shared-with-deps)
  set(torch_gcu_zip_full_name libtorch_gcu-${PACKAGE_ABI_FLAG}-${LIBTORCH_VERSION}+${PACKAGE_VERSION}.zip)
  message(WARNING ${CMAKE_INSTALL_PREFIX})
  add_custom_target(
    package_libtorch_gcu_zip
    COMMAND cd ${CMAKE_INSTALL_PREFIX} && zip -r ${torch_gcu_zip_full_name} libtorch_gcu
  )
endif()

# ###############################################################################
# #########           EXTERNAL PROJECTS                               ###########
# ###############################################################################
if(BUILD_LIBTORCH_GCU)
  set(TORCH_GCU_VERSION ${PACKAGE_VERSION})

  # CMake config for external projects.
  configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/TorchGCUConfigVersion.cmake.in
    ${PROJECT_BINARY_DIR}/TorchGCUConfigVersion.cmake
    @ONLY)

  configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/TorchGCUConfig.cmake.in
    ${PROJECT_BINARY_DIR}/TorchGCUConfig.cmake
    @ONLY)

  install(FILES
    ${PROJECT_BINARY_DIR}/TorchGCUConfigVersion.cmake
    ${PROJECT_BINARY_DIR}/TorchGCUConfig.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/libtorch_gcu/share/cmake/TorchGCU)
endif()
