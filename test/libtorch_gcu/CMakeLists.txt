cmake_minimum_required(VERSION 3.18)
project(libtorch_gcu_test CXX)

set(CMAKE_CXX_STANDARD 17)

# Skip CUDA detection (GCU environment)
set(CAFFE2_USE_CUDNN OFF)
set(USE_CUDA OFF)
set(CMAKE_DISABLE_FIND_PACKAGE_CUDA TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit TRUE)

find_package(Torch REQUIRED)

# torch_gcu include dir
if(NOT DEFINED TORCHGCU_INCLUDE_DIR)
    set(TORCHGCU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../torch_gcu/csrc"
        CACHE PATH "torch_gcu include dir (contains torch_gcu.h)")
endif()

# torch_gcu library dir
if(NOT DEFINED TORCHGCU_LIB_DIR)
    execute_process(
        COMMAND python3 -c "import torch_gcu; print(torch_gcu.__path__[0] + '/lib')"
        OUTPUT_VARIABLE _TORCHGCU_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_TORCHGCU_LIB_DIR)
        set(TORCHGCU_LIB_DIR "${_TORCHGCU_LIB_DIR}" CACHE PATH "torch_gcu lib dir")
    else()
        message(FATAL_ERROR "Cannot detect torch_gcu lib dir. Set -DTORCHGCU_LIB_DIR=...")
    endif()
endif()

# GTest: try system first, then FetchContent as fallback
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Try manual detection
    find_library(GTEST_LIB gtest PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/lib)
    if(GTEST_LIB)
        find_library(GTEST_MAIN_LIB gtest_main PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/lib)
        set(GTEST_BOTH_LIBRARIES ${GTEST_LIB} ${GTEST_MAIN_LIB})
        message(STATUS "GTest found manually: ${GTEST_LIB}")
    else()
        # Download GTest via FetchContent
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
            GIT_SHALLOW TRUE
        )
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        set(GTest_FOUND TRUE)
        message(STATUS "GTest fetched via FetchContent")
    endif()
endif()

message(STATUS "TORCHGCU_INCLUDE_DIR = ${TORCHGCU_INCLUDE_DIR}")
message(STATUS "TORCHGCU_LIB_DIR = ${TORCHGCU_LIB_DIR}")

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc" "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(TEST_SOURCE IN LISTS TEST_SOURCES)
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PRIVATE
        ${TORCHGCU_INCLUDE_DIR}
        /opt/tops/include
        /opt/tops/include/gcu
        /usr/include/gcu
    )
    target_link_directories(${TEST_NAME} PRIVATE ${TORCHGCU_LIB_DIR} /opt/tops/lib)

    if(TARGET GTest::gtest)
        target_link_libraries(${TEST_NAME} ${TORCH_LIBRARIES} torch_gcu GTest::gtest GTest::gtest_main pthread)
    elseif(TARGET gtest)
        target_link_libraries(${TEST_NAME} ${TORCH_LIBRARIES} torch_gcu gtest gtest_main pthread)
    else()
        target_link_libraries(${TEST_NAME} ${TORCH_LIBRARIES} torch_gcu ${GTEST_BOTH_LIBRARIES} pthread)
    endif()

    target_compile_options(${TEST_NAME} PRIVATE -U_GLIBCXX_USE_CXX11_ABI -D_GLIBCXX_USE_CXX11_ABI=1)
endforeach()
