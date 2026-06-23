#!/bin/bash
#
# Copyright 2023-2025 Enflame. All Rights Reserved.
#
# Run all torch_gcu tests: op tests, libtorch C++ tests, CUDA-compat API tests.
#
# Usage:
#   bash run_all_test.sh              # Run all tests
#   bash run_all_test.sh op           # Run only op tests
#   bash run_all_test.sh libtorch     # Run only libtorch C++ tests
#   bash run_all_test.sh cuda         # Run only CUDA-compat API tests
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
SKIPPED=0
FAILED_TESTS=""

run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local test_dir="$3"
    local log_name="${test_name//\//_}"

    echo -e "${YELLOW}[RUN]${NC} $test_name"
    pushd "$test_dir" > /dev/null 2>&1 || true

    if eval "$test_cmd" > "/tmp/torch_gcu_test_${log_name}.log" 2>&1; then
        echo -e "${GREEN}[PASS]${NC} $test_name"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}[FAIL]${NC} $test_name (see /tmp/torch_gcu_test_${log_name}.log)"
        FAILED=$((FAILED + 1))
        FAILED_TESTS="$FAILED_TESTS\n  - $test_name"
    fi

    popd > /dev/null 2>&1 || true
}

# =============================================================================
# Op Tests (Python)
# =============================================================================
run_op_tests() {
    echo ""
    echo "=============================="
    echo "  Op Tests (Python)"
    echo "=============================="

    local op_dir="$SCRIPT_DIR/op_test"
    if [ ! -d "$op_dir" ]; then
        echo -e "${YELLOW}[SKIP]${NC} op_test directory not found"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    for test_file in "$op_dir"/test_*.py; do
        if [ -f "$test_file" ]; then
            local test_name="op_test/$(basename "$test_file" .py)"
            run_test "$test_name" "python3 $(basename "$test_file")" "$op_dir"
        fi
    done
}

# =============================================================================
# Libtorch C++ Tests
# =============================================================================
run_libtorch_tests() {
    echo ""
    echo "=============================="
    echo "  Libtorch C++ Tests"
    echo "=============================="

    local libtorch_dir="$SCRIPT_DIR/libtorch_gcu"
    local build_dir="$libtorch_dir/build"
    local project_root="$SCRIPT_DIR/.."

    # Source tree headers: torch_gcu/csrc/ contains torch_gcu.h and gcu/ sub-headers
    local torchgcu_src_include="$project_root/torch_gcu/csrc"

    # Detect installed packages
    local torch_cmake_dir=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || echo "")
    local torch_lib_dir=$(python3 -c "import torch; print(torch.__path__[0] + '/lib')" 2>/dev/null || echo "")
    local torch_gcu_lib_dir=$(python3 -c "import torch_gcu; print(torch_gcu.__path__[0] + '/lib')" 2>/dev/null || echo "")

    # Setup LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="${torch_gcu_lib_dir}:${torch_lib_dir}:/opt/tops/lib:/opt/tops/extras/TOPSPTI/lib64:${LD_LIBRARY_PATH}"

    # Check for pre-built test binaries
    local bin_dir=""
    for sp in "$build_dir" "$libtorch_dir"; do
        if [ -f "$sp/test_resnet50" ]; then
            bin_dir="$sp"
            echo -e "${GREEN}[OK]${NC} Found pre-built binaries: $bin_dir"
            break
        fi
    done

    # If not found, compile using CMakeLists.txt in libtorch_gcu/
    if [ -z "$bin_dir" ]; then
        echo -e "${YELLOW}[INFO]${NC} Compiling libtorch tests..."

        if [ -z "$torch_cmake_dir" ]; then
            echo -e "${RED}[SKIP]${NC} PyTorch not installed"
            SKIPPED=$((SKIPPED + 1))
            return
        fi

        if [ -z "$torch_gcu_lib_dir" ] || [ ! -d "$torch_gcu_lib_dir" ]; then
            echo -e "${RED}[SKIP]${NC} torch_gcu not installed (pip install torch_gcu-*.whl)"
            SKIPPED=$((SKIPPED + 1))
            return
        fi

        if [ ! -f "$torchgcu_src_include/torch_gcu.h" ]; then
            echo -e "${RED}[SKIP]${NC} Headers not found: $torchgcu_src_include/torch_gcu.h"
            SKIPPED=$((SKIPPED + 1))
            return
        fi

        if [ ! -f "$libtorch_dir/CMakeLists.txt" ]; then
            echo -e "${RED}[SKIP]${NC} CMakeLists.txt not found in $libtorch_dir"
            SKIPPED=$((SKIPPED + 1))
            return
        fi

        # Ensure GTest is installed
        if [ ! -f "/usr/local/lib/libgtest.a" ] && [ ! -f "/usr/lib/x86_64-linux-gnu/libgtest.a" ]; then
            echo -e "${YELLOW}[INFO]${NC} Installing GTest..."
            apt-get install -y libgtest-dev > /dev/null 2>&1 || true

            # libgtest-dev may only install source on some Ubuntu; compile it
            if [ ! -f "/usr/lib/x86_64-linux-gnu/libgtest.a" ]; then
                local gtest_src=""
                for d in "/usr/src/googletest" "/usr/src/gtest"; do
                    [ -d "$d" ] && gtest_src="$d" && break
                done
                if [ -n "$gtest_src" ]; then
                    echo -e "${YELLOW}[INFO]${NC} Building GTest from $gtest_src..."
                    local tmpdir="/tmp/gtest_build_$$"
                    mkdir -p "$tmpdir"
                    pushd "$tmpdir" > /dev/null
                    cmake "$gtest_src" -DCMAKE_INSTALL_PREFIX=/usr/local > /dev/null 2>&1 || true
                    make -j$(nproc) > /dev/null 2>&1 || true
                    make install > /dev/null 2>&1 || true
                    popd > /dev/null
                    rm -rf "$tmpdir"
                fi
            fi
        fi

        # Build tests
        mkdir -p "$build_dir"
        pushd "$build_dir" > /dev/null

        echo "  Headers:  $torchgcu_src_include"
        echo "  Library:  $torch_gcu_lib_dir"

        if ! cmake "$libtorch_dir" \
            -DCMAKE_PREFIX_PATH="$torch_cmake_dir" \
            -DTORCHGCU_INCLUDE_DIR="$torchgcu_src_include" \
            -DTORCHGCU_LIB_DIR="$torch_gcu_lib_dir" \
            > /tmp/torch_gcu_test_libtorch_build.log 2>&1; then
            echo -e "${RED}[SKIP]${NC} CMake configure failed:"
            cat /tmp/torch_gcu_test_libtorch_build.log
            popd > /dev/null
            SKIPPED=$((SKIPPED + 1))
            return
        fi

        if ! make -j$(nproc) >> /tmp/torch_gcu_test_libtorch_build.log 2>&1; then
            echo -e "${RED}[SKIP]${NC} Build failed:"
            tail -30 /tmp/torch_gcu_test_libtorch_build.log
            popd > /dev/null
            SKIPPED=$((SKIPPED + 1))
            return
        fi
        popd > /dev/null
        bin_dir="$build_dir"
        echo -e "${GREEN}[OK]${NC} Build successful"
    fi

    # Generate resnet50 model
    if [ ! -f "$bin_dir/resnet50_traced.pt" ]; then
        echo -e "${YELLOW}[INFO]${NC} Generating resnet50_traced.pt..."
        pushd "$bin_dir" > /dev/null
        python3 "$libtorch_dir/gen_resnet50.py" || true
        popd > /dev/null
    fi

    # Run test binaries (check executable and not a script/source file)
    local found_tests=0
    for test_bin in "$bin_dir"/test_*; do
        if [ -x "$test_bin" ] && [ -f "$test_bin" ] && [[ ! "$test_bin" =~ \.(py|sh|cc|cpp|h|cmake|txt|log)$ ]]; then
            local test_name="libtorch/$(basename "$test_bin")"
            run_test "$test_name" "$test_bin" "$bin_dir"
            found_tests=1
        fi
    done

    if [ "$found_tests" -eq 0 ]; then
        echo -e "${YELLOW}[SKIP]${NC} No libtorch test binaries found"
        SKIPPED=$((SKIPPED + 1))
    fi
}

# =============================================================================
# CUDA-compat API Tests (Python)
# =============================================================================
run_cuda_tests() {
    echo ""
    echo "=============================="
    echo "  CUDA-compat API Tests"
    echo "=============================="

    local cuda_dir="$SCRIPT_DIR/cuda"
    if [ ! -d "$cuda_dir" ]; then
        echo -e "${YELLOW}[SKIP]${NC} cuda test directory not found"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    for test_file in "$cuda_dir"/test_*.py; do
        if [ -f "$test_file" ]; then
            local test_name="cuda/$(basename "$test_file" .py)"
            run_test "$test_name" "python3 $(basename "$test_file")" "$cuda_dir"
        fi
    done
}

# =============================================================================
# Main
# =============================================================================
echo "============================================"
echo "  torch_gcu Test Suite"
echo "============================================"
echo "  Test directory: $SCRIPT_DIR"
echo "  Date: $(date)"
echo "============================================"

TARGET="${1:-all}"

case "$TARGET" in
    op)
        run_op_tests
        ;;
    libtorch)
        # TODO: add libtorch_gcu tests
        echo "libtorch_gcu tests are not supported yet"
        ;;
    cuda)
        run_cuda_tests
        ;;
    all)
        run_op_tests
        # TODO: add libtorch_gcu tests
        echo "libtorch_gcu tests are not supported yet"
        run_cuda_tests
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: $0 [all|op|libtorch|cuda]"
        exit 1
        ;;
esac

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================"
echo "  Test Summary"
echo "============================================"
echo -e "  ${GREEN}PASSED${NC}: $PASSED"
echo -e "  ${RED}FAILED${NC}: $FAILED"
echo -e "  ${YELLOW}SKIPPED${NC}: $SKIPPED"

if [ $FAILED -gt 0 ]; then
    echo -e "\n  Failed tests:${FAILED_TESTS}"
    echo ""
    echo "============================================"
    exit 1
else
    echo ""
    echo -e "  ${GREEN}All tests passed!${NC}"
    echo "============================================"
    exit 0
fi
