#!/bin/bash
set -eu -o pipefail
python3.8 -u gen_backend_stubs.py \
    --source_yaml ../gcu_native_functions.yaml \
    --output_dir ./tmp \
    --impl_path ../csrc/aten/GCUNativeFunctions.cpp
