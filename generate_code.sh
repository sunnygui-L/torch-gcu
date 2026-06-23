#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd $CDIR

python_execute="$1"
pytorch_version="$2"

IFS='.' read -ra version_parts <<< "$pytorch_version"

pytorch_dir="v${version_parts[0]}r${version_parts[1]}"

source_yaml="$CDIR/codegen/gcu_native_functions.yaml"

vision_yaml="$CDIR/codegen/vision_native_functions.yaml"

vision_source_yaml="$CDIR/codegen/gcu_vision_functions.yaml"

${python_execute} codegen/gen_yaml.py

${python_execute} codegen/gen_backend_stubs.py  \
  --output_dir="torch_gcu/csrc/aten/" \
  --source_yaml="$source_yaml" \
  --impl_path="$CDIR/torch_gcu/csrc/aten/GCUNativeFunctions.cpp" \
  --vision_yaml_path="$vision_yaml" \
  --vision_source_yaml="$vision_source_yaml"
sed -i '/void resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {/,/^}/ s/^/\/\//' torch_gcu/csrc/aten/RegisterGCU.cpp
sed -i '/void check_inplace(const Tensor &self, IntArrayRef sizes, const TensorOptions &options) {/,/^}/ s/^/\/\//' torch_gcu/csrc/aten/RegisterGCU.cpp
sed -i '/void resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {/,/^}/ s/^/\/\//' torch_gcu/csrc/aten/RegisterOptionalGCU.cpp
sed -i '/void check_inplace(const Tensor &self, IntArrayRef sizes, const TensorOptions &options) {/,/^}/ s/^/\/\//' torch_gcu/csrc/aten/RegisterOptionalGCU.cpp
