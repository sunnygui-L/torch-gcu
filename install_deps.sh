#!/bin/bash

build_path="build/Release"
if [ "$DEBUG" == "1" ]; then
    build_path="build/Debug"
fi

# check sudo
if command -v sudo &> /dev/null; then
    sudo_it() {
        sudo "$@"
    }
else
    sudo_it() {
        "$@"
    }
fi

# python
PYTHON_VERSION=${1:-3.10}
PYTHON_CMD="python$PYTHON_VERSION"
echo "PYTHON_CMD: ${PYTHON_CMD}"

# topsruntime,topstx,topscc,topspti
sudo_it bash $build_path/_deps/prebuild_caps_deb*-src/TopsPlatform_*.run -y -C topsruntime,topstx,topscc,topspti

# eccl
sudo_it dpkg -i $build_path/eccl_*_amd64.deb

# topsaten
sudo_it dpkg -i $build_path/topsaten_*_amd64.deb
