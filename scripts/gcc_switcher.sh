#!/bin/bash

compilerversion="-14"
cudagccversion="-14"
compiler="gcc"    
if [ "$compiler" = "clang" ]; then
    export CC="/usr/bin/clang${compilerversion}"
    export CXX="/usr/bin/clang++${compilerversion}"
    export CMAKE_C_COMPILER="/usr/bin/clang${compilerversion}"
    export CMAKE_CXX_COMPILER="/usr/bin/clang++${compilerversion}"
    export CUDAHOSTCXX="/usr/bin/g++${cudagccversion}"
    export CUDA_HOST_COMPILER="/usr/bin/g++${cudagccversion}"
    export CMAKE_CUDA_HOST_COMPILER="/usr/bin/g++${cudagccversion}"
    export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++${cudagccversion}"
else
    export CC="/usr/bin/gcc${compilerversion}"
    export CXX="/usr/bin/g++${compilerversion}"
    export CMAKE_C_COMPILER="/usr/bin/gcc${compilerversion}"
    export CMAKE_CXX_COMPILER="/usr/bin/g++${compilerversion}"
    export CUDAHOSTCXX="/usr/bin/g++${compilerversion}"
    export CUDA_HOST_COMPILER="/usr/bin/g++${compilerversion}"
    export CMAKE_CUDA_HOST_COMPILER="/usr/bin/g++${compilerversion}"
    export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++${compilerversion}"
fi