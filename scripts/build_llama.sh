# dependencies : ninja-build ccache cmake
# cuda toolkit dependencies for llama.cpp : 13.0

source ./scripts/gcc_switcher.sh
source ./scripts/cuda_toolkit.sh

./scripts/cuda_compile.sh