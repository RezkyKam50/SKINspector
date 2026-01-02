export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCH=89
cd 3rdparty/TransformerEngine
rm -rf build && rm -rf *.egg-info
pip install . --no-build-isolation